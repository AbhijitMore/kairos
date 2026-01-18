import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from kairos.data.transformers import AdultFeatureEngineer
from kairos.core.models import HybridEnsemble

from kairos.utils.logging import setup_kairos_logger

logger = setup_kairos_logger("kairos.core.pipeline")


def create_kairos_pipeline(config):
    """
    Factory function to create the unified KAIROS pipeline.
    """
    # 1. Feature Engineering
    feature_engineer = AdultFeatureEngineer()

    # 2. Preprocessing
    # These names are now used to dynamically find columns in the ensemble
    categorical_features = [
        "workclass",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    numeric_features = [
        "age",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "capital_net",
        "age_bin",
        "hours_per_edu",
        "age_edu",
        "hrs_edu",
        "cap_gain_tax",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # 3. Ensemble Model
    # Indices are now handled internally by HybridEnsemble using name/type detection
    ensemble = HybridEnsemble(
        lgb_params=config["model"]["lgbm_params"],
        cat_params=config["model"]["catboost_params"],
        n_folds=config["model"]["n_folds"],
    )

    full_pipeline = Pipeline(
        steps=[
            ("engineer", feature_engineer),
            ("preprocess", preprocessor),
            ("classifier", ensemble),
        ]
    )

    return full_pipeline


logger = logging.getLogger(__name__)


class KairosInferenceEngine:
    """
    High-level engine that manages the pipeline and post-hoc calibration.
    Uses native serialization for high-performance model components.
    """

    def __init__(self, pipeline, calibrator=None):
        self.pipeline = pipeline
        self.calibrator = calibrator

    def predict_calibrated(self, X):
        """Full inference flow: Raw Input -> Calibrated Probabilities."""
        # Use numpy underlying data if possible for faster pipeline throughput
        raw_probs = self.pipeline.predict_proba(X)[:, 1]

        if self.calibrator:
            # Reshape for sklearn calibrator
            cal_probs = self.calibrator.predict(raw_probs.reshape(-1, 1))
            return cal_probs
        return raw_probs

    def save(self, path: str):
        """
        Decomposes the engine into stable components.
        """
        import joblib
        import os

        os.makedirs(path, exist_ok=True)

        # 1. Save Transformers (Preprocessing)
        # We split the pipeline to save the Ensemble natively
        ensemble = self.pipeline.named_steps["classifier"]
        preprocess_pipe = Pipeline(
            steps=[
                ("engineer", self.pipeline.named_steps["engineer"]),
                ("preprocess", self.pipeline.named_steps["preprocess"]),
            ]
        )
        joblib.dump(preprocess_pipe, os.path.join(path, "preprocessor.joblib"))

        # 2. Save Ensemble Natively
        ensemble.save(os.path.join(path, "ensemble"))

        # 3. Save Calibrator
        if self.calibrator:
            joblib.dump(self.calibrator, os.path.join(path, "calibrator.joblib"))

    @classmethod
    def load(cls, path: str):
        """
        Reconstructs the engine from stable components.
        Includes compatibility layer for legacy models pickled with 'src.kairos' imports.
        """
        import joblib
        import os
        import pickle  # nosec B403 - only used for loading trusted model files

        # Custom unpickler to remap 'src.kairos' -> 'kairos' (for legacy models)
        class LegacyUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("src.kairos"):
                    module = module.replace("src.kairos", "kairos")
                return super().find_class(module, name)

        # Helper function to load with fallback to legacy unpickler
        def load_with_compat(filepath):
            try:
                # Try standard joblib loading first (for new models)
                return joblib.load(filepath)
            except (ModuleNotFoundError, AttributeError) as e:
                # Fall back to legacy unpickler if module not found
                if "src.kairos" in str(e) or "src" in str(e):
                    logger.info(f"Loading legacy model format from {filepath}")
                    with open(filepath, "rb") as f:
                        return LegacyUnpickler(f).load()
                raise

        # 1. Load Preprocessor with compatibility
        preprocess_pipe = load_with_compat(os.path.join(path, "preprocessor.joblib"))

        # 2. Load Ensemble Natively
        ensemble = HybridEnsemble.load(os.path.join(path, "ensemble"))

        # Check version
        meta_path = os.path.join(path, "ensemble", "metadata.joblib")
        if os.path.exists(meta_path):
            meta = load_with_compat(meta_path)
            version = meta.get("version", "legacy")
            logger.info(f"🦅 Model Loaded: HybridEnsemble v{version}")

        # 3. Reconstruct full pipeline
        full_pipeline = Pipeline(
            steps=[*preprocess_pipe.steps, ("classifier", ensemble)]
        )

        # 4. Load Calibrator
        calibrator = None
        cal_path = os.path.join(path, "calibrator.joblib")
        if os.path.exists(cal_path):
            calibrator = load_with_compat(cal_path)

        return cls(full_pipeline, calibrator)
