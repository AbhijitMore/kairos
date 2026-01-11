import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Any
import logging
import os
import joblib

logger = logging.getLogger(__name__)

class HybridEnsemble(BaseEstimator, ClassifierMixin):
    """
    SOTA Probabilistic Ensemble combining LightGBM and CatBoost.
    Refactored to support native model serialization.
    """
    def __init__(self, lgb_params: Dict[str, Any], cat_params: Dict[str, Any], n_folds: int = 5):
        self.lgb_params = lgb_params
        self.cat_params = cat_params
        self.n_folds = n_folds
        self.models: List[Tuple[str, Any]] = []
        self.classes_ = np.array([0, 1])
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: Any, y: Any):
        self.models = []
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = [str(c) for c in X.columns.tolist()]
            X_data = X
        else:
            X_data = pd.DataFrame(X)
            self.feature_names_ = [str(c) for c in X_data.columns.tolist()]

            
        y_data = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        # Determine categorical features by name for robustness
        cat_features = X_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        logger.info(f"Training Hybrid Ensemble: {self.n_folds} folds")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
            X_tr, X_vl = X_data.iloc[train_idx], X_data.iloc[val_idx]
            y_tr, y_vl = y_data.iloc[train_idx], y_data.iloc[val_idx]
            
            # 1. LightGBM
            lgb_p = deepcopy(self.lgb_params)
            lgb_p['seed'] = 42 + fold
            
            # Use name-based categorical features
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names_, categorical_feature=cat_features)
            dval = lgb.Dataset(X_vl, label=y_vl, reference=dtrain, feature_name=self.feature_names_, categorical_feature=cat_features)
            
            bst_lgb = lgb.train(
                lgb_p, dtrain, 
                num_boost_round=2500, 
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=0)]
            )
            self.models.append(('lgb', bst_lgb))
            
            # 2. CatBoost
            bst_cat = CatBoostClassifier(**self.cat_params, random_seed=42+fold)
            bst_cat.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), use_best_model=True, cat_features=cat_features, verbose=False)
            self.models.append(('cat', bst_cat))
            
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        # Optimized for throughput: Use numpy directly if possible or minimize DF overhead
        if isinstance(X, pd.DataFrame):
            X_vals = X.values
        else:
            X_vals = X
            
        probs_accum = []
        for name, model in self.models:
            if name == 'lgb':
                # LightGBM predict on raw numpy is faster
                p = model.predict(X_vals, num_iteration=model.best_iteration)
            else:
                # CatBoost predict_proba on numpy is also fast
                p = model.predict_proba(X_vals)[:, 1]
            probs_accum.append(p)
            
        avg_prob = np.mean(probs_accum, axis=0)
        return np.vstack([1-avg_prob, avg_prob]).T
    
    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save(self, directory: str):
        """Native serialization for models."""
        os.makedirs(directory, exist_ok=True)
        for i, (name, model) in enumerate(self.models):
            path = os.path.join(directory, f"model_{i}_{name}.bin")
            if name == 'lgb':
                model.save_model(path)
            else:
                model.save_model(path, format="cbm")
        
        # Save metadata (feature names, classes) as JSON
        meta = {
            'feature_names': self.feature_names_,
            'classes': self.classes_.tolist(),
            'model_types': [m[0] for m in self.models],
            'version': '1.0.0'  # FAANG-Grade Versioning
        }
        joblib.dump(meta, os.path.join(directory, "metadata.joblib"))

    @classmethod
    def load(cls, directory: str):
        """Reconstruct ensemble from native artifacts."""
        meta = joblib.load(os.path.join(directory, "metadata.joblib"))
        instance = cls(lgb_params={}, cat_params={}) # Empty params as models are pretrained
        instance.feature_names_ = meta['feature_names']
        instance.classes_ = np.array(meta['classes'])
        instance.models = []
        
        for i, name in enumerate(meta['model_types']):
            path = os.path.join(directory, f"model_{i}_{name}.bin")
            if name == 'lgb':
                model = lgb.Booster(model_file=path)
            else:
                model = CatBoostClassifier().load_model(path, format="cbm")
            instance.models.append((name, model))
        return instance

