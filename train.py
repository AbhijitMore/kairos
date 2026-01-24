import logging
import argparse
import os
from sklearn.model_selection import train_test_split

from kairos.utils.config_loader import load_config
from kairos.data.loader import load_adult_data, load_home_credit_data
from kairos.core.pipeline import create_kairos_pipeline, KairosInferenceEngine
from kairos.tracking.mlflow_manager import MLflowManager
from kairos.tuning.optuna_hpo import OptunaHPO
from kairos.tracking.metrics import evaluate_probabilistic_model
from kairos.core.calibration import calibrate_model

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("KAIROS-MAIN")


def main(args):
    config = load_config()
    experiment_name = f"{config['project']['experiment_name']}-{args.dataset}"
    mlflow_mgr = MLflowManager(experiment_name)

    logger.info(f">>> Loading Raw Data for: {args.dataset}")
    if args.dataset == "adult":
        df = load_adult_data()
        X = df.drop(columns=["target", "income", "fnlwgt", "education"])
        y = df["target"]
    elif args.dataset == "home_credit":
        df = load_home_credit_data()
        # Home Credit specific drops (keep features defined in pipeline.py)
        X = df.drop(columns=["target", "TARGET"])
        y = df["target"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42
    )

    if args.hpo:
        logger.info(">>> Starting Hyperparameter Optimization")
        hpo = OptunaHPO(config, X_train, y_train)
        best_params = hpo.run_hpo(n_trials=args.trials)
        # Update config with best params
        for k, v in best_params.items():
            config["model"]["lgbm_params"][k] = v

    logger.info(f">>> Initializing Unified Pipeline for {args.dataset}")
    pipeline = create_kairos_pipeline(config, dataset_type=args.dataset)

    logger.info(">>> Training Pipeline (Pre-processing + Ensemble)")
    # Note: X_train is RAW data here. Preprocessing happens INSIDE fitting.
    pipeline.fit(X_train, y_train)

    logger.info(">>> Fitting Post-hoc Calibrator")
    # Calibration needs raw probabilities from the fitted pipeline on a validation set
    calibrator = calibrate_model(pipeline, X_val, y_val)

    engine = KairosInferenceEngine(pipeline, calibrator)

    logger.info(">>> Evaluating Performance")
    cal_probs = engine.predict_calibrated(X_test)
    metrics = evaluate_probabilistic_model(y_test, cal_probs)

    logger.info(f"Final Metrics: {metrics}")

    logger.info(">>> Saving Robust ML Artifacts")
    output_path = f"outputs/{args.dataset}_model"
    os.makedirs(output_path, exist_ok=True)
    engine.save(output_path)

    # Generate and Save plots
    from kairos.core.calibration import plot_reliability_diagram

    plot_reliability_diagram(
        y_test, cal_probs, save_path=f"outputs/{args.dataset}_calibration_curve.png"
    )

    # Log to MLflow
    mlflow_mgr.log_run(
        params=config["model"]["lgbm_params"],
        metrics={
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "log_loss": metrics["log_loss"],
        },
        model_pipeline=pipeline,
        artifacts={"calibration_curve": f"outputs/{args.dataset}_calibration_curve.png"}
        if os.path.exists(f"outputs/{args.dataset}_calibration_curve.png")
        else None,
    )

    logger.info(
        f">>> {args.dataset.upper()} Pipeline Modernization Complete. Artifacts in {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAIROS Modernized Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["adult", "home_credit"],
        default="adult",
        help="Dataset to train on",
    )
    parser.add_argument(
        "--hpo", action="store_true", help="Run hyperparameter optimization"
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of HPO trials")
    args = parser.parse_args()
    main(args)
