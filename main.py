import logging
import argparse
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.kairos.utils.config_loader import load_config
from src.kairos.data.loader import load_adult_data
from src.kairos.core.pipeline import create_kairos_pipeline, KairosInferenceEngine
from src.kairos.tracking.mlflow_manager import MLflowManager
from src.kairos.tuning.optuna_hpo import OptunaHPO
from src.kairos.tracking.metrics import evaluate_probabilistic_model
from src.kairos.core.calibration import calibrate_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KAIROS-MAIN")

def main(args):
    config = load_config()
    mlflow_mgr = MLflowManager(config['project']['experiment_name'])
    
    logger.info(">>> Loading Raw Data")
    df = load_adult_data()
    X = df.drop(columns=['target', 'income', 'fnlwgt', 'education'])
    y = df['target']
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)

    if args.hpo:
        logger.info(">>> Starting Hyperparameter Optimization")
        hpo = OptunaHPO(config, X_train, y_train)
        best_params = hpo.run_hpo(n_trials=args.trials)
        # Update config with best params
        for k, v in best_params.items():
            config['model']['lgbm_params'][k] = v

    logger.info(">>> Initializing Unified Pipeline")
    pipeline = create_kairos_pipeline(config)
    
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
    os.makedirs("outputs", exist_ok=True)
    engine.save("outputs/kairos_model")

    
    # Log to MLflow
    mlflow_mgr.log_run(
        params=config['model']['lgbm_params'],
        metrics={
            "accuracy": metrics['accuracy'],
            "roc_auc": metrics['roc_auc'],
            "log_loss": metrics['log_loss']
        },
        model_pipeline=pipeline,
        artifacts={"calibration_curve": "outputs/calibration_curve.png"} if os.path.exists("outputs/calibration_curve.png") else None
    )
    
    logger.info(">>> Pipeline Modernization Complete. Artifacts stored in outputs/ and MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAIROS Modernized Pipeline")
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=10, help="Number of HPO trials")
    args = parser.parse_args()
    main(args)
