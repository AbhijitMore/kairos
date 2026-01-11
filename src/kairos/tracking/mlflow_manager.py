import mlflow
import mlflow.sklearn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Handles logging of parameters, metrics, and artifacts to MLflow.
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model_pipeline: Any,
        artifacts: Dict[str, str] = None,
    ):
        """
        Logs a full training run.
        """
        with mlflow.start_run():
            # Log Parameters
            mlflow.log_params(params)

            # Log Metrics
            mlflow.log_metrics(metrics)

            # Log Model
            mlflow.sklearn.log_model(model_pipeline, "model")

            # Log Artifacts (Plots, CSVs)
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path)

            logger.info(
                f"Run logged to MLflow under experiment: {self.experiment_name}"
            )
