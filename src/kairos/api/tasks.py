from celery import Celery
from typing import List, Dict, Any
import os
from datetime import datetime, timezone
import pandas as pd
from kairos.core.pipeline import KairosInferenceEngine

# Initialize Celery
# Using environment variables for broker/backend to support Docker/Local flexibility
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("kairos", broker=REDIS_URL, backend=REDIS_URL)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Global engine placeholder for worker process
_engine = None


def get_engine(dataset_name: str = "adult"):
    """
    Loads a specific engine for the worker.
    Workers typically load models from the 'outputs' or 'models' directory.
    """
    # Look for the model in the standard outputs directory first
    model_path = f"outputs/{dataset_name}_model"
    if not os.path.exists(model_path):
        # Fallback to a models/ directory or env var
        model_path = os.getenv("MODEL_PATH", "outputs/adult_model")

    return KairosInferenceEngine.load(model_path)


@celery_app.task(name="kairos.predict_batch")
def predict_batch_task(
    records: List[Dict[str, Any]], dataset: str = "adult"
) -> Dict[str, Any]:
    """
    Background task to process a batch of records for a specific dataset.
    """
    engine = get_engine(dataset)

    # Convert list of dicts to DataFrame for batch inference efficiency
    df = pd.DataFrame(records)

    # Run calibrated inference
    probs = engine.predict_calibrated(df)

    return {
        "dataset": dataset,
        "predictions": probs.tolist(),
        "count": len(records),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
