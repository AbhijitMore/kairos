from celery import Celery
from typing import List, Dict, Any
import os
from datetime import datetime, UTC
import pandas as pd
from src.kairos.core.pipeline import KairosInferenceEngine

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


def get_engine():
    global _engine
    if _engine is None:
        model_path = os.getenv("MODEL_PATH", "outputs/kairos_model")
        _engine = KairosInferenceEngine.load(model_path)
    return _engine


@celery_app.task(name="kairos.predict_batch")
def predict_batch_task(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Background task to process a batch of records.
    """
    engine = get_engine()

    # Convert list of dicts to DataFrame for batch inference efficiency
    df = pd.DataFrame(records)

    # Run calibrated inference
    probs = engine.predict_calibrated(df)

    # Standard KAIROS policy application (could be made configurable later)
    # For now, we return probabilities and the user can apply policy or we can
    # import KairosPolicy here.

    return {
        "predictions": probs.tolist(),
        "count": len(records),
        "processed_at": datetime.now(UTC).isoformat(),
    }
