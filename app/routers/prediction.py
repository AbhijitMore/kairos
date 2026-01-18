from fastapi import APIRouter, HTTPException, Depends, Request
import pandas as pd
import numpy as np
import logging
from typing import List
from celery.result import AsyncResult

from app.schemas import BatchInferenceRequest, PredictionResponse
from app.dependencies import get_inference_deps, get_api_key, limiter
from app.tasks import celery_app, predict_batch_task
from prometheus_client import Counter

logger = logging.getLogger("kairos.api.prediction")
router = APIRouter(prefix="/predict", tags=["prediction"])

# Custom Metrics
DECISION_COUNTER = Counter(
    "kairos_decisions_total", "Total number of decisions made by KAIROS", ["decision"]
)


def sanitize_val(v):
    if not np.isfinite(v):
        return 0.5
    return float(v)


@router.post("", response_model=List[PredictionResponse])
@limiter.limit("500/minute")
async def predict(
    request: Request,
    inference_request: BatchInferenceRequest,
    deps=Depends(get_inference_deps),
    _api_key: str = Depends(get_api_key),
):
    """
    Executes decision logic for a batch of instances.
    """
    engine, policy = deps

    try:
        X = pd.DataFrame([inst.model_dump() for inst in inference_request.instances])
        cal_probs = engine.predict_calibrated(X)

        results = []
        for prob in cal_probs:
            prob_val = sanitize_val(prob)
            decision = policy.decide(prob_val)
            uncertainty = sanitize_val(1.0 - 2.0 * abs(prob_val - 0.5))

            results.append(
                PredictionResponse(
                    decision=decision,
                    probability=prob_val,
                    uncertainty=uncertainty,
                    cost_risk=float(uncertainty * 100),
                )
            )
            DECISION_COUNTER.labels(decision=decision).inc()

        return results

    except Exception as e:
        logger.error(f"Inference Fault: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error.")


@router.post("/batch/async")
async def predict_batch_async(
    inference_request: BatchInferenceRequest,
    _api_key: str = Depends(get_api_key),
):
    """
    Kicks off an asynchronous batch inference task.
    """
    records = [inst.model_dump() for inst in inference_request.instances]
    task = predict_batch_task.delay(records)

    return {
        "task_id": task.id,
        "status": "PENDING",
        "poll_url": f"/api/v1/predict/status/{task.id}",
    }


@router.get("/status/{task_id}")
async def get_task_status(task_id: str, _api_key: str = Depends(get_api_key)):
    """
    Polls the status of an asynchronous batch task.
    """
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}

    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)

    return response
