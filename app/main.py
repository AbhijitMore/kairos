from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security.api_key import APIKeyHeader
import pandas as pd
import numpy as np
import os
import logging
from contextlib import asynccontextmanager
from typing import List

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.schemas import BatchInferenceRequest, PredictionResponse
from app.config import settings
from src.kairos.core.pipeline import KairosInferenceEngine
from src.kairos.core.policy import KairosPolicy

# Observability: Prometheus
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KAIROS-API")

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


# Global state
class APIState:
    engine: KairosInferenceEngine = None
    policy: KairosPolicy = None


state = APIState()


# Custom Metrics
DECISION_COUNTER = Counter(
    "kairos_decisions_total", "Total number of decisions made by KAIROS", ["decision"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for model and policy."""
    model_dir = settings.model_dir
    if os.path.exists(model_dir):
        logger.info(f"Loading KAIROS Engine from {model_dir}...")
        state.engine = KairosInferenceEngine.load(model_dir)

        # Initialize Policy from Settings (Environment) or Config (YAML)
        tau_low = settings.tau_low
        tau_high = settings.tau_high

        state.policy = KairosPolicy(tau_low=tau_low, tau_high=tau_high)
        logger.info(
            f">>> Engine and Policy initialized. Thresholds: {tau_low}/{tau_high}"
        )
    else:
        logger.warning(">>> Engine artifact not found. API in restricted mode.")
    yield
    state.engine = None
    state.policy = None


app = FastAPI(
    title="KAIROS: Risk Decision Service",
    description="Probabilistic inference service for risk assessment.",
    version="2.1.0",
    lifespan=lifespan,
)


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == settings.api_key:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")


def get_engine():
    if state.engine is None or state.policy is None:
        raise HTTPException(status_code=503, detail="Service not ready.")
    return state.engine, state.policy


@app.post("/predict", response_model=List[PredictionResponse])
@limiter.limit(settings.rate_limit)
async def predict(
    request: Request,
    inference_request: BatchInferenceRequest,
    deps=Depends(get_engine),
    _api_key: str = Depends(get_api_key),
):
    """
    Executes decision logic.
    Accepts raw instances, applies pipeline transformations, and returns policy decisions.
    """
    engine, policy = deps

    def sanitize_val(v):
        if not np.isfinite(v):
            return 0.5
        return float(v)

    try:
        # Convert to DataFrame (Minimal overhead for batching)
        X = pd.DataFrame([inst.model_dump() for inst in inference_request.instances])

        # 1. Probabilistic Inference
        cal_probs = engine.predict_calibrated(X)

        # 2. Vectorized Decision Making (Decoupled Policy)
        results = []
        for prob in cal_probs:
            prob_val = sanitize_val(prob)
            decision = policy.decide(prob_val)

            # Uncertainty: 0.0 at p=0/1, 1.0 at p=0.5
            uncertainty = sanitize_val(1.0 - 2.0 * abs(prob_val - 0.5))

            results.append(
                PredictionResponse(
                    decision=decision,
                    probability=prob_val,
                    uncertainty=uncertainty,
                    cost_risk=float(uncertainty * 100),
                )
            )

            # Increment Metric
            DECISION_COUNTER.labels(decision=decision).inc()

        return results

    except Exception as e:
        logger.error(f"Inference Fault: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error.")


@app.get("/health")
async def health():
    return {"status": "healthy", "engine_ready": state.engine is not None}


# Instrumentation
Instrumentator().instrument(app)
