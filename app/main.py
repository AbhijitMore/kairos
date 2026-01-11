from fastapi import FastAPI, HTTPException, Depends
import joblib
import pandas as pd
import numpy as np
import os
import logging
from contextlib import asynccontextmanager
from typing import List

from app.schemas import BatchInferenceRequest, PredictionResponse
from src.kairos.utils.config_loader import load_config
from src.kairos.core.pipeline import KairosInferenceEngine, KairosPolicy

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KAIROS-API")
CONFIG = load_config()

# Global state
class APIState:
    engine: KairosInferenceEngine = None
    policy: KairosPolicy = None

state = APIState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for model and policy."""
    model_dir = "outputs/kairos_model"
    if os.path.exists(model_dir):
        logger.info(f"Loading KAIROS Engine from {model_dir}...")
        state.engine = KairosInferenceEngine.load(model_dir)
        
        # Initialize Policy from Config
        state.policy = KairosPolicy(
            tau_low=CONFIG['policy']['scenario_k']['tau_low'],
            tau_high=CONFIG['policy']['scenario_k']['tau_high']
        )
        logger.info(">>> Engine and Policy initialized.")
    else:
        logger.warning(">>> Engine artifact not found. API in restricted mode.")
    yield
    state.engine = None
    state.policy = None


app = FastAPI(
    title="KAIROS: Risk Decision Service",
    description="Probabilistic inference service for risk assessment.",
    version="2.1.0",
    lifespan=lifespan
)

def get_engine():
    if state.engine is None or state.policy is None:
        raise HTTPException(status_code=503, detail="Service not ready.")
    return state.engine, state.policy

@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: BatchInferenceRequest, deps=Depends(get_engine)):
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
        X = pd.DataFrame([inst.model_dump() for inst in request.instances])
        
        # 1. Probabilistic Inference
        cal_probs = engine.predict_calibrated(X)
        
        # 2. Vectorized Decision Making (Decoupled Policy)
        results = []
        for prob in cal_probs:
            prob_val = sanitize_val(prob)
            decision = policy.decide(prob_val)
            
            # Uncertainty: 0.0 at p=0/1, 1.0 at p=0.5
            uncertainty = sanitize_val(1.0 - 2.0 * abs(prob_val - 0.5))
            
            results.append(PredictionResponse(
                decision=decision,
                probability=prob_val,
                uncertainty=uncertainty,
                cost_risk=float(uncertainty * 100)
            ))
            
        return results

    except Exception as e:
        logger.error(f"Inference Fault: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error.")

@app.get("/health")
async def health():
    return {"status": "healthy", "engine_ready": state.engine is not None}

