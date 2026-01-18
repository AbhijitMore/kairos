from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from app.config import settings
from src.kairos.core.pipeline import KairosInferenceEngine
from src.kairos.core.policy import KairosPolicy
import os
import logging

logger = logging.getLogger("kairos.api.deps")

# API Key Security
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


# Global shared state (Singleton-ish)
class APIState:
    _engine: KairosInferenceEngine = None
    _policy: KairosPolicy = None
    _initialized: bool = False

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return

        model_dir = settings.model_dir
        if os.path.exists(model_dir):
            logger.info(f"Loading KAIROS Engine from {model_dir}...")
            cls._engine = KairosInferenceEngine.load(model_dir)
            cls._policy = KairosPolicy(
                tau_low=settings.tau_low, tau_high=settings.tau_high
            )
            cls._initialized = True
            logger.info("🦅 Engine and Policy initialized.")
        else:
            logger.warning("Engine artifact not found. API in restricted mode.")

    @classmethod
    def get_engine(cls):
        if not cls._initialized:
            cls.initialize()
        if cls._engine is None:
            raise HTTPException(status_code=503, detail="Service not ready.")
        return cls._engine, cls._policy


# Dependencies
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == settings.api_key:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")


def get_inference_deps():
    return APIState.get_engine()
