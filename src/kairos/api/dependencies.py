from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from kairos.api.config import settings
from kairos.core.pipeline import KairosInferenceEngine
from kairos.core.policy import KairosPolicy
from slowapi import Limiter
from slowapi.util import get_remote_address
import os
import logging

logger = logging.getLogger("kairos.api.deps")

# Rate Limiter (shared)
limiter = Limiter(key_func=get_remote_address)

# API Key Security
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


# Global shared state (Singleton-ish)
class APIState:
    _engines: dict[str, KairosInferenceEngine] = {}
    _policy: KairosPolicy = None
    _initialized: bool = False

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return

        # Load standard policy
        cls._policy = KairosPolicy(tau_low=settings.tau_low, tau_high=settings.tau_high)

        # Potential model locations (development and production)
        search_dirs = ["outputs", "models"]

        for base_dir in search_dirs:
            if not os.path.exists(base_dir):
                continue

            for folder in os.listdir(base_dir):
                if folder.endswith("_model"):
                    dataset_name = folder.replace("_model", "")
                    model_path = os.path.join(base_dir, folder)
                    try:
                        logger.info(
                            f"Loading KAIROS Engine for '{dataset_name}' from {model_path}..."
                        )
                        cls._engines[dataset_name] = KairosInferenceEngine.load(
                            model_path
                        )
                    except Exception as e:
                        logger.error(f"Failed to load engine for {dataset_name}: {e}")

        # Fallback for legacy/hardcoded paths from settings
        if not cls._engines and os.path.exists(settings.model_dir):
            try:
                logger.info(f"Loading fallback engine from {settings.model_dir}...")
                cls._engines["adult"] = KairosInferenceEngine.load(settings.model_dir)
            except Exception as e:
                logger.error(f"Fallback load failed: {e}")

        cls._initialized = True
        logger.info(f"🦅 Loaded {len(cls._engines)} inference engines.")

    @classmethod
    def get_engine(cls, dataset_name: str = "adult"):
        if not cls._initialized:
            cls.initialize()

        engine = cls._engines.get(dataset_name)
        if engine is None:
            # If specifically adult is missing, try picking the first available as fallback
            if dataset_name == "adult" and cls._engines:
                dataset_name = list(cls._engines.keys())[0]
                engine = cls._engines[dataset_name]
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Inference engine for '{dataset_name}' not ready or not found.",
                )

        return engine, cls._policy


# Dependencies
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == settings.api_key:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")


def get_inference_deps():
    return APIState.get_engine()
