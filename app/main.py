from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator

from app.dependencies import APIState
from app.routers import prediction, monitoring
from src.kairos.utils.logging import setup_kairos_logger

# Standardized Logger
logger = setup_kairos_logger("kairos.api", level=logging.INFO)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management."""
    logger.info("🚀 KAIROS API is starting up...")
    APIState.initialize()
    yield
    logger.info("🛑 KAIROS API is shutting down...")


app = FastAPI(
    title="KAIROS: Risk Decision Service",
    description="Probabilistic inference service for risk assessment.",
    version="2.2.0",
    lifespan=lifespan,
)

# Exception Handlers
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Include Routers
# Versioned API
app.include_router(prediction.router, prefix="/api/v1")

# Monitoring (Root level for convenience)
app.include_router(monitoring.router)

# Legacy Redirect/Mapping (Optional: can be added if needed for compatibility)
# app.include_router(prediction.router) # Keep root /predict as well


@app.get("/")
async def root():
    return {
        "service": "KAIROS API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }
