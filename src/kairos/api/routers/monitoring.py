from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from kairos.api.dependencies import APIState

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health")
async def health():
    """
    Service health check. Does not raise 503.
    """
    ready_engines = list(APIState._engines.keys())
    return {
        "status": "healthy",
        "engines_loaded": ready_engines,
        "is_initialized": APIState._initialized,
    }
