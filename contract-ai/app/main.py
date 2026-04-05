"""
FastAPI application entry-point.

Startup sequence:
  1. FastAPI app created with metadata
  2. Lifespan context manager attempts to load the fine-tuned model.
     → If the model directory does not exist yet (first run before training),
       the app still starts but /predict will return 503 until training completes.
  3. API router mounted at settings.api_prefix (/api/v1)
"""
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.model_loader import model_registry, ModelNotFoundError

# ------------------------------------------------------------------ #
# Logging                                                              #
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Lifespan (startup + shutdown)                                        #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load the ML model into memory once when the server starts.
    Release resources when the server shuts down.
    """
    logger.info("=== %s v%s starting up ===", settings.app_name, settings.app_version)
    try:
        model_registry.load()
        logger.info("✅ Model loaded successfully — API is ready.")
    except ModelNotFoundError as exc:
        logger.warning(
            "⚠️  Model not found at startup (%s). "
            "The server will run but /predict will return 503 until training finishes. "
            "Call POST /api/v1/train to start training.",
            exc,
        )
    except Exception as exc:
        logger.error("❌ Unexpected error loading model: %s", exc, exc_info=True)

    yield  # Application runs here

    logger.info("=== Shutting down — releasing model resources ===")
    model_registry.unload()


# ------------------------------------------------------------------ #
# FastAPI app                                                          #
# ------------------------------------------------------------------ #

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-ready API for classifying contract clauses using a "
        "fine-tuned DistilBERT model. Supports training, inference, and health checks."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — tighten origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes under /api/v1
app.include_router(router, prefix=settings.api_prefix)


# ------------------------------------------------------------------ #
# Entrypoint for `uv run python app/main.py`                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
