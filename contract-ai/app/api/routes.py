"""
API route definitions.

Endpoints
---------
GET  /api/v1/health   → HealthResponse
POST /api/v1/predict  → PredictResponse
POST /api/v1/train    → TrainResponse (launches background training task)
"""
import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, HTTPException, status, BackgroundTasks

from app.api.schema import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from app.core.config import settings
from app.core.model_loader import model_registry
from app.services.inference import predict_clause

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------ #
# Health check                                                         #
# ------------------------------------------------------------------ #

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["system"],
)
async def health_check() -> HealthResponse:
    """
    Returns the health status of the API and whether the model is loaded.
    Useful for Kubernetes liveness / readiness probes.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_registry.is_loaded,
        version=settings.app_version,
    )


# ------------------------------------------------------------------ #
# Predict                                                              #
# ------------------------------------------------------------------ #

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a contract clause",
    tags=["inference"],
)
async def predict(body: PredictRequest) -> PredictResponse:
    """
    Classify a single contract clause into a legal category.

    - **text**: Raw clause text (5 – 10 000 chars)

    Returns the predicted **label** and the model's **confidence** (0–1).
    """
    if not model_registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded yet. "
                "Either the server is still starting up or the model hasn't been trained. "
                "Call POST /api/v1/train first."
            ),
        )

    try:
        # Run CPU-bound inference in a thread to keep the event loop free
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, predict_clause, body.text)
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )

    logger.info(
        "POST /predict → label='%s' conf=%.4f latency=%.1f ms",
        result.label,
        result.confidence,
        result.latency_ms,
    )

    return PredictResponse(label=result.label, confidence=result.confidence)


# ------------------------------------------------------------------ #
# Train                                                                #
# ------------------------------------------------------------------ #

# Track if a training job is already running to prevent double-starts
_training_running: bool = False


def _run_training_sync(overrides: dict) -> None:
    """
    Thin wrapper executed in a background thread by BackgroundTasks.
    Imports training lazily to avoid slow startup on import.
    """
    global _training_running
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    _log = _logging.getLogger(__name__)
    try:
        from app.services.training import run_training
        results = run_training(**overrides)
        _log.info("Training finished. Results: %s", results)

        # Reload the model in the registry
        from app.core.model_loader import model_registry as reg
        reg.unload()
        reg.load()
        _log.info("Model registry refreshed with newly trained weights.")
    except Exception as exc:
        _log.exception("Training failed: %s", exc)
    finally:
        _training_running = False


@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Launch model fine-tuning",
    tags=["training"],
)
async def train(
    body: TrainRequest,
    background_tasks: BackgroundTasks,
) -> TrainResponse:
    """
    Start fine-tuning DistilBERT on the HuggingFace dataset in the background.

    Training runs asynchronously — use GET /health to know when the model is ready.

    **Only one training job can run at a time.** A 409 is returned if training is
    already in progress.
    """
    global _training_running

    if _training_running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A training job is already running. Please wait for it to finish.",
        )

    overrides = {
        "num_train_epochs": body.num_train_epochs,
        "per_device_train_batch_size": body.per_device_train_batch_size,
        "learning_rate": body.learning_rate,
        "test_size": body.test_size,
    }

    _training_running = True
    background_tasks.add_task(_run_training_sync, overrides)

    logger.info("Training job queued with overrides: %s", overrides)

    return TrainResponse(
        status="training_started",
        message=(
            "Fine-tuning started in the background. "
            "Check GET /api/v1/health — when model_loaded=true the new model is ready."
        ),
    )
