"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Shared                                                              #
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(
        ..., description="True when the ML model is ready to serve predictions"
    )
    version: str = Field(..., examples=["1.0.0"])


# ------------------------------------------------------------------ #
# Predict                                                             #
# ------------------------------------------------------------------ #

class PredictRequest(BaseModel):
    """Request schema for POST /predict."""

    text: str = Field(
        ...,
        min_length=5,
        max_length=10_000,
        examples=["This agreement may be terminated by either party with 30 days notice."],
        description="Raw contract clause text to classify.",
    )


class PredictResponse(BaseModel):
    """Response schema for POST /predict."""

    label: str = Field(..., examples=["Termination"])
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax confidence score for the predicted label (0–1).",
    )


# ------------------------------------------------------------------ #
# Train                                                               #
# ------------------------------------------------------------------ #

class TrainRequest(BaseModel):
    """Optional overrides passed to POST /train."""

    num_train_epochs: int = Field(default=3, ge=1, le=20)
    per_device_train_batch_size: int = Field(default=16, ge=1, le=64)
    learning_rate: float = Field(default=2e-5, gt=0)
    test_size: float = Field(default=0.15, gt=0, lt=1)


class TrainResponse(BaseModel):
    """Response schema for POST /train."""

    status: str = Field(..., examples=["training_started"])
    message: str
