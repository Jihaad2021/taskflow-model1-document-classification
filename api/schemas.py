# api/schemas.py

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for text classification inference."""
    texts: list[str] = Field(..., min_length=1, description="Input texts to classify.")


class PredictResponse(BaseModel):
    """Response schema for text classification inference."""
    labels: list[int] = Field(..., description="Predicted label ids per input text.")
    probabilities: list[list[float]] = Field(
        ..., description="Softmax probabilities per input text."
    )
    logits: list[list[float]] = Field(..., description="Raw logits per input text.")
