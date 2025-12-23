# api/app.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.schemas import PredictRequest, PredictResponse
from src.models.onnx_inference import OnnxSequenceClassifier


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_path: str


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value}") from exc


def _resolve_model_path() -> Path:
    """Resolve ONNX model path from environment variables.

    Priority:
      1) MODEL1_ONNX_PATH
      2) Default: artifacts/onnx_bench/optimized.onnx (local dev)
    """
    raw = os.getenv("MODEL1_ONNX_PATH", "artifacts/onnx_bench/optimized.onnx")
    return Path(raw)


def _resolve_tokenizer_path() -> str:
    """Resolve tokenizer name or local path from environment variables.

    Priority:
      1) MODEL1_TOKENIZER_NAME_OR_PATH
      2) Default: hf-internal-testing/tiny-random-distilbert (local dev)
    """
    return os.getenv("MODEL1_TOKENIZER_NAME_OR_PATH", "hf-internal-testing/tiny-random-distilbert")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="TaskFlow Model 1 - Inference API", version="0.1.0")

    model_path = _resolve_model_path()
    tokenizer_name_or_path = _resolve_tokenizer_path()
    max_length = _get_env_int("MODEL1_MAX_LENGTH", 16)

    # Lazy-initialized singleton
    clf: Optional[OnnxSequenceClassifier] = None

    @app.on_event("startup")
    def _startup() -> None:
        nonlocal clf
        if not model_path.exists():
            raise RuntimeError(f"ONNX model file not found: {model_path}")

        clf = OnnxSequenceClassifier.from_pretrained(
            onnx_path=model_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_length=max_length,
            providers=["CPUExecutionProvider"],
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_path=model_path.as_posix())

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        nonlocal clf
        if clf is None:
            raise HTTPException(status_code=503, detail="Model is not initialized.")

        try:
            pred = clf.predict(req.texts)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PredictResponse(
            labels=pred.labels,
            probabilities=pred.probabilities,
            logits=pred.logits,
        )

    return app


app = create_app()
