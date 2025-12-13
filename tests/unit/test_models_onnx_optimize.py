# tests/unit/test_models_onnx_optimize.py

from pathlib import Path

import onnxruntime as ort
import pytest

from src.models.onnx_export import export_sequence_classification_to_onnx
from src.models.onnx_optimize import optimize_onnx_with_ort, quantize_onnx_dynamic_int8


class TestOnnxOptimize:
    """Unit tests for ONNX optimization utilities."""

    @pytest.mark.slow
    def test_optimize_onnx_with_ort_creates_file(self, tmp_path: Path) -> None:
        model_name = "hf-internal-testing/tiny-random-distilbert"
        baseline = tmp_path / "onnx" / "baseline.onnx"
        optimized = tmp_path / "onnx" / "optimized.onnx"

        export_sequence_classification_to_onnx(
            model_name_or_path=model_name,
            output_path=baseline,
            max_length=16,
            opset=13,
        )

        out_path = optimize_onnx_with_ort(baseline, optimized)

        assert out_path.exists()
        # Ensure ORT can load the optimized model
        _ = ort.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])

    @pytest.mark.slow
    def test_quantize_dynamic_int8_creates_file(self, tmp_path: Path) -> None:
        model_name = "hf-internal-testing/tiny-random-distilbert"
        baseline = tmp_path / "onnx" / "baseline.onnx"
        int8_path = tmp_path / "onnx" / "int8.onnx"

        export_sequence_classification_to_onnx(
            model_name_or_path=model_name,
            output_path=baseline,
            max_length=16,
            opset=13,
        )

        out_path = quantize_onnx_dynamic_int8(baseline, int8_path)

        assert out_path.exists()
        _ = ort.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])
