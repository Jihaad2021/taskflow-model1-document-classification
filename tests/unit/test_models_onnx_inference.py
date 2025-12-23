# tests/unit/test_models_onnx_inference.py

from pathlib import Path

import onnxruntime as ort
import pytest

from src.models.onnx_export import export_sequence_classification_to_onnx
from src.models.onnx_inference import OnnxSequenceClassifier


class TestOnnxSequenceClassifier:
    """Unit tests for ONNX inference wrapper."""

    @pytest.mark.slow
    def test_predict_returns_expected_shapes(self, tmp_path: Path) -> None:
        model_name = "hf-internal-testing/tiny-random-distilbert"
        onnx_path = tmp_path / "onnx" / "tiny.onnx"

        export_sequence_classification_to_onnx(
            model_name_or_path=model_name,
            output_path=onnx_path,
            max_length=16,
            opset=13,
        )

        clf = OnnxSequenceClassifier.from_pretrained(
            onnx_path=onnx_path,
            tokenizer_name_or_path=model_name,
            max_length=16,
            providers=["CPUExecutionProvider"],
        )

        pred = clf.predict(["hello world", "breaking news"])

        assert len(pred.labels) == 2
        assert len(pred.logits) == 2
        assert len(pred.probabilities) == 2

        # Number of classes should be consistent across logits/probabilities.
        assert len(pred.logits[0]) == len(pred.probabilities[0])
        assert len(pred.logits[1]) == len(pred.probabilities[1])
