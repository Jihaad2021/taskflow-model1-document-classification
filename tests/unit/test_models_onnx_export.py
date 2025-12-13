# tests/unit/test_models_onnx_export.py

from pathlib import Path

import pytest

from src.models.onnx_export import export_sequence_classification_to_onnx


class TestExportSequenceClassificationToOnnx:
    """Unit tests for export_sequence_classification_to_onnx."""

    @pytest.mark.slow
    def test_export_creates_onnx_file_with_tiny_model(self, tmp_path: Path) -> None:
        """Export should succeed and create an ONNX file for a tiny model.

        This test uses a tiny random DistilBERT model from the HuggingFace Hub
        to keep the export lightweight and fast. It only checks that the file
        is created successfully and does not validate the ONNX graph content.
        """
        model_name = "hf-internal-testing/tiny-random-distilbert"
        output_path = tmp_path / "onnx" / "tiny_model.onnx"

        exported_path = export_sequence_classification_to_onnx(
            model_name_or_path=model_name,
            output_path=output_path,
            max_length=16,
            opset=13,
        )

        assert exported_path.exists()
        assert exported_path.is_file()
