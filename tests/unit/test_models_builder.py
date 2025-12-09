# tests/unit/test_models_builder.py

import pytest
from transformers import PreTrainedModel

from src.config import ModelConfig
from src.models.builder import build_classification_model


TINY_MODEL_NAME = "hf-internal-testing/tiny-random-distilbert"


class TestBuildClassificationModel:
    """Test suite for the build_classification_model function."""

    def test_build_model_with_explicit_num_labels(self) -> None:
        """Model is built successfully when num_labels is provided explicitly."""
        model_cfg = ModelConfig(
            model_name=TINY_MODEL_NAME,
            max_length=128,
            num_labels=None,  # Intentionally None, we use the argument instead
        )

        model = build_classification_model(
            model_cfg=model_cfg,
            num_labels=4,
        )

        assert isinstance(model, PreTrainedModel)
        assert model.config.num_labels == 4

    def test_build_model_uses_config_num_labels_when_not_overridden(self) -> None:
        """If num_labels is not passed, it falls back to ModelConfig.num_labels."""
        model_cfg = ModelConfig(
            model_name=TINY_MODEL_NAME,
            max_length=128,
            num_labels=3,
        )

        model = build_classification_model(model_cfg=model_cfg)

        assert model.config.num_labels == 3

    def test_raise_error_when_num_labels_missing(self) -> None:
        """Raise ValueError when num_labels is missing from both arg and config."""
        model_cfg = ModelConfig(
            model_name=TINY_MODEL_NAME,
            max_length=128,
            num_labels=None,
        )

        with pytest.raises(ValueError):
            _ = build_classification_model(model_cfg=model_cfg)

    def test_label_mappings_are_applied(self) -> None:
        """id2label and label2id mappings are correctly applied to the config."""
        id2label = {0: "world", 1: "sports"}
        label2id = {"world": 0, "sports": 1}

        model_cfg = ModelConfig(
            model_name=TINY_MODEL_NAME,
            max_length=128,
            num_labels=2,
        )

        model = build_classification_model(
            model_cfg=model_cfg,
            id2label=id2label,
            label2id=label2id,
        )

        assert model.config.id2label == id2label
        assert model.config.label2id == label2id
