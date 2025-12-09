# tests/unit/test_evaluation_metrics.py

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_classification_metrics,
    hf_compute_classification_metrics,
)


class DummyEvalPrediction:
    """Minimal stand-in for transformers.EvalPrediction used in tests."""

    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


class TestComputeClassificationMetrics:
    """Test suite for compute_classification_metrics."""

    def test_perfect_multiclass_classification(self) -> None:
        """Metrics are perfect (1.0) when predictions match labels exactly."""
        # 3 samples, 3 classes
        logits = np.array(
            [
                [10.0, 0.0, 0.0],  # predicted class 0
                [0.0, 5.0, 0.0],   # predicted class 1
                [0.0, 0.0, 3.0],   # predicted class 2
            ]
        )
        labels = np.array([0, 1, 2])

        metrics = compute_classification_metrics(logits=logits, labels=labels)

        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_binary_logits_vector_is_supported(self) -> None:
        """Binary classification with 1D logits is handled correctly."""
        # logits > 0 -> class 1, logits < 0 -> class 0 (after stacking)
        logits = np.array([-2.0, -1.0, 0.5, 3.0])
        labels = np.array([0, 0, 1, 1])

        metrics = compute_classification_metrics(logits=logits, labels=labels)

        # All predictions should be correct in this synthetic example
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_shape_mismatch_raises_error(self) -> None:
        """Raise ValueError when logits and labels have incompatible shapes."""
        logits = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 samples
        labels = np.array([0, 1, 2])  # 3 labels -> mismatch

        with pytest.raises(ValueError):
            _ = compute_classification_metrics(logits=logits, labels=labels)

    def test_invalid_logit_rank_raises_error(self) -> None:
        """Raise ValueError when logits have invalid rank."""
        logits = np.zeros((2, 2, 2))  # Rank-3 tensor (invalid)
        labels = np.array([0, 1])

        with pytest.raises(ValueError):
            _ = compute_classification_metrics(logits=logits, labels=labels)


class TestHfComputeClassificationMetrics:
    """Test suite for hf_compute_classification_metrics."""

    def test_wrapper_uses_same_logic_as_base_function(self) -> None:
        """HF wrapper returns the same metrics as the base function."""
        logits = np.array(
            [
                [2.0, 0.0],   # class 0
                [0.0, 3.0],   # class 1
                [-1.0, 1.0],  # class 1
            ]
        )
        labels = np.array([0, 1, 1])

        eval_pred = DummyEvalPrediction(predictions=logits, label_ids=labels)

        metrics = hf_compute_classification_metrics(eval_pred)

        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
