from typing import Dict, TYPE_CHECKING

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    # Optional import: only needed when integrating with HF Trainer
    from transformers import EvalPrediction
except ImportError:  # pragma: no cover - transformers may not be available in some environments
    EvalPrediction = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from transformers import EvalPrediction
    
def compute_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute standard classification metrics given logits and labels.

    This function assumes a multi-class classification setting and
    computes accuracy, precision, recall, and F1 (macro-averaged).

    Args:
        logits: Array of model outputs before softmax. Shape:
            - (num_samples, num_classes) for multi-class classification.
            - (num_samples,) or (num_samples, 1) for binary classification.
        labels: Array of ground-truth class indices. Shape: (num_samples,).

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy.
            - precision: Macro-averaged precision.
            - recall: Macro-averaged recall.
            - f1: Macro-averaged F1 score.

    Raises:
        ValueError: If the shapes of logits and labels are incompatible.
    """
    if logits.ndim == 1:
        # Binary case with a single logit per sample: convert to 2-class logits
        # by stacking negative and positive scores.
        logits = np.stack([-logits, logits], axis=-1)
    elif logits.ndim != 2:
        raise ValueError(
            f"Expected logits to have shape (num_samples, num_classes) or "
            f"(num_samples,), but got shape {logits.shape}."
        )

    if labels.ndim != 1:
        raise ValueError(
            f"Expected labels to have shape (num_samples,), but got shape {labels.shape}."
        )

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Number of samples in logits and labels must match. "
            f"Got {logits.shape[0]} logits and {labels.shape[0]} labels."
        )

    # Predicted class index is the argmax over class dimension
    preds = np.argmax(logits, axis=-1)

    accuracy = float(accuracy_score(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def hf_compute_classification_metrics(eval_pred: "EvalPrediction") -> Dict[str, float]:
    """HuggingFace Trainer-compatible metrics function.

    This is a thin wrapper around `compute_classification_metrics` that
    accepts a `transformers.EvalPrediction` object and returns metrics
    in the format expected by the Trainer API.

    Args:
        eval_pred: EvalPrediction object containing:
            - predictions: Model outputs (logits).
            - label_ids: Ground-truth class indices.

    Returns:
        Dictionary with the same keys as `compute_classification_metrics`:
            - accuracy
            - precision
            - recall
            - f1
    """
    logits = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids)

    return compute_classification_metrics(logits=logits, labels=labels)
