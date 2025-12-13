# src/evaluation/reporting.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.metrics import classification_report, confusion_matrix


def build_classification_report(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a detailed classification report.

    This function wraps scikit-learn's ``classification_report`` and returns
    a dictionary that can be easily serialized to JSON or logged. It supports
    optional human-readable label names via ``label_names``.

    Args:
        y_true: Ground truth labels as integer class indices.
        y_pred: Predicted labels as integer class indices.
        label_names: Optional list of human-readable names for each label. If
            provided, its length must match the number of unique labels.

    Returns:
        A dictionary containing per-class precision, recall, f1-score, and
        support, as well as aggregated metrics such as accuracy, macro avg,
        and weighted avg.

    Raises:
        ValueError: If the lengths of ``y_true`` and ``y_pred`` do not match.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch between y_true ({len(y_true)}) and "
            f"y_pred ({len(y_pred)})."
        )

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    # classification_report returns a nested dict with keys that are either
    # class indices (as strings) or label names, plus 'accuracy', 'macro avg',
    # and 'weighted avg'. This structure is already JSON-serializable.
    return report


def build_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    *,
    normalize: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a confusion matrix.

    Args:
        y_true: Ground truth labels as integer class indices.
        y_pred: Predicted labels as integer class indices.
        normalize: Normalization mode for the confusion matrix. Valid options
            are ``None`` (counts), ``'true'``, ``'pred'``, or ``'all'``, as
            defined by scikit-learn.

    Returns:
        A dictionary with:
            - ``"matrix"``: 2D list representing the confusion matrix.
            - ``"normalize"``: Normalization mode used.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch between y_true ({len(y_true)}) and "
            f"y_pred ({len(y_pred)})."
        )

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    return {
        "matrix": cm.tolist(),
        "normalize": normalize,
    }


def save_report_to_json(report: Dict[str, Any], path: str | Path) -> None:
    """Save a classification report or evaluation summary to a JSON file.

    Args:
        report: Dictionary containing evaluation results. This is typically
            the output of :func:`build_classification_report` or a custom
            aggregation of metrics.
        path: Destination path for the JSON file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
