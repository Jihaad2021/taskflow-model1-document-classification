# tests/unit/test_evaluation_reporting.py

from pathlib import Path
from typing import Any, Dict

import json

from src.evaluation.reporting import (
    build_classification_report,
    build_confusion_matrix,
    save_report_to_json,
)


class TestBuildClassificationReport:
    """Unit tests for build_classification_report."""

    def test_binary_classification_report_with_label_names(self) -> None:
        """Report should include per-class metrics and aggregated metrics."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        label_names = ["negative", "positive"]

        report: Dict[str, Any] = build_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            label_names=label_names,
        )

        # Ensure per-class entries exist
        assert "negative" in report
        assert "positive" in report

        # Ensure aggregated metrics exist
        assert "accuracy" in report
        assert "macro avg" in report
        assert "weighted avg" in report

        # Sanity check on accuracy range
        accuracy = report["accuracy"]
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_length_mismatch_raises_value_error(self) -> None:
        """A ValueError should be raised if y_true and y_pred lengths differ."""
        y_true = [0, 1, 1]
        y_pred = [0, 1]

        try:
            _ = build_classification_report(y_true=y_true, y_pred=y_pred)
        except ValueError as exc:
            assert "Length mismatch" in str(exc)
        else:  # pragma: no cover - defensive branch
            raise AssertionError("Expected ValueError was not raised.")


class TestBuildConfusionMatrix:
    """Unit tests for build_confusion_matrix."""

    def test_confusion_matrix_counts(self) -> None:
        """Confusion matrix should match expected counts."""
        # Example:
        # y_true: 0, 0, 1, 1
        # y_pred: 0, 1, 0, 1
        #
        # Confusion matrix (rows = true, cols = pred):
        # [[1, 1],
        #  [1, 1]]
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]

        result = build_confusion_matrix(y_true=y_true, y_pred=y_pred)

        assert result["normalize"] is None
        matrix = result["matrix"]
        assert matrix == [
            [1, 1],
            [1, 1],
        ]

    def test_confusion_matrix_length_mismatch_raises_value_error(self) -> None:
        """A ValueError should be raised if y_true and y_pred lengths differ."""
        y_true = [0, 1, 1]
        y_pred = [0, 1]

        try:
            _ = build_confusion_matrix(y_true=y_true, y_pred=y_pred)
        except ValueError as exc:
            assert "Length mismatch" in str(exc)
        else:  # pragma: no cover - defensive branch
            raise AssertionError("Expected ValueError was not raised.")


class TestSaveReportToJson:
    """Unit tests for save_report_to_json."""

    def test_save_report_to_json_writes_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Report should be written to disk as valid JSON."""
        report: Dict[str, Any] = {
            "accuracy": 0.9,
            "macro avg": {"precision": 0.88, "recall": 0.9, "f1-score": 0.89, "support": 4},
        }

        output_path = tmp_path / "reports" / "classification_report.json"

        save_report_to_json(report, output_path)

        assert output_path.exists()

        with output_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == report
