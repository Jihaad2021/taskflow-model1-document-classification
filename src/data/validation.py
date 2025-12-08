# src/data/validation.py
from __future__ import annotations

from typing import Dict, List

from datasets import Dataset, DatasetDict  # type: ignore

from src.config import DataConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _ensure_required_splits(ds: DatasetDict) -> None:
    """Ensure that required splits are present in the dataset.

    Raises:
        ValueError: If 'train' or 'test' splits are missing.
    """
    required_splits = ("train", "test")
    missing = [split for split in required_splits if split not in ds]

    if missing:
        raise ValueError(
            f"Missing required dataset splits: {missing}. "
            "Expected at least 'train' and 'test' splits in DatasetDict."
        )


def _ensure_required_columns(ds: Dataset, split_name: str, cfg: DataConfig) -> None:
    """Ensure that required columns exist in a given split.

    Raises:
        ValueError: If text or label columns are missing.
    """
    cols = ds.column_names

    missing_cols: List[str] = []
    if cfg.text_column not in cols:
        missing_cols.append(cfg.text_column)
    if cfg.label_column not in cols:
        missing_cols.append(cfg.label_column)

    if missing_cols:
        raise ValueError(
            f"Missing required columns in '{split_name}' split: {missing_cols}. "
            f"Available columns: {cols}"
        )


def _check_non_empty_split(ds: Dataset, split_name: str) -> None:
    """Ensure that a split contains at least one row."""
    n_rows = len(ds)
    if n_rows == 0:
        raise ValueError(f"Split '{split_name}' is empty. At least one row is required.")
    logger.info("Split '%s' contains %d rows.", split_name, n_rows)


def _compute_label_stats(labels: List[int]) -> Dict[int, int]:
    """Compute frequency of each label in a split."""
    counts: Dict[int, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _validate_labels(train_ds: Dataset, cfg: DataConfig) -> None:
    """Validate label distribution in the training split.

    Checks:
        - At least 2 distinct classes.
        - Warns if label imbalance is extreme.
    """
    labels = list(train_ds[cfg.label_column])  # type: ignore[index]
    if not labels:
        raise ValueError("Training labels are empty.")

    label_counts = _compute_label_stats(labels)
    n_classes = len(label_counts)

    if n_classes < 2:
        raise ValueError(
            f"Training labels contain fewer than 2 classes. "
            f"Label counts: {label_counts}"
        )

    logger.info(
        "Training label distribution: %s",
        label_counts,
    )

    max_count = max(label_counts.values())
    min_count = min(label_counts.values())

    if min_count == 0:
        logger.warning(
            "Some labels in training set have zero samples. Label counts: %s",
            label_counts,
        )
    else:
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 10.0:
            logger.warning(
                "Severe label imbalance detected in training set. "
                "Max/min ratio = %.2f. Label counts: %s",
                imbalance_ratio,
                label_counts,
            )


def _validate_text_quality(ds: Dataset, split_name: str, cfg: DataConfig) -> None:
    """Perform basic text quality checks on the given split.

    Checks:
        - Warns if many rows have empty/whitespace-only text.
        - Logs average text length in tokens (approx, by whitespace split).
    """
    texts = ds[cfg.text_column]  # type: ignore[index]
    n_rows = len(texts)

    if n_rows == 0:
        # This is already checked elsewhere, but we guard anyway.
        return

    empty_count = 0
    total_tokens = 0

    for text in texts:
        if text is None:
            empty_count += 1
            continue

        if isinstance(text, str):
            stripped = text.strip()
            if not stripped:
                empty_count += 1
                continue

            # Rough token count using whitespace
            total_tokens += len(stripped.split())
        else:
            # Non-string text is treated as empty / invalid
            empty_count += 1

    empty_ratio = empty_count / n_rows

    if empty_ratio > 0.0:
        logger.warning(
            "Split '%s' contains %d/%d (%.2f%%) empty or invalid text rows.",
            split_name,
            empty_count,
            n_rows,
            empty_ratio * 100.0,
        )

    if total_tokens > 0:
        avg_tokens = total_tokens / max(1, (n_rows - empty_count))
        logger.info(
            "Split '%s' average text length (rough tokens): %.2f",
            split_name,
            avg_tokens,
        )


def validate_dataset(ds: DatasetDict, cfg: DataConfig) -> None:
    """Run structural and basic semantic validation on the dataset.

    This function is intended to be called once after ingestion and before
    preprocessing / tokenization.

    It will:
        - Ensure required splits ('train', 'test') exist.
        - Ensure required columns (text & label) exist in each split.
        - Ensure splits are non-empty.
        - Validate label distribution in the training split.
        - Perform basic text quality checks (empty texts, avg length logging).

    Raises:
        ValueError: If critical issues are found that should stop the pipeline.
    """
    logger.info("Starting dataset validation.")

    # 1) Check required splits
    _ensure_required_splits(ds)

    train_ds = ds["train"]
    test_ds = ds["test"]

    # 2) Check required columns for each split
    _ensure_required_columns(train_ds, "train", cfg)
    _ensure_required_columns(test_ds, "test", cfg)

    # 3) Check non-empty splits
    _check_non_empty_split(train_ds, "train")
    _check_non_empty_split(test_ds, "test")

    # 4) Validate labels on training split
    _validate_labels(train_ds, cfg)

    # 5) Text quality checks on both splits (logs + warnings only)
    _validate_text_quality(train_ds, "train", cfg)
    _validate_text_quality(test_ds, "test", cfg)

    logger.info("Dataset validation completed successfully.")
