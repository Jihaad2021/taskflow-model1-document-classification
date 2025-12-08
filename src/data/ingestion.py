# src/data/ingestion.py
from __future__ import annotations

from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset  # type: ignore

from src.config import DataConfig, get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _apply_subset(
    split: Dataset,
    max_samples: Optional[int],
    split_name: str,
) -> Dataset:
    """Optionally subset a split to max_samples for faster local development.

    Args:
        split: HuggingFace Dataset split.
        max_samples: Optional maximum number of samples to keep.
        split_name: Name of the split for logging purposes.

    Returns:
        Potentially subsetted Dataset.
    """
    if max_samples is None:
        return split

    original_size = len(split)
    keep = min(max_samples, original_size)

    if keep < original_size:
        logger.info(
            "Subsampling %s split from %d to %d samples.",
            split_name,
            original_size,
            keep,
        )
        split = split.select(range(keep))
    else:
        logger.info(
            "Requested max_samples (%d) >= size of %s split (%d). Using full split.",
            max_samples,
            split_name,
            original_size,
        )

    return split


def load_raw_dataset(cfg: DataConfig) -> DatasetDict:
    """Load the raw dataset for Model 1 in a standardized DatasetDict format.

    This function is responsible for:
      - Loading the dataset from HuggingFace (or other sources in the future)
      - Ensuring a DatasetDict with at least 'train' and 'test' splits
      - Optionally subsetting splits for fast local development

    Args:
        cfg: DataConfig instance with dataset_name, column names,
             and optional max_train_samples / max_eval_samples.

    Returns:
        A DatasetDict containing raw (unprocessed) data.

    Raises:
        ValueError: If required splits are missing.
    """
    settings = get_settings()

    logger.info(
        "Loading dataset.",
        extra={
            "dataset_name": cfg.dataset_name,
            "data_dir": str(settings.data_dir),
        },
    )

    # NOTE:
    # For now we assume a HuggingFace dataset by name, e.g. "ag_news".
    # Later this can be extended (csv, parquet, DB, etc.) while keeping
    # the same DatasetDict output contract.
    raw_ds = load_dataset(
        cfg.dataset_name,
        cache_dir=str(settings.data_dir / "hf_cache"),
    )

    if not isinstance(raw_ds, DatasetDict):
        raise ValueError(
            f"Expected DatasetDict from load_dataset, got {type(raw_ds)} instead."
        )

    # Ensure mandatory splits exist
    missing_splits = [split for split in ("train", "test") if split not in raw_ds]
    if missing_splits:
        raise ValueError(
            f"Missing required dataset splits: {missing_splits}. "
            f"Dataset '{cfg.dataset_name}' must contain at least 'train' and 'test'."
        )

    logger.info(
        "Dataset loaded successfully.",
        extra={
            "train_size": len(raw_ds["train"]),
            "test_size": len(raw_ds["test"]),
        },
    )

    # Optional subsampling for fast local experiments
    train_split = _apply_subset(
        raw_ds["train"],
        cfg.max_train_samples,
        split_name="train",
    )
    test_split = _apply_subset(
        raw_ds["test"],
        cfg.max_eval_samples,
        split_name="test",
    )

    raw_ds = DatasetDict(
        {
            "train": train_split,
            "test": test_split,
            # If the original dataset has validation or other splits,
            # we simply pass them through untouched.
            **{
                name: split
                for name, split in raw_ds.items()
                if name not in ("train", "test")
            },
        }
    )

    logger.info(
        "Final dataset splits after optional subsampling.",
        extra={
            "train_size": len(raw_ds["train"]),
            "test_size": len(raw_ds["test"]),
            "available_splits": list(raw_ds.keys()),
        },
    )

    return raw_ds
