# src/data/preprocessing.py
from __future__ import annotations

import re
from typing import Any, Dict

from datasets import DatasetDict  # type: ignore

from src.config import DataConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: Any) -> str:
    """Basic text cleaning for Transformer-based models.

    Since we are using Transformer tokenizers (e.g. DistilBERT),
    we intentionally keep preprocessing minimal:

        - Convert non-string inputs to empty string.
        - Strip leading/trailing whitespace.
        - Collapse multiple whitespace characters into a single space.

    We do NOT:
        - remove punctuation
        - remove stopwords
        - apply stemming/lemmatization

    Args:
        text: Raw text value from the dataset.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Strip leading/trailing whitespace
    stripped = text.strip()

    if not stripped:
        return ""

    # Normalize internal whitespace (spaces, tabs, newlines) to a single space
    normalized = _WHITESPACE_RE.sub(" ", stripped)

    return normalized


def _preprocess_batch(batch: Dict[str, Any], text_column: str) -> Dict[str, Any]:
    """Apply clean_text to all examples in a batch for the given text column."""
    texts = batch.get(text_column, [])

    # If somehow the column is missing or not a list, just return as-is
    if not isinstance(texts, list):
        return batch

    cleaned_texts = [clean_text(t) for t in texts]
    batch[text_column] = cleaned_texts
    return batch


def apply_preprocessing(ds: DatasetDict, cfg: DataConfig) -> DatasetDict:
    """Apply minimal, safe preprocessing to the text column in all splits.

    This function:
        - Leaves the overall DatasetDict structure unchanged.
        - Only modifies the configured text_column in each split.
        - Applies the same cleaning function (clean_text) across splits.

    Args:
        ds: Raw (ingested and validated) DatasetDict.
        cfg: DataConfig specifying the text column name.

    Returns:
        A new DatasetDict with cleaned text.

    Notes:
        Preprocessing is intentionally minimal because modern Transformer
        tokenizers are robust and expect near-raw text. Heavy normalization
        (e.g. removing punctuation, stopwords) is left out on purpose.
    """
    text_col = cfg.text_column

    logger.info(
        "Starting text preprocessing.",
        extra={"text_column": text_col, "splits": list(ds.keys())},
    )

    # Use DatasetDict.map to apply cleaning to each split
    processed_ds = ds.map(
        _preprocess_batch,
        fn_kwargs={"text_column": text_col},
        batched=True,
        desc="Applying text preprocessing",
    )

    logger.info(
        "Text preprocessing completed.",
        extra={
            "text_column": text_col,
            "splits": list(processed_ds.keys()),
        },
    )

    return processed_ds
