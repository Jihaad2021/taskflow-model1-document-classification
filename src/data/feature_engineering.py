# src/data/feature_engineering.py
from __future__ import annotations

from typing import Any, Dict

from datasets import DatasetDict  # type: ignore
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import DataConfig, ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_tokenizer(model_cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer based on the configured model name.

    Args:
        model_cfg: Minimal model configuration containing model_name and max_length.

    Returns:
        A HuggingFace tokenizer instance.
    """
    logger.info(
        "Loading tokenizer.",
        extra={
            "model_name": model_cfg.model_name,
            "max_length": model_cfg.max_length,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)

    # Some tokenizers (e.g. GPT-style) might not have padding side set by default.
    # For encoder-based models like DistilBERT this is usually fine,
    # but we keep this here for safety / clarity.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _tokenize_batch(
    batch: Dict[str, Any],
    *,
    tokenizer: PreTrainedTokenizerBase,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
) -> Dict[str, Any]:
    """Tokenize a batch of examples for transformer models.

    This function:
        - Reads raw text from data_cfg.text_column
        - Uses the configured tokenizer to produce input_ids and attention_mask
        - Renames the label column to 'labels' to be compatible with HF Trainer
    """
    texts = batch.get(data_cfg.text_column, [])

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=model_cfg.max_length,
    )

    # Map label column to 'labels' (HF Trainer expects this)
    labels = batch.get(data_cfg.label_column, None)
    if labels is not None:
        tokenized["labels"] = labels

    return tokenized


def tokenize_dataset(
    ds: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
) -> DatasetDict:
    """Apply tokenizer to all splits in the DatasetDict.

    This function:
        - Assumes text has already been preprocessed.
        - Applies the same tokenizer configuration to every split.
        - Produces 'input_ids', 'attention_mask', and 'labels' columns.
        - Returns a new DatasetDict ready to be used by the training pipeline.

    Args:
        ds: Preprocessed DatasetDict containing raw text & label columns.
        tokenizer: Loaded HuggingFace tokenizer.
        model_cfg: Model configuration (for max_length, etc.).
        data_cfg: Data configuration (for column names).

    Returns:
        Tokenized DatasetDict.
    """
    logger.info(
        "Starting tokenization.",
        extra={
            "splits": list(ds.keys()),
            "text_column": data_cfg.text_column,
            "label_column": data_cfg.label_column,
            "max_length": model_cfg.max_length,
        },
    )

    # Apply tokenization to each split using map
    tokenized_ds = ds.map(
        _tokenize_batch,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model_cfg": model_cfg,
            "data_cfg": data_cfg,
        },
        remove_columns=[
            col
            for col in ds["train"].column_names
            if col not in (data_cfg.text_column, data_cfg.label_column)
        ],
        desc="Tokenizing dataset",
    )

    logger.info(
        "Tokenization completed.",
        extra={
            "splits": list(tokenized_ds.keys()),
            "example_keys": tokenized_ds["train"].column_names,
        },
    )

    return tokenized_ds
