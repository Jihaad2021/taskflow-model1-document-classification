# src/data/pipeline.py
from __future__ import annotations

from datasets import DatasetDict  # type: ignore

from src.config import AppConfig
from src.data.ingestion import load_raw_dataset
from src.data.validation import validate_dataset
from src.data.preprocessing import apply_preprocessing
from src.data.feature_engineering import build_tokenizer, tokenize_dataset
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_tokenized_datasets(config: AppConfig) -> DatasetDict:
    """Full data pipeline for Model 1 (data → tokens).

    Steps:
        1. Ingestion      : Load raw dataset as DatasetDict.
        2. Validation     : Run structural & basic semantic checks.
        3. Preprocessing  : Clean text (whitespace normalization).
        4. Feature Eng.   : Tokenize text into input_ids & attention_mask.

    Args:
        config: AppConfig containing runtime, data, and model configs.

    Returns:
        Tokenized DatasetDict ready to be used by the training pipeline.
    """
    data_cfg = config.data
    model_cfg = config.model

    logger.info("Starting data pipeline for Model 1.")

    # 1) Ingestion
    raw_ds = load_raw_dataset(data_cfg)

    # 2) Validation
    validate_dataset(raw_ds, data_cfg)

    # 3) Preprocessing (minimal cleaning)
    preprocessed_ds = apply_preprocessing(raw_ds, data_cfg)

    # 4) Feature Engineering (tokenization)
    tokenizer = build_tokenizer(model_cfg)
    tokenized_ds = tokenize_dataset(
        preprocessed_ds,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
    )

    logger.info(
        "Data pipeline completed.",
        extra={
            "splits": list(tokenized_ds.keys()),
            "example_keys": tokenized_ds["train"].column_names,
        },
    )

    return tokenized_ds
