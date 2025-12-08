# tests/integration/test_data_pipeline.py
from __future__ import annotations

from datasets import DatasetDict  # type: ignore

from src.config import load_app_config, get_settings
from src.data.pipeline import build_tokenized_datasets


def test_build_tokenized_datasets_end_to_end():
    """End-to-end: config -> data pipeline -> tokenized DatasetDict ready for training."""

    settings = get_settings()
    config_path = settings.configs_dir / "local.yaml"

    app_cfg = load_app_config(config_path)
    tokenized = build_tokenized_datasets(app_cfg)

    # Output harus DatasetDict
    assert isinstance(tokenized, DatasetDict)

    # Harus punya minimal train & test
    assert "train" in tokenized
    assert "test" in tokenized

    # Tidak boleh kosong
    assert len(tokenized["train"]) > 0
    assert len(tokenized["test"]) > 0

    # Ambil satu contoh dari train
    example = tokenized["train"][0]

    # Kolom wajib untuk training HF Trainer
    assert "input_ids" in example
    assert "attention_mask" in example
    assert "labels" in example

    # input_ids & attention_mask harus panjangnya sama
    assert len(example["input_ids"]) == len(example["attention_mask"])

    # Label bertipe integer
    assert isinstance(example["labels"], int)
