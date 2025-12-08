# tests/unit/test_ingestion.py
from __future__ import annotations

from datasets import DatasetDict  # type: ignore

from src.config import DataConfig
from src.data.ingestion import load_raw_dataset


def test_load_raw_dataset_ag_news_basic():
    """Dataset AG News berhasil di-load sebagai DatasetDict dengan train & test."""

    cfg = DataConfig(
        dataset_name="ag_news",
        text_column="text",
        label_column="label",
        max_train_samples=None,
        max_eval_samples=None,
    )

    ds = load_raw_dataset(cfg)

    # Harus berbentuk DatasetDict
    assert isinstance(ds, DatasetDict)

    # Harus punya minimal split train & test
    assert "train" in ds
    assert "test" in ds

    # Harus ada data di masing-masing split
    assert len(ds["train"]) > 0
    assert len(ds["test"]) > 0


def test_load_raw_dataset_with_subsampling():
    """Subsampling max_train_samples dan max_eval_samples bekerja sebagaimana mestinya."""

    max_train = 100
    max_eval = 20

    cfg = DataConfig(
        dataset_name="ag_news",
        text_column="text",
        label_column="label",
        max_train_samples=max_train,
        max_eval_samples=max_eval,
    )

    ds = load_raw_dataset(cfg)

    # Tetap DatasetDict yang valid
    assert isinstance(ds, DatasetDict)
    assert "train" in ds
    assert "test" in ds

    # Ukuran split harus <= batas yang diminta
    assert len(ds["train"]) <= max_train
    assert len(ds["train"]) > 0  # tapi tidak boleh 0
    assert len(ds["test"]) <= max_eval
    assert len(ds["test"]) > 0
