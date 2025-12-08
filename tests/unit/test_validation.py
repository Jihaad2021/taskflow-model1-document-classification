# tests/unit/test_validation.py
from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict  # type: ignore

from src.config import DataConfig
from src.data.validation import validate_dataset


def _build_valid_dataset() -> DatasetDict:
    texts = ["hello world", "another sample", "taskflow model1", "data pipeline"]
    labels = [0, 1, 0, 1]

    train = Dataset.from_dict({"text": texts, "label": labels})
    test = Dataset.from_dict({"text": texts[:2], "label": labels[:2]})

    return DatasetDict({"train": train, "test": test})


def test_validate_dataset_success_on_valid_data():
    """Dataset valid → validate_dataset() tidak melempar error."""

    ds = _build_valid_dataset()
    cfg = DataConfig(
        dataset_name="dummy",
        text_column="text",
        label_column="label",
        max_train_samples=None,
        max_eval_samples=None,
    )

    # Tidak boleh raise exception apa pun
    validate_dataset(ds, cfg)


def test_validate_dataset_missing_test_split_raises():
    """Jika split 'test' hilang → harus raise ValueError."""

    texts = ["hello"]
    labels = [0]
    train = Dataset.from_dict({"text": texts, "label": labels})

    ds = DatasetDict({"train": train})  # tidak ada 'test'
    cfg = DataConfig(
        dataset_name="dummy",
        text_column="text",
        label_column="label",
        max_train_samples=None,
        max_eval_samples=None,
    )

    with pytest.raises(ValueError) as exc:
        validate_dataset(ds, cfg)

    assert "Missing required dataset splits" in str(exc.value)


def test_validate_dataset_missing_required_columns_raises():
    """Jika kolom text/label hilang → harus raise ValueError."""

    texts = ["hello", "world"]
    labels = [0, 1]

    # Salah: pakai nama kolom 'wrong_text' bukan 'text'
    train = Dataset.from_dict({"wrong_text": texts, "label": labels})
    test = Dataset.from_dict({"wrong_text": texts, "label": labels})

    ds = DatasetDict({"train": train, "test": test})

    cfg = DataConfig(
        dataset_name="dummy",
        text_column="text",   # kolom ini tidak ada
        label_column="label",
        max_train_samples=None,
        max_eval_samples=None,
    )

    with pytest.raises(ValueError) as exc:
        validate_dataset(ds, cfg)

    assert "Missing required columns" in str(exc.value)


def test_validate_dataset_single_class_labels_raises():
    """Jika label di training hanya 1 kelas → harus raise ValueError."""

    texts = ["sample 1", "sample 2", "sample 3"]
    labels = [0, 0, 0]  # hanya satu kelas

    train = Dataset.from_dict({"text": texts, "label": labels})
    test = Dataset.from_dict({"text": texts, "label": labels})

    ds = DatasetDict({"train": train, "test": test})

    cfg = DataConfig(
        dataset_name="dummy",
        text_column="text",
        label_column="label",
        max_train_samples=None,
        max_eval_samples=None,
    )

    with pytest.raises(ValueError) as exc:
        validate_dataset(ds, cfg)

    assert "fewer than 2 classes" in str(exc.value)
