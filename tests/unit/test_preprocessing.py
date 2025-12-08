# tests/unit/test_preprocessing.py
from __future__ import annotations

from datasets import Dataset, DatasetDict  # type: ignore

from src.config import DataConfig
from src.data.preprocessing import clean_text, apply_preprocessing


def test_clean_text_basic_normalization():
    """clean_text harus strip dan normalize whitespace dengan benar."""
    raw = "   Hello   world\tfrom   TaskFlow   "
    cleaned = clean_text(raw)
    assert cleaned == "Hello world from TaskFlow"


def test_clean_text_non_string_returns_empty():
    """Jika input bukan string (None, angka, dll) → harus jadi string kosong."""
    assert clean_text(None) == ""  # type: ignore[arg-type]
    assert clean_text(123) == ""   # type: ignore[arg-type]


def test_clean_text_empty_string():
    """String kosong atau whitespace-only → jadi string kosong."""
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""


def test_apply_preprocessing_on_datasetdict():
    """apply_preprocessing harus membersihkan kolom teks di semua split tanpa mengubah struktur."""
    texts_train = ["  hello   world  ", "TaskFlow   project"]
    labels_train = [0, 1]

    texts_test = ["  another   text ", " second   sample  "]
    labels_test = [1, 0]

    train = Dataset.from_dict({"text": texts_train, "label": labels_train})
    test = Dataset.from_dict({"text": texts_test, "label": labels_test})

    raw_ds = DatasetDict({"train": train, "test": test})

    cfg = DataConfig(
        dataset_name="dummy",
        text_column="text",
        label_column="label",
        max_train_samples=None,
        max_eval_samples=None,
    )

    processed = apply_preprocessing(raw_ds, cfg)

    # Struktur harus tetap DatasetDict dengan split yang sama
    assert isinstance(processed, DatasetDict)
    assert set(processed.keys()) == {"train", "test"}

    # Jumlah sample di setiap split tidak berubah
    assert len(processed["train"]) == len(raw_ds["train"])
    assert len(processed["test"]) == len(raw_ds["test"])

    # Teks harus sudah dibersihkan
    train_texts = processed["train"]["text"]
    test_texts = processed["test"]["text"]

    assert train_texts[0] == "hello world"
    assert train_texts[1] == "TaskFlow project"

    assert test_texts[0] == "another text"
    assert test_texts[1] == "second sample"

    # Label tidak boleh berubah
    assert processed["train"]["label"] == labels_train
    assert processed["test"]["label"] == labels_test
