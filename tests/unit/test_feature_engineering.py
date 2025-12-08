# tests/unit/test_feature_engineering.py
from datasets import Dataset, DatasetDict  # type: ignore
from transformers import AutoTokenizer

from src.config import DataConfig, ModelConfig
from src.data.feature_engineering import tokenize_dataset


def _build_small_dataset() -> DatasetDict:
    texts = ["hello world", "taskflow project"]
    labels = [0, 1]
    train = Dataset.from_dict({"text": texts, "label": labels})
    test = Dataset.from_dict({"text": texts[:1], "label": labels[:1]})
    return DatasetDict({"train": train, "test": test})


def test_tokenize_dataset_shapes_and_columns():
    ds = _build_small_dataset()

    data_cfg = DataConfig(
        dataset_name="dummy",
        text_column="text",
        label_column="label",
    )

    model_cfg = ModelConfig(
        model_name="distilbert-base-uncased",
        num_labels=2,
        max_length=16,
        dropout=0.1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)

    tokenized = tokenize_dataset(ds, tokenizer, model_cfg, data_cfg)

    train = tokenized["train"]
    sample = train[0]

    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample

    # Karena kita belum memanggil set_format("torch"), input_ids masih list[int]
    assert isinstance(sample["input_ids"], list)
    assert isinstance(sample["attention_mask"], list)

    assert len(sample["input_ids"]) == model_cfg.max_length
    assert len(sample["attention_mask"]) == model_cfg.max_length
