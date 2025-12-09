from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from src.config import AppConfig, DataConfig, ModelConfig, RuntimeConfig, TrainingConfig
from src.training import trainer as trainer_module

class TestInferNumLabels:
    """Unit tests for infer_num_labels."""

    def test_infer_num_labels_from_unique_values(self) -> None:
        """Infer number of labels from unique values in the 'labels' column."""
        data = {
            "input_ids": [[1, 2], [3, 4], [5, 6]],
            "attention_mask": [[1, 1], [1, 1], [1, 1]],
            "labels": [0, 1, 1],
        }
        dataset = Dataset.from_dict(data)

        num_labels = trainer_module.infer_num_labels(dataset)

        assert num_labels == 2

    def test_raise_error_when_labels_column_missing(self) -> None:
        """Raise KeyError if 'labels' column is not present."""
        data = {
            "input_ids": [[1, 2], [3, 4]],
            "attention_mask": [[1, 1], [1, 1]],
        }
        dataset = Dataset.from_dict(data)

        with pytest.raises(KeyError):
            _ = trainer_module.infer_num_labels(dataset)


class TestCreateTrainingArguments:
    """Unit tests for create_training_arguments."""

    def _build_minimal_app_cfg(self) -> AppConfig:
        runtime_cfg = RuntimeConfig(run_mode="local", device="cpu", seed=42)
        data_cfg = DataConfig(
            dataset_name="ag_news",
            text_column="text",
            label_column="label",
        )
        model_cfg = ModelConfig(
            model_name="hf-internal-testing/tiny-random-distilbert",
            max_length=128,
        )
        training_cfg = TrainingConfig(
            output_dir_name="unit_test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1",
            greater_is_better=True,
        )

        return AppConfig(
            runtime=runtime_cfg,
            data=data_cfg,
            model=model_cfg,
            training=training_cfg,
        )

    def test_create_training_arguments_uses_output_dir_name(self, tmp_path: Path) -> None:
        """TrainingArguments should respect the configured output_dir_name."""
        app_cfg = self._build_minimal_app_class 

    """Unit tests for create_training_arguments."""

    def _build_minimal_app_cfg(self) -> AppConfig:
        runtime_cfg = RuntimeConfig(run_mode="local", device="cpu", seed=42)
        data_cfg = DataConfig(
            dataset_name="ag_news",
            text_column="text",
            label_column="label",
        )
        model_cfg = ModelConfig(
            model_name="hf-internal-testing/tiny-random-distilbert",
            max_length=128,
        )
        training_cfg = TrainingConfig(
            output_dir_name="unit_test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1",
            greater_is_better=True,
        )

        return AppConfig(
            runtime=runtime_cfg,
            data=data_cfg,
            model=model_cfg,
            training=training_cfg,
        )

    def test_create_training_arguments_uses_output_dir_name(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TrainingArguments should use models_dir / output_dir_name as output_dir."""
        app_cfg = self._build_minimal_app_cfg()

        class DummySettings:
            def __init__(self, models_dir: Path, artifacts_dir: Path):
                self.models_dir = models_dir
                self.artifacts_dir = artifacts_dir

        def dummy_get_settings() -> DummySettings:
            return DummySettings(models_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")

        # IMPORTANT: patch get_settings in the trainer module, not in src.config
        monkeypatch.setattr(trainer_module, "get_settings", dummy_get_settings)

        training_args = trainer_module.create_training_arguments(app_cfg=app_cfg)

        expected_output_dir = tmp_path / app_cfg.training.output_dir_name
        assert training_args.output_dir == str(expected_output_dir)