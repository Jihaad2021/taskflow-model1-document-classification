from pathlib import Path
from typing import Any, Dict

import pytest
from datasets import Dataset, DatasetDict

from src.config import AppConfig, DataConfig, ModelConfig, RuntimeConfig, TrainingConfig
from src.training import tuner as tuner_module


class DummyTrial:
    """Simple dummy trial that mimics the Optuna Trial API for unit testing.

    This class does not depend on Optuna and is used to test the behavior of
    `suggest_training_config` in isolation.
    """

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        """Return a fixed value within the given range and record it."""
        # For testing, always return the midpoint (or geometric midpoint if log=True).
        if log:
            value = (low * high) ** 0.5
        else:
            value = (low + high) / 2.0
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int) -> int:
        """Return the middle integer between low and high and record it."""
        value = (low + high) // 2
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: list[int]) -> int:
        """Return the first choice and record it."""
        value = choices[0]
        self.params[name] = value
        return value


class TestSuggestTrainingConfig:
    """Unit tests for suggest_training_config."""

    def _build_base_training_cfg(self) -> TrainingConfig:
        return TrainingConfig(
            output_dir_name="unit_test_output",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.0,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1",
            greater_is_better=True,
        )

    def test_suggest_training_config_overrides_hyperparameters(self) -> None:
        """Suggested training config should override selected hyperparameters."""
        base_cfg = self._build_base_training_cfg()
        dummy_trial = DummyTrial()

        updated_cfg = tuner_module.suggest_training_config(
            trial=dummy_trial,
            base_training_cfg=base_cfg,
        )

        # Ensure that core hyperparameters are updated and differ from the base config
        assert updated_cfg.learning_rate != base_cfg.learning_rate
        assert updated_cfg.weight_decay != base_cfg.weight_decay
        assert updated_cfg.num_train_epochs != base_cfg.num_train_epochs
        assert updated_cfg.per_device_train_batch_size != base_cfg.per_device_train_batch_size

        # Ensure eval batch size is at least the training batch size
        assert updated_cfg.per_device_eval_batch_size >= updated_cfg.per_device_train_batch_size

        # Ensure other fields are preserved
        assert updated_cfg.output_dir_name == base_cfg.output_dir_name
        assert updated_cfg.metric_for_best_model == base_cfg.metric_for_best_model
        assert updated_cfg.greater_is_better == base_cfg.greater_is_better


@pytest.mark.skipif(
    tuner_module.optuna is None,
    reason="Optuna is not available; skipping hyperparameter search tests.",
)
class TestRunHyperparameterSearch:
    """Unit-level tests for run_hyperparameter_search with a mocked trainer."""

    def _build_minimal_app_cfg(self) -> AppConfig:
        runtime_cfg = RuntimeConfig(run_mode="local", device="cpu", seed=42)
        data_cfg = DataConfig(
            dataset_name="ag_news",
            text_column="text",
            label_column="label",
            max_train_samples=10,
            max_eval_samples=5,
        )
        model_cfg = ModelConfig(
            model_name="hf-internal-testing/tiny-random-distilbert",
            max_length=32,
        )
        training_cfg = TrainingConfig(
            output_dir_name="unit_test_output",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            weight_decay=0.0,
            warmup_ratio=0.0,
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

    def _build_dummy_tokenized_datasets(self) -> DatasetDict:
        """Create a minimal tokenized DatasetDict for testing.

        The content is not used by the mocked trainer, but the presence of a
        'train' split is required by run_hyperparameter_search.
        """
        data = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "labels": [0],
        }
        train_dataset = Dataset.from_dict(data)
        return DatasetDict({"train": train_dataset})

    def test_run_hyperparameter_search_basic_flow_with_mocked_trainer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Hyperparameter search should complete and return best params and value.

        The underlying training is mocked so that the test does not depend on
        HuggingFace Trainer or GPU resources.
        """
        app_cfg = self._build_minimal_app_cfg()
        tokenized_datasets = self._build_dummy_tokenized_datasets()

        # Mock train_classification_model to avoid real training.
        # The returned value depends on the learning rate so that Optuna
        # has a meaningful objective to optimize.
        def fake_train_classification_model(
            app_cfg: AppConfig,
            tokenized_datasets: DatasetDict,
        ) -> Dict[str, float]:
            lr = app_cfg.training.learning_rate
            # Simple synthetic objective: higher value for smaller learning rates
            # within the search range. This is arbitrary but deterministic.
            score = float(1.0 / lr)
            return {"eval_f1": score}

        monkeypatch.setattr(
            tuner_module,
            "train_classification_model",
            fake_train_classification_model,
        )

        result = tuner_module.run_hyperparameter_search(
            app_cfg=app_cfg,
            tokenized_datasets=tokenized_datasets,
            n_trials=3,
            study_name="unit_test_study",
        )

        # Basic structure checks
        assert "best_params" in result
        assert "best_value" in result
        assert "n_trials" in result

        assert isinstance(result["best_params"], dict)
        assert isinstance(result["best_value"], float)
        assert result["n_trials"] == 3

        # Ensure that at least the learning_rate is part of the tuned parameters
        assert "learning_rate" in result["best_params"]
