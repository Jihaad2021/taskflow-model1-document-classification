# tests/unit/test_train_cli.py

import json
import sys
from typing import Any, Dict

import pytest

import train as train_module
from src.config import AppConfig, DataConfig, ModelConfig, RuntimeConfig, TrainingConfig


def _build_dummy_app_cfg() -> AppConfig:
    """Build a minimal AppConfig instance for CLI tests."""
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


def test_train_cli_training_only_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI should run training pipeline only when no tuning flags are provided."""
    dummy_app_cfg = _build_dummy_app_cfg()
    called: Dict[str, Any] = {"run_training_called": False, "app_cfg": None}

    # Mock load_app_config to return a dummy config without touching YAML.
    def fake_load_app_config(config_path: str) -> AppConfig:
        return dummy_app_cfg

    # Mock run_training_pipeline to avoid real training.
    def fake_run_training_pipeline(app_cfg: AppConfig) -> Dict[str, float]:
        called["run_training_called"] = True
        called["app_cfg"] = app_cfg
        return {"eval_accuracy": 0.9}

    monkeypatch.setattr(train_module, "load_app_config", fake_load_app_config)
    monkeypatch.setattr(train_module, "run_training_pipeline", fake_run_training_pipeline)

    # No tuning flags: training-only mode
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/local.yaml"],
    )

    train_module.main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert called["run_training_called"] is True
    assert "metrics" in output
    assert output["metrics"]["eval_accuracy"] == 0.9


def test_train_cli_tune_only_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI should run hyperparameter tuning only when --tune-only is provided."""
    dummy_app_cfg = _build_dummy_app_cfg()
    called: Dict[str, Any] = {
        "build_tokenized_called": False,
        "run_training_called": False,
        "run_tuning_called": False,
        "n_trials": None,
    }

    def fake_load_app_config(config_path: str) -> AppConfig:
        return dummy_app_cfg

    def fake_build_tokenized_datasets(app_cfg: AppConfig) -> str:
        called["build_tokenized_called"] = True
        return "DUMMY_TOKENIZED"

    def fake_run_training_pipeline(app_cfg: AppConfig) -> Dict[str, float]:
        called["run_training_called"] = True
        return {"eval_accuracy": 0.0}

    def fake_run_hyperparameter_search(
        app_cfg: AppConfig,
        tokenized_datasets: Any,
        n_trials: int = 10,
        study_name: str = "unit_test_study",
        direction: str = "maximize",
        storage: str | None = None,
    ) -> Dict[str, Any]:
        called["run_tuning_called"] = True
        called["n_trials"] = n_trials
        assert tokenized_datasets == "DUMMY_TOKENIZED"
        return {
            "best_params": {"learning_rate": 1e-4},
            "best_value": 0.95,
            "n_trials": n_trials,
        }

    monkeypatch.setattr(train_module, "load_app_config", fake_load_app_config)
    monkeypatch.setattr(train_module, "build_tokenized_datasets", fake_build_tokenized_datasets)
    monkeypatch.setattr(train_module, "run_training_pipeline", fake_run_training_pipeline)
    monkeypatch.setattr(train_module, "run_hyperparameter_search", fake_run_hyperparameter_search)

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/local.yaml", "--tune-only", "--n-trials", "3"],
    )

    train_module.main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert called["build_tokenized_called"] is True
    assert called["run_tuning_called"] is True
    assert called["run_training_called"] is False
    assert called["n_trials"] == 3

    assert "tuning_result" in output
    assert output["tuning_result"]["best_value"] == 0.95


def test_train_cli_tune_and_train_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI should run tuning first and then training with best hyperparameters when --tune-and-train is provided."""
    dummy_app_cfg = _build_dummy_app_cfg()
    called: Dict[str, Any] = {
        "build_tokenized_called": False,
        "run_tuning_called": False,
        "run_training_called": False,
        "training_lr_used": None,
    }

    def fake_load_app_config(config_path: str) -> AppConfig:
        return dummy_app_cfg

    def fake_build_tokenized_datasets(app_cfg: AppConfig) -> str:
        called["build_tokenized_called"] = True
        return "DUMMY_TOKENIZED"

    def fake_run_hyperparameter_search(
        app_cfg: AppConfig,
        tokenized_datasets: Any,
        n_trials: int = 10,
        study_name: str = "unit_test_study",
        direction: str = "maximize",
        storage: str | None = None,
    ) -> Dict[str, Any]:
        called["run_tuning_called"] = True
        assert tokenized_datasets == "DUMMY_TOKENIZED"
        return {
            "best_params": {"learning_rate": 1e-4},
            "best_value": 0.95,
            "n_trials": n_trials,
        }

    def fake_run_training_pipeline(app_cfg: AppConfig) -> Dict[str, float]:
        called["run_training_called"] = True
        # Capture the learning rate used in the final training config
        called["training_lr_used"] = app_cfg.training.learning_rate
        return {"eval_accuracy": 0.92}

    monkeypatch.setattr(train_module, "load_app_config", fake_load_app_config)
    monkeypatch.setattr(train_module, "build_tokenized_datasets", fake_build_tokenized_datasets)
    monkeypatch.setattr(train_module, "run_hyperparameter_search", fake_run_hyperparameter_search)
    monkeypatch.setattr(train_module, "run_training_pipeline", fake_run_training_pipeline)

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/local.yaml", "--tune-and-train", "--n-trials", "2"],
    )

    train_module.main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert called["build_tokenized_called"] is True
    assert called["run_tuning_called"] is True
    assert called["run_training_called"] is True

    # Ensure the final training used the tuned learning rate
    assert called["training_lr_used"] == pytest.approx(1e-4)

    assert "tuning_result" in output
    assert "final_metrics" in output
    assert output["final_metrics"]["eval_accuracy"] == 0.92


def test_train_cli_invalid_flag_combination(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should raise ValueError when both --tune-only and --tune-and-train are provided."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--tune-only", "--tune-and-train"],
    )

    with pytest.raises(ValueError):
        train_module.main()
