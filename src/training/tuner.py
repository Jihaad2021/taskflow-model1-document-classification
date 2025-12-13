from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from datasets import DatasetDict

from src.config import AppConfig, TrainingConfig
from src.training.trainer import train_classification_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from optuna.trial import Trial

try:
    import optuna 

    _OPTUNA_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - import-time guard
    optuna = None  # type: ignore[assignment]
    _OPTUNA_IMPORT_ERROR = exc


def _ensure_optuna_available() -> None:
    """Ensure that Optuna is available, otherwise raise a clear error."""
    if optuna is None:
        raise ImportError(
            "Optuna is required for hyperparameter tuning but is not installed. "
            "Install it with `pip install optuna`."
        ) from _OPTUNA_IMPORT_ERROR


def suggest_training_config(
    trial: Trial,
    base_training_cfg: TrainingConfig,
) -> TrainingConfig:
    """Create a TrainingConfig instance with hyperparameters suggested by Optuna.

    This function defines the hyperparameter search space for training. It
    starts from the base TrainingConfig and overrides selected fields with
    values sampled from the search space.

    Args:
        trial: Optuna Trial object used to sample hyperparameters.
        base_training_cfg: Baseline TrainingConfig used as a starting point.

    Returns:
        A new TrainingConfig instance with trial-specific hyperparameters.
    """
    # Learning rate: log-uniform between 1e-5 and 5e-4
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        5e-4,
        log=True,
    )

    # Weight decay: uniform between 0.0 and 0.1
    weight_decay = trial.suggest_float(
        "weight_decay",
        0.0,
        0.1,
    )

    # Number of epochs: discrete choices
    num_train_epochs = trial.suggest_int(
        "num_train_epochs",
        1,
        3,
    )

    # Batch size: discrete choices (per device)
    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size",
        [8, 16, 32],
    )
    per_device_eval_batch_size = max(
        per_device_train_batch_size, base_training_cfg.per_device_eval_batch_size
    )

    return base_training_cfg.model_copy(
        update={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
        }
    )


def run_hyperparameter_search(
    app_cfg: AppConfig,
    tokenized_datasets: DatasetDict,
    *,
    n_trials: int = 10,
    study_name: str = "model_1_hparam_search",
    direction: str = "maximize",
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """Run hyperparameter search for Model 1 using Optuna.

    This function performs hyperparameter optimization over the training
    configuration using the provided tokenized datasets. For each trial, a
    new TrainingConfig is sampled, the AppConfig is updated, and a training
    run is executed on the same tokenized datasets. The objective is to
    maximize the primary evaluation metric (e.g. F1 score).

    Args:
        app_cfg: Baseline application configuration.
        tokenized_datasets: Pre-tokenized DatasetDict used for all trials.
        n_trials: Number of Optuna trials to run.
        study_name: Optional name for the Optuna study.
        direction: Optimization direction ("maximize" or "minimize").
        storage: Optional Optuna storage URL (e.g. sqlite:///study.db). If
            provided, the study will be persisted; otherwise, it will be kept
            in memory.

    Returns:
        A dictionary containing:
            - "best_params": Best hyperparameters found by Optuna.
            - "best_value": Best objective value (e.g. best F1 score).
            - "n_trials": Number of completed trials.
    """
    _ensure_optuna_available()

    if "train" not in tokenized_datasets:
        raise KeyError("Tokenized datasets must contain a 'train' split.")

    # Select evaluation metric key based on Trainer configuration
    primary_metric = f"eval_{app_cfg.training.metric_for_best_model}"

    logger.info(
        "Starting hyperparameter search",
        extra={
            "study_name": study_name,
            "direction": direction,
            "primary_metric": primary_metric,
            "n_trials": n_trials,
        },
    )

    def objective(trial: Trial) -> float:
        """Optuna objective function for a single trial."""
        # Sample hyperparameters for this trial
        trial_training_cfg = suggest_training_config(
            trial=trial,
            base_training_cfg=app_cfg.training,
        )

        # Build a trial-specific AppConfig with the updated TrainingConfig
        trial_app_cfg = app_cfg.model_copy(
            update={"training": trial_training_cfg},
        )

        logger.info(
            "Running trial",
            extra={
                "trial_number": trial.number,
                "learning_rate": trial_training_cfg.learning_rate,
                "weight_decay": trial_training_cfg.weight_decay,
                "num_train_epochs": trial_training_cfg.num_train_epochs,
                "per_device_train_batch_size": trial_training_cfg.per_device_train_batch_size,
            },
        )

        metrics = train_classification_model(
            app_cfg=trial_app_cfg,
            tokenized_datasets=tokenized_datasets,
        )

        if primary_metric not in metrics:
            logger.warning(
                "Primary metric not found in metrics; falling back to 0.0",
                extra={"primary_metric": primary_metric, "metrics": metrics},
            )
            return 0.0

        value = float(metrics[primary_metric])
        trial.report(value, step=0)

        return value

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        "Hyperparameter search completed",
        extra={
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        },
    )

    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
    }
