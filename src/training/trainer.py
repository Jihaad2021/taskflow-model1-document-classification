# src/training/trainer.py (bagian atas)

from pathlib import Path
from typing import Dict, Optional

from datasets import Dataset, DatasetDict
from transformers import TrainingArguments  # safe: tidak butuh accelerate

from src.config import AppConfig, get_settings
from src.evaluation.metrics import hf_compute_classification_metrics
from src.models.builder import build_classification_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    # Import Trainer lazily and guard against missing dependencies (e.g. accelerate)
    from transformers import Trainer as HfTrainer  # type: ignore[import]
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # catch broad because HF may raise different error types
    HfTrainer = None  # type: ignore[assignment]
    _TRAINER_IMPORT_ERROR = exc

def infer_num_labels(train_dataset: Dataset) -> int:
    """Infer the number of labels from a tokenized training dataset.

    This function inspects the `labels` feature or the `labels` column
    to determine how many distinct classes exist.

    Args:
        train_dataset: Tokenized training dataset containing a `labels` column.

    Returns:
        The inferred number of distinct labels.

    Raises:
        KeyError: If the dataset does not contain a `labels` column.
        ValueError: If the number of labels cannot be inferred.
    """
    if "labels" not in train_dataset.column_names:
        raise KeyError("Training dataset must contain a 'labels' column.")

    # Prefer feature metadata if available
    labels_feature = train_dataset.features.get("labels")
    num_labels: Optional[int] = None

    if labels_feature is not None and hasattr(labels_feature, "num_classes"):
        num_labels = getattr(labels_feature, "num_classes", None)

    if num_labels is None:
        # Fallback: derive from unique values in the column
        label_values = train_dataset["labels"]
        unique_labels = set(int(label) for label in label_values)
        num_labels = len(unique_labels)

    if num_labels is None or num_labels < 2:
        raise ValueError(
            "Could not infer a valid number of labels from the training dataset."
        )

    return num_labels


def create_training_arguments(
    app_cfg: AppConfig,
    output_dir: Optional[Path] = None,
) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from AppConfig and Settings.

    Args:
        app_cfg: Application configuration including training and runtime sections.
        output_dir: Optional explicit output directory for checkpoints. If not
            provided, the directory is computed from Settings.models_dir and
            training.output_dir_name.

    Returns:
        Configured TrainingArguments instance.
    """
    settings = get_settings()

    base_output_dir = output_dir or (
        settings.models_dir / app_cfg.training.output_dir_name
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(base_output_dir),
        num_train_epochs=app_cfg.training.num_train_epochs,
        per_device_train_batch_size=app_cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=app_cfg.training.per_device_eval_batch_size,
        learning_rate=app_cfg.training.learning_rate,
        weight_decay=app_cfg.training.weight_decay,
        warmup_ratio=app_cfg.training.warmup_ratio,
        logging_steps=app_cfg.training.logging_steps,
        evaluation_strategy=app_cfg.training.evaluation_strategy,
        save_strategy=app_cfg.training.save_strategy,
        metric_for_best_model=app_cfg.training.metric_for_best_model,
        greater_is_better=app_cfg.training.greater_is_better,
        load_best_model_at_end=True,
        seed=app_cfg.runtime.seed,
        logging_dir=str(settings.artifacts_dir / "tensorboard"),
        report_to=[],
    )

    return training_args


def train_classification_model(
    app_cfg: AppConfig,
    tokenized_datasets: DatasetDict,
) -> Dict[str, float]:
    """Train a text classification model using HuggingFace Trainer.

    This function expects a tokenized DatasetDict produced by the data
    pipeline (Phase 1) and trains a classification model defined by the
    current AppConfig. The best model is saved to disk, and evaluation
    metrics are returned.

    Args:
        app_cfg: Application configuration (runtime, model, training, data).
        tokenized_datasets: Tokenized datasets containing at least a 'train'
            split and optionally 'validation' or 'test' for evaluation.

    Returns:
        Dictionary of evaluation metrics produced by Trainer.evaluate().
        If no evaluation split is available, an empty dictionary is returned.
    """
    if "train" not in tokenized_datasets:
        raise KeyError("Tokenized datasets must contain a 'train' split.")

    train_dataset = tokenized_datasets["train"]
    eval_dataset: Optional[Dataset] = None

    if "validation" in tokenized_datasets:
        eval_dataset = tokenized_datasets["validation"]
    elif "test" in tokenized_datasets:
        eval_dataset = tokenized_datasets["test"]

    num_labels = infer_num_labels(train_dataset)
    logger.info(
        "Inferred number of labels from dataset",
        extra={"num_labels": num_labels},
    )

    model = build_classification_model(
        model_cfg=app_cfg.model,
        num_labels=num_labels,
    )

    training_args = create_training_arguments(app_cfg=app_cfg)
    
    # Ensure Trainer is available (accelerate/torch installed) before using it
    if HfTrainer is None:
        raise ImportError(
            "HuggingFace Trainer is not available. Make sure 'torch' and "
            "'accelerate' are installed in your environment."
        ) from _TRAINER_IMPORT_ERROR

    try:
        trainer = HfTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=hf_compute_classification_metrics
            if eval_dataset is not None
            else None,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Failed to initialize HuggingFace Trainer. This is most likely caused by "
            "a version mismatch between 'transformers' and 'accelerate'. "
            "Please ensure they are compatible (e.g., upgrade both via "
            "'pip install -U transformers accelerate')."
        ) from exc


    logger.info("Starting training loop")
    trainer.train()

    metrics: Dict[str, float] = {}
    if eval_dataset is not None:
        logger.info("Running evaluation on validation/test split")
        metrics = trainer.evaluate()

    logger.info(
        "Training complete. Saving best model.",
        extra={"output_dir": training_args.output_dir},
    )
    trainer.save_model(training_args.output_dir)

    return metrics
