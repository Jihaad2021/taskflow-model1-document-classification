from typing import Dict

from datasets import DatasetDict

from src.config import AppConfig
from src.data.pipeline import build_tokenized_datasets
from src.training.trainer import train_classification_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_training_pipeline(app_cfg: AppConfig) -> Dict[str, float]:
    """Run the full training pipeline for Model 1 (classification).

    This function orchestrates the following steps:
        1. Build tokenized datasets using the data pipeline (Phase 1).
        2. Train a classification model using the training module (Phase 2).
        3. Return evaluation metrics produced by the trainer.

    Args:
        app_cfg: Application configuration including runtime, data, model,
            and training sections.

    Returns:
        A dictionary of evaluation metrics from the training run. The exact
        keys depend on the Trainer configuration (e.g. 'eval_loss',
        'eval_accuracy', 'eval_f1'). If no evaluation split is available,
        an empty dictionary is returned.
    """
    logger.info("Starting training pipeline for Model 1 (classification)")

    # 1) Build tokenized datasets via data pipeline (Phase 1)
    logger.info("Building tokenized datasets via data pipeline")
    tokenized_datasets: DatasetDict = build_tokenized_datasets(app_cfg)

    # 2) Run training
    logger.info("Running training for classification model")
    metrics = train_classification_model(
        app_cfg=app_cfg,
        tokenized_datasets=tokenized_datasets,
    )

    logger.info(
        "Training pipeline completed",
        extra={"metrics": metrics},
    )

    return metrics
