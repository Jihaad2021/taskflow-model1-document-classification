from typing import Dict, Optional

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)

from src.config import ModelConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_classification_model(
    model_cfg: ModelConfig,
    *,
    num_labels: Optional[int] = None,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
) -> PreTrainedModel:
    """Build a sequence classification model from configuration.

    This function constructs a HuggingFace sequence classification model
    (e.g. DistilBERT) using the given ModelConfig. The caller must provide
    the number of labels either via the function argument or via
    `model_cfg.num_labels`. Optional label mappings can also be provided
    for better interpretability.

    Args:
        model_cfg: Pydantic ModelConfig containing model name and defaults.
        num_labels: Optional explicit number of labels. If not provided,
            the function will fall back to `model_cfg.num_labels`.
        id2label: Optional mapping from integer class index to label name.
            If provided, this takes precedence over `model_cfg.id2label`.
        label2id: Optional mapping from label name to integer class index.
            If provided, this takes precedence over `model_cfg.label2id`.

    Returns:
        An instance of `PreTrainedModel` ready for training.

    Raises:
        ValueError: If the number of labels cannot be resolved from either
            the function argument or the ModelConfig.
    """
    resolved_num_labels = num_labels or model_cfg.num_labels
    if resolved_num_labels is None:
        raise ValueError(
            "num_labels must be provided either via the function argument "
            "or ModelConfig.num_labels."
        )

    config_kwargs: Dict[str, object] = {
        "num_labels": resolved_num_labels,
    }

    if id2label is not None:
        config_kwargs["id2label"] = id2label
    elif model_cfg.id2label is not None:
        config_kwargs["id2label"] = model_cfg.id2label

    if label2id is not None:
        config_kwargs["label2id"] = label2id
    elif model_cfg.label2id is not None:
        config_kwargs["label2id"] = model_cfg.label2id

    logger.info(
        "Loading classification model",
        extra={
            "model_name": model_cfg.model_name,
            "num_labels": resolved_num_labels,
        },
    )

    hf_config = AutoConfig.from_pretrained(
        model_cfg.model_name,
        **config_kwargs,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.model_name,
        config=hf_config,
        ignore_mismatched_sizes=True,  # <-- key change
    )

    logger.info(
        "Model loaded successfully",
        extra={
            "model_type": model.config.model_type,
            "num_labels": model.config.num_labels,
        },
    )

    return model
