import pytest

from src.config import AppConfig, get_settings, load_app_config
from src.pipeline.training_pipeline import run_training_pipeline
from src.training import trainer as trainer_module


@pytest.mark.integration
def test_training_pipeline_end_to_end_uses_tokenized_data_and_trains_model() -> None:
    """End-to-end test: config -> data pipeline -> training pipeline.

    This test verifies that:
        - The application config can be loaded from YAML.
        - The data pipeline can build tokenized datasets.
        - The training pipeline runs without errors.
        - Evaluation metrics are returned as a dictionary.
    """
    # Skip this test if HuggingFace Trainer is not available
    if getattr(trainer_module, "HfTrainer", None) is None:
        pytest.skip(
            "HuggingFace Trainer is not available (missing torch/accelerate); "
            "skipping training pipeline integration test."
        )

    settings = get_settings()
    config_path = settings.configs_dir / "local.yaml"

    app_cfg: AppConfig = load_app_config(config_path)

    # For integration tests, use a tiny model to reduce runtime and memory.
    tiny_model_name = "hf-internal-testing/tiny-random-distilbert"
    app_cfg.model = app_cfg.model.model_copy(
        update={"model_name": tiny_model_name},
    )

    metrics = run_training_pipeline(app_cfg)

    assert isinstance(metrics, dict)
