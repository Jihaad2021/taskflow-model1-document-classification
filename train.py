# train.py

import argparse
import json
from pathlib import Path

from src.config import AppConfig, load_app_config
from src.data.pipeline import build_tokenized_datasets
from src.pipeline.training_pipeline import run_training_pipeline
from src.training.tuner import run_hyperparameter_search
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train Model 1 - Document Classification",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/local.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Run hyperparameter tuning only (no final training).",
    )
    parser.add_argument(
        "--tune-and-train",
        action="store_true",
        help=(
            "Run hyperparameter tuning first, then train a final model "
            "using the best hyperparameters."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials to run for hyperparameter tuning.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for running training and optional hyperparameter tuning."""
    args = parse_args()
    config_path = Path(args.config)

    if args.tune_only and args.tune_and_train:
        raise ValueError(
            "Invalid combination of flags: --tune-only and --tune-and-train "
            "cannot be used together."
        )

    logger.info(
        "Loading application configuration",
        extra={"config_path": str(config_path)},
    )

    app_cfg: AppConfig = load_app_config(config_path)

    # Case 1: tuning-only mode (no final training)
    if args.tune_only:
        logger.info(
            "Running in tune-only mode",
            extra={"n_trials": args.n_trials},
        )

        tokenized_datasets = build_tokenized_datasets(app_cfg)

        tuning_result = run_hyperparameter_search(
            app_cfg=app_cfg,
            tokenized_datasets=tokenized_datasets,
            n_trials=args.n_trials,
        )

        logger.info(
            "Tuning completed (tune-only mode)",
            extra=tuning_result,
        )

        print(json.dumps({"tuning_result": tuning_result}, indent=2))
        return

    # Case 2: tune-and-train mode (tuning -> final training)
    if args.tune_and_train:
        logger.info(
            "Running in tune-and-train mode",
            extra={"n_trials": args.n_trials},
        )

        tokenized_datasets = build_tokenized_datasets(app_cfg)

        tuning_result = run_hyperparameter_search(
            app_cfg=app_cfg,
            tokenized_datasets=tokenized_datasets,
            n_trials=args.n_trials,
        )

        logger.info(
            "Tuning completed; starting final training with best hyperparameters",
            extra=tuning_result,
        )

        best_params = tuning_result.get("best_params", {})

        # Build a tuned TrainingConfig from the baseline config and best_params
        tuned_training_cfg = app_cfg.training.model_copy(
            update=best_params,
        )
        tuned_app_cfg = app_cfg.model_copy(
            update={"training": tuned_training_cfg},
        )

        metrics = run_training_pipeline(tuned_app_cfg)

        logger.info(
            "Tune-and-train workflow completed",
            extra={"tuning_result": tuning_result, "final_metrics": metrics},
        )

        print(
            json.dumps(
                {
                    "tuning_result": tuning_result,
                    "final_metrics": metrics,
                },
                indent=2,
            )
        )
        return

    # Case 3: default mode (training only, no tuning)
    logger.info("Running in training-only mode")

    metrics = run_training_pipeline(app_cfg)

    logger.info("Training finished", extra={"metrics": metrics})
    print(json.dumps({"metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
