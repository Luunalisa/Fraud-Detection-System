

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import setup_logging
from src.pipelines.training_pipeline import train_model
from src.data_pipeline import run_pipeline
from src.pipelines.inference_pipeline import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full training + optional inference pipeline.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="After training, run inference on the validation split as a sanity check.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    model, metrics = train_model(config_path=args.config)
    print("Training metrics:", metrics)

    if args.smoke_test:
        _, _, X_val, _, _, _ = run_pipeline()
        results = run_inference(X_val, config_path=args.config)
        print(
            f"Smoke-test inference — positives: {results['predictions'].sum()} / {len(results['predictions'])}"
        )
