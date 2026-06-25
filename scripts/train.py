
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import get_logger
from src.pipelines.training_pipeline import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Fraud Detection XGBoost model.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the project YAML config (default: configs/config.yaml).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    get_logger(__name__)
    args = parse_args()
    train_model(config_path=args.config)
