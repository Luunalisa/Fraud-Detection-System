"""
scripts/train.py — Entry-point script for training the Fraud Detection model.

Previously the  `if __name__ == "__main__":`  block at the bottom of train.py.
Separated here so the pipeline can be imported as a library without side-effects,
and so CI/CD systems have a clean, single-file entry point to invoke.

Usage
-----
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
"""

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
