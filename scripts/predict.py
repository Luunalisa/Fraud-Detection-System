"""
scripts/predict.py — Entry-point script for running batch inference.

Usage
-----
    python scripts/predict.py --input data/new_transactions.csv
    python scripts/predict.py --input data/new_transactions.csv --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.logger import setup_logging
from src.pipelines.inference_pipeline import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch fraud inference.")
    parser.add_argument("--input",  required=True, help="Path to input CSV file.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--output", default=None, help="Optional path to save predictions CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    X = pd.read_csv(args.input)
    results = run_inference(X, config_path=args.config)

    if args.output:
        out = pd.DataFrame({
            "probability": results["probabilities"],
            "prediction":  results["predictions"],
        })
        out.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print(results)
