
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.models.predict_model import load_model, predict, predict_proba
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_inference(X, config_path: str = "configs/config.yaml") -> dict:

    cfg = load_config(config_path)
    model_path = cfg["model"]["path"]

    # ── Load threshold from sidecar if available ──────────────────────────────
    meta_path = Path(model_path).with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        threshold     = meta.get("threshold", 0.5)
        model_version = meta.get("version", "unknown")
    else:
        threshold     = cfg["training"].get("default_threshold", 0.5)
        model_version = "unknown"
        logger.warning("No .meta.json found — using default threshold %.3f", threshold)

    model = load_model(model_path)
    probs = predict_proba(model, X)
    preds = (probs >= threshold).astype(int)

    logger.info(
        "Inference complete — samples=%d | threshold=%.4f | positives=%d",
        len(probs), threshold, int(preds.sum()),
    )
    return {
        "probabilities": probs,
        "predictions":   preds,
        "threshold":     threshold,
        "model_version": model_version,
    }
