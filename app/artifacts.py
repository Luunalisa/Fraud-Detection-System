from __future__ import annotations

import json
import logging
import pickle
import joblib
from functools import lru_cache
from pathlib import Path
from src.utils.config import load_config

import joblib

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")


@lru_cache(maxsize=None)
def get_artifacts() -> dict:
    
    data_cfg = cfg = load_config("configs/data_config.yaml")
    cfg = load_config("configs/config.yaml")
    logger.info("Loading inference artifacts from %s ...", ARTIFACTS_DIR)
    
    # ── Model ─────────────────────────────────────────────────────────────────
    model_path  = Path(cfg["artifacts"]["model"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    logger.info("  ✓ model loaded")

    # ── Threshold ──────────────────────────────────
    threshold_path = ARTIFACTS_DIR / "evaluation/threshold_analysis.json"
    threshold = 0.5
    if threshold_path.exists():
        threshold_data = json.loads(threshold_path.read_text())
        threshold = threshold_data.get("best_threshold", 0.5)
        logger.info("  ✓ threshold loaded (%.4f)", threshold)
    else:
        logger.warning("  ⚠ threshold_analysis.json not found — using 0.5")

    # ── DataPreprocessor ──────────────────────────────────────────────────────
    pp_path  = Path(data_cfg["paths"]["preprocessor"])
    if not pp_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {pp_path}")
    with open(pp_path, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("  ✓ preprocessor loaded")

    # ── FeatureEngineer ───────────────────────────────────────────────────────
    fe_path = Path(data_cfg["paths"]["feature_engineer"])
    if not fe_path.exists():
        raise FileNotFoundError(f"FeatureEngineer not found: {fe_path}")
    with open(fe_path, "rb") as f:
        feature_engineer = joblib.load(f)
    logger.info("  ✓ feature_engineer loaded")

    # ── Feature names ─────────────────────────────────────────────────────────
    fn_path = Path(data_cfg["paths"]["feature_names"])
    feature_names = json.loads(fn_path.read_text()) if fn_path.exists() else []
    logger.info("  ✓ feature_names loaded (%d features)", len(feature_names))
    
    # ── Model Version ─────────────────────────────────────────────────────────
    version_file = model_path.with_suffix(".version")
    if version_file.exists():
        model_version = version_file.read_text().strip()
        logger.info("  ✓ model version: %s", model_version)
    else:
        model_version = "v1.0.0"
        logger.warning("  ⚠ .version file not found — using default v1.0.0")
    
    #  ── Metrics ────────────────────────────────────────────────────────
    metrics_path = Path(cfg["artifacts"]["test_metrics"])
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        logger.info("  ✓ metrics loaded (roc_auc=%.4f)", metrics.get("test_roc_auc", 0))
    
    


    logger.info("All artifacts ready.")
    return {
        "model":            model,
        "preprocessor":     preprocessor,
        "feature_engineer": feature_engineer,
        "feature_names":    feature_names,
        "threshold":        float(threshold),
        "model_version":    model_version,
        "metrics":          metrics,
    }
