"""
src/models/evaluate_model.py — Model evaluation utilities.

Extracted from train.py:
  - tune_threshold()        : F-beta sweep on the validation set (improvement #2).
  - cross_validate_model()  : StratifiedKFold AUPRC distribution (improvement #4).
  - log_feature_importance(): CSV artefact + top-N console log (improvement #5).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score, 
    recall_score
)
from sklearn.model_selection import StratifiedKFold
from src.utils.config import load_config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Threshold Tuning — F-beta sweep on validation set
# ──────────────────────────────────────────────────────────────────────────────

def tune_threshold(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    beta: float = 0.5,           # beta < 1 → favour precision over recall
    n_thresholds: int = 200,
) -> tuple[float, list[dict]]:
    """
    Sweep decision thresholds and pick the one that maximises F-beta on the
    validation set.  beta=0.5 weights precision twice as much as recall,
    which is appropriate for fraud where false positives are costly.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score, best_thresh = -1.0, 0.5
    sweep = []
    
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        score = fbeta_score(y_val, preds, beta=beta, zero_division=0)
        sweep.append({
            "threshold": round(float(t), 4),
            "fbeta":     round(float(score), 4),
            "precision": round(float(precision_score(y_val, preds, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_val, preds, zero_division=0)), 4),
        })
        if score > best_score:
            best_score = score
            best_thresh = float(t)


    logger.info(
        "Threshold tuned on val set — threshold=%.4f | F-%.1f=%.4f",
        best_thresh, beta, best_score,
    )
    return best_thresh , sweep


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cross-Validation — AUPRC distribution
# ──────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> dict[str, float]:
    """Run StratifiedKFold and return AUPRC stats (mean, std, min, max)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: list[float] = []

    for tr_idx, va_idx in skf.split(X_train, y_train):
        clone = xgb.XGBClassifier(**model.get_params())
        """
        clone.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            verbose=False,
        )
        probs = clone.predict_proba(X_train[va_idx])[:, 1]
        """

        clone.fit(
            X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            eval_set=[(X_train.iloc[va_idx], y_train.iloc[va_idx])],
            verbose=False,
        )
        probs = clone.predict_proba(X_train.iloc[va_idx])[:, 1]
        scores.append(average_precision_score(y_train[va_idx], probs))

    stats = {
        "cv_auprc_mean": float(np.mean(scores)),
        "cv_auprc_std":  float(np.std(scores)),
        "cv_auprc_min":  float(np.min(scores)),
        "cv_auprc_max":  float(np.max(scores)),
    }
    logger.info(
        "CV AUPRC — mean=%.4f ± %.4f  [%.4f – %.4f]",
        stats["cv_auprc_mean"], stats["cv_auprc_std"],
        stats["cv_auprc_min"],  stats["cv_auprc_max"],
    )
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 5. Feature Importance
# ──────────────────────────────────────────────────────────────────────────────

def log_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str] | None,
    top_n: int = 20,
    output_dir: Path = Path("artifacts/evaluation"),
) -> Path:
    """
    Dump feature importances (weight, gain, cover) to a CSV artefact and log
    the top-N to console.  Returns the CSV path.
    """
    cfg = load_config("configs/config.yaml")

    output_dir.mkdir(parents=True, exist_ok=True)
    importance_types = ["weight", "gain", "cover"]
    records: list[dict] = []

    for imp_type in importance_types:
        scores = model.get_booster().get_score(importance_type=imp_type)
        for feat, val in scores.items():
            records.append({"feature": feat, "importance_type": imp_type, "value": val})

    df = pd.DataFrame(records)

    # Rename f0, f1 … → actual feature names if provided
    if feature_names is not None:
        idx_map = {f"f{i}": name for i, name in enumerate(feature_names)}
        df["feature"] = df["feature"].map(lambda x: idx_map.get(x, x))

    csv_path = Path(cfg["artifacts"]["feature_importance"])
    df.to_csv(csv_path, index=False)

    # Top-N by 'gain'
    top = (
        df[df["importance_type"] == "gain"]
        .nlargest(top_n, "value")[["feature", "value"]]
        .reset_index(drop=True)
    )
    logger.info("Top-%d features by gain:\n%s", top_n, top.to_string())
    return csv_path
