"""
app/inference.py — Applies the full preprocessing + feature engineering chain,
then runs the XGBoost model.

Chain (mirrors run_pipeline exactly):
  raw DataFrame
    → DataPreprocessor.transform()      (step 4)
    → FeatureEngineer.transform()       (step 6)
    → model.predict_proba()             (XGBoost)

No fitting happens here — all fitted state comes from loaded artifacts.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Raw columns that come in from the API request (before preprocessing)
RAW_COLUMNS = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "Amount","Time",
]


def run_single(request_dict: dict, artifacts: dict) -> dict:
    """
    Score a single transaction.

    Parameters
    ----------
    request_dict : dict from PredictRequest.dict()
    artifacts    : dict returned by get_artifacts()

    Returns
    -------
    dict with prediction, probability, threshold, model_version, is_fraud
    """
    t0 = time.perf_counter()

    df = pd.DataFrame([request_dict])[RAW_COLUMNS]

    prob, pred = _score(df, artifacts)

    latency_ms = (time.perf_counter() - t0) * 1000
    
    """
    Why measure latency?

    In an API, latency tells you how fast your model responds.

    Typical inference latencies:

    < 10 ms → excellent
    10–100 ms → very good
    100–500 ms → acceptable for many applications
    > 1 second → may feel slow to users
    """
    logger.info(
        "single inference | pred=%d | prob=%.4f | threshold=%.4f | latency=%.1fms",
        pred[0], prob[0], artifacts["threshold"], latency_ms,
    )

    return {
        "prediction":    int(pred[0]),
        "probability":   round(float(prob[0]), 6),
        "threshold":     artifacts["threshold"],
        "model_version": artifacts["model_version"],
        "is_fraud":      bool(pred[0] == 1),
    }


def run_batch(requests: list[dict], artifacts: dict) -> dict:
    """
    Score a batch of transactions efficiently.
    XGBoost is vectorised — batching is much faster than looping.
    """
    t0 = time.perf_counter()

    df = pd.DataFrame(requests)[RAW_COLUMNS]

    probs, preds = _score(df, artifacts)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "batch inference | n=%d | fraud_count=%d | latency=%.1fms",
        len(preds), int(preds.sum()), latency_ms,
    )

    return {
        "predictions":   [int(p) for p in preds],
        "probabilities": [round(float(p), 6) for p in probs],
        "threshold":     artifacts["threshold"],
        "model_version": artifacts["model_version"],
        "fraud_count":   int(preds.sum()),
        "total":         len(preds),
    }


def _score(df: pd.DataFrame, artifacts: dict):
    """
    Internal: apply preprocessing chain and return (probs, preds).
    """
    preprocessor     = artifacts["preprocessor"]
    feature_engineer = artifacts["feature_engineer"]
    model            = artifacts["model"]
    threshold        = artifacts["threshold"]

    # Step 4 — DataPreprocessor (outlier flagging, dtype fixes, imputation)
    df_clean = preprocessor.transform(df)

    # Step 6 — FeatureEngineer (time/amount features, scaling)
    df_eng = feature_engineer.transform(df_clean)

    # Align columns to what the model was trained on
    expected = artifacts["feature_names"]
    for col in expected:
        if col not in df_eng.columns:
            df_eng[col] = 0.0
    df_eng = df_eng[expected]

    probs = model.predict_proba(df_eng)[:, 1]
    preds = (probs >= threshold).astype(int)

    return probs, preds
