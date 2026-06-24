"""
src/models/predict_model.py — Inference helpers for the Fraud Detection System.

Provides:
  - load_model()   : deserialise a joblib model from disk.
  - predict_proba(): return raw fraud probabilities for a feature matrix.
  - predict()      : apply a decision threshold and return binary labels.

These are thin wrappers so that the inference pipeline and any downstream
consumer never import joblib / xgboost directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load and return a serialised model from *model_path*."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


def predict_proba(model, X) -> np.ndarray:
    """
    Return fraud probabilities (class-1 scores) for feature matrix *X*.

    Parameters
    ----------
    model : fitted XGBClassifier (or any sklearn-compatible estimator).
    X     : array-like of shape (n_samples, n_features).

    Returns
    -------
    probs : np.ndarray of shape (n_samples,).
    """
    return model.predict_proba(X)[:, 1]


def predict(model, X, threshold: float = 0.5) -> np.ndarray:
    """
    Return binary fraud predictions using *threshold*.

    Parameters
    ----------
    model     : fitted estimator.
    X         : feature matrix.
    threshold : decision cut-off (default 0.5; override with tuned value).

    Returns
    -------
    preds : np.ndarray of shape (n_samples,) with values in {0, 1}.
    """
    probs = predict_proba(model, X)
    return (probs >= threshold).astype(int)
