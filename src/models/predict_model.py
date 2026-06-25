

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

    return model.predict_proba(X)[:, 1]


def predict(model, X, threshold: float = 0.5) -> np.ndarray:

    probs = predict_proba(model, X)
    return (probs >= threshold).astype(int)
