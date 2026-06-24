"""
tests/test_model.py — Unit tests for model training and evaluation utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

from src.models.evaluate_model import cross_validate_model, tune_threshold
from src.models.predict_model import predict, predict_proba


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_dataset():
    """Tiny balanced dataset for fast unit tests."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    y = (rng.random(200) > 0.5).astype(int)
    import pandas as pd
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)


@pytest.fixture
def trained_model(small_dataset):
    X, y = small_dataset
    model = xgb.XGBClassifier(n_estimators=10, random_state=42, n_jobs=1)
    model.fit(X, y)
    return model, X, y


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_predict_proba_shape(trained_model):
    model, X, _ = trained_model
    probs = predict_proba(model, X)
    assert probs.shape == (len(X),)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_predict_binary(trained_model):
    model, X, _ = trained_model
    preds = predict(model, X, threshold=0.5)
    assert set(preds).issubset({0, 1})


def test_tune_threshold_returns_float(trained_model):
    model, X, y = trained_model
    probs = predict_proba(model, X)
    thresh = tune_threshold(probs, y.values, beta=0.5)
    if isinstance(thresh, tuple):
        thresh = thresh[0]

    assert isinstance(thresh, float)
    assert 0.0 < thresh < 1.0



def test_cross_validate_keys(trained_model, small_dataset):
    model, _, _ = trained_model
    X, y = small_dataset
    stats = cross_validate_model(model, X, y, n_splits=3)
    assert all(k in stats for k in ["cv_auprc_mean", "cv_auprc_std", "cv_auprc_min", "cv_auprc_max"])
