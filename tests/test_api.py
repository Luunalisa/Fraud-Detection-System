from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


VALID_TRANSACTION = {
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,  "V4": 1.3782,
    "V5": -0.3383, "V6":  0.4624, "V7": 0.2396,  "V8": 0.0987,
    "V9":  0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914,"V14": -0.3112,"V15": 1.4682,  "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040,  "V20": 0.2514,
    "V21": -0.0183,"V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891,"V27": 0.1336,  "V28": -0.0211,
    "Amount": 149.62,
    "Time": 0.0,
}


def make_mock_artifacts(prediction=0, probability=0.02):
    preprocessor = MagicMock()
    preprocessor.transform.side_effect = lambda df: df

    fe = MagicMock()
    fe.transform.side_effect = lambda df: df

    model = MagicMock()
    model.predict.return_value = np.array([prediction])
    model.predict_proba.return_value = np.array([[1 - probability, probability]])

    return {
        "model":            model,
        "preprocessor":     preprocessor,
        "feature_engineer": fe,
        "feature_names":    list(VALID_TRANSACTION.keys()),
        "threshold":        0.5,
        "model_version":    "test-v1.0.0",
        "metrics":          {},
    }


@pytest.fixture
def client_legitimate():
    mock_arts = make_mock_artifacts(prediction=0, probability=0.02)
    with patch("app.artifacts.get_artifacts", return_value=mock_arts):
        with patch("app.main.get_artifacts", return_value=mock_arts):
            from app.main import app
            with TestClient(app) as c:
                yield c


@pytest.fixture
def client_fraud():
    mock_arts = make_mock_artifacts(prediction=1, probability=0.97)
    with patch("app.artifacts.get_artifacts", return_value=mock_arts):
        with patch("app.main.get_artifacts", return_value=mock_arts):
            from app.main import app
            with TestClient(app) as c:
                yield c


class TestHealth:
    def test_health_returns_200(self, client_legitimate):
        r = client_legitimate.get("/health")
        assert r.status_code == 200

    def test_health_schema(self, client_legitimate):
        data = client_legitimate.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestPredict:
    def test_predict_legitimate(self, client_legitimate):
        r = client_legitimate.post("/predict", json=VALID_TRANSACTION)
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] == 0
        assert data["is_fraud"] is False

    def test_predict_fraud(self, client_fraud):
        r = client_fraud.post("/predict", json=VALID_TRANSACTION)
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] == 1
        assert data["is_fraud"] is True

    def test_predict_missing_field_returns_422(self, client_legitimate):
        incomplete = {k: v for k, v in VALID_TRANSACTION.items() if k != "Amount"}
        r = client_legitimate.post("/predict", json=incomplete)
        assert r.status_code == 422

    def test_predict_negative_amount_returns_422(self, client_legitimate):
        bad = {**VALID_TRANSACTION, "Amount": -100.0}
        r = client_legitimate.post("/predict", json=bad)
        assert r.status_code == 422


class TestPredictBatch:
    def _make_batch(self, n):
        return {"transactions": [VALID_TRANSACTION] * n}

    def test_batch_multiple_items(self, client_legitimate):
        r = client_legitimate.post("/predict/batch", json=self._make_batch(10))
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 10

    def test_batch_empty_returns_422(self, client_legitimate):
        r = client_legitimate.post("/predict/batch", json={"transactions": []})
        assert r.status_code == 422

    def test_batch_over_limit_returns_422(self, client_legitimate):
        r = client_legitimate.post("/predict/batch", json=self._make_batch(1001))
        assert r.status_code == 422


class TestMetrics:
    def test_metrics_returns_200(self, client_legitimate):
        r = client_legitimate.get("/metrics")
        assert r.status_code == 200