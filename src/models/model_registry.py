"""
src/models/model_registry.py — Local model serialisation and versioning.

Extracted from train.py (improvement #6 — Model Versioning):
  - save_model()    : dump model to canonical path + a versioned snapshot.
  - write_meta()    : persist threshold, metrics, CV stats as a .meta.json file.
  - register_mlflow(): log the model to the MLflow model registry.

The MLflow registry interaction (improvement #3) is co-located here because
it is conceptually part of "registering a trained model", not part of the
training loop itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.xgboost

from src.utils.config import _model_version

logger = logging.getLogger(__name__)


def save_model(model, model_path: str) -> tuple[str, str]:
    """
    Persist *model* to *model_path* (canonical) and a versioned snapshot.

    Returns
    -------
    (version_tag, versioned_path_str)
    """
    version_tag = _model_version(model_path)
    versioned_path = (
        Path(model_path).parent
        / f"{Path(model_path).stem}_{version_tag}{Path(model_path).suffix}"
    )

    joblib.dump(model, model_path)           # canonical path (for API)
    joblib.dump(model, versioned_path)       # versioned snapshot

    logger.info("Model saved → %s (%s)", model_path, version_tag)
    logger.info("Versioned snapshot → %s", versioned_path)
    return version_tag, str(versioned_path)


def write_meta(
    model_path: str,
    version_tag: str,
    run_id: str,
    run_ts: str,
    threshold: float,
    test_metrics: dict[str, float],
    cv_stats: dict[str, float],
) -> Path:
    """
    Write a JSON sidecar file (<model_path>.meta.json) with run metadata.
    Returns the path to the written file.
    """
    meta: dict[str, Any] = {
        "version":   version_tag,
        "run_id":    run_id,
        "timestamp": run_ts,
        "threshold": threshold,
        "metrics":   test_metrics,
        "cv":        cv_stats,
    }
    meta_path = Path(model_path).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Model metadata written → %s", meta_path)
    return meta_path


def register_mlflow(
    model,
    meta_path: Path,
    registered_model_name: str,
    version_tag: str,
) -> None:
    """
    Log *model* to the MLflow model registry and attach version/deploy tags.

    Improvement #3 — Experiment Tracking / Model Registry.
    """
    # ── 3. MLflow model registry ──────────────────────────────────────────────
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name=registered_model_name,
    )
    mlflow.log_artifact(str(meta_path), artifact_path="model_meta")
    mlflow.set_tags({
        "model_version": version_tag,
        "deploy_status": "APPROVED",
    })
    logger.info("Model registered in MLflow as '%s' (%s)", registered_model_name, version_tag)
