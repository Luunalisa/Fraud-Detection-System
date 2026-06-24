"""
src/pipelines/training_pipeline.py — End-to-end training orchestration.

This module contains train_model(), which was previously the main function in
train.py.  It wires together every improvement in the correct order:

  1. Hyperparameter Tuning    — src/models/train_model.py  (run_optuna)
  2. Threshold Tuning         — src/models/evaluate_model.py (tune_threshold)
  3. Experiment Tracking      — MLflow run wrapping the whole pipeline
  4. Cross-Validation         — src/models/evaluate_model.py (cross_validate_model)
  5. Feature Importance       — src/models/evaluate_model.py (log_feature_importance)
  6. Model Versioning         — src/models/model_registry.py (save_model / write_meta)
  7. Data Drift Detection     — src/utils/helpers.py (psi_report)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    roc_auc_score,
)

from src.data_pipeline import prepare_dataset
from src.models.evaluate_model import (
    cross_validate_model,
    log_feature_importance,
    tune_threshold,
)
from src.models.model_registry import register_mlflow, save_model, write_meta
from src.models.train_model import run_optuna
from src.utils.config import load_config
from src.utils.helpers import psi_report

from src.utils.logger import get_logger

logger = get_logger(__name__)


## -------------------- Main training function-----------------------

def train_model(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    #run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # ── Data ──────────────────────────────────────────────────────────────────
    #X_train, X_test, X_val, y_train, y_test, y_val = prepare_dataset()
    X_train = pd.read_parquet("data/precessed/X_train_bal.parquet")
    X_val   = pd.read_parquet("data/precessed/X_val.parquet")
    X_test  = pd.read_parquet("data/precessed/X_test.parquet")

    y_train = pd.read_parquet("data/precessed/y_train_bal.parquet")["Class"]
    y_val   = pd.read_parquet("data/precessed/y_val.parquet")["Class"]
    y_test  = pd.read_parquet("data/precessed/y_test.parquet")["Class"]

    feature_names: list[str] | None = cfg.get("features", {}).get("names")
    logger.info(f"Feature names from config: {feature_names}")

    # ── 7. Data Drift Detection ───────────────────────────────────────────────
    logger.info("Running PSI drift checks …")

    import pandas as pd
    X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train # Convert DataFrames to NumPy arrays
    X_val_arr   = X_val.values   if isinstance(X_val,   pd.DataFrame) else X_val
    X_test_arr  = X_test.values  if isinstance(X_test,  pd.DataFrame) else X_test

    drift_train_val  = psi_report(X_train_arr, X_val_arr,  label="train_vs_val",  feature_names=feature_names)
    drift_train_test = psi_report(X_train_arr, X_test_arr, label="train_vs_test", feature_names=feature_names)
    
    """
        I wanted to separate them to know exactly where high drift features are val or test
           so i commented this part and continue working with the part after this one
           
    high_drift_cols = [ 
        feat for feat, psi_val in {**drift_train_val, **drift_train_test}.items()  #{**drift_train_val, **drift_train_test} Merge both drift dictionaries
        if psi_val > cfg["drift"].get("psi_warn_threshold", 0.2)
    ]  # This selects features whose PSI exceeds the warning threshold.
    if high_drift_cols:
        logger.warning("⚠  High PSI detected for features: %s", high_drift_cols)
    """

    threshold = cfg["drift"].get("psi_warn_threshold", 0.2)

    # Features with high drift between train and validation
    high_drift_val = [
        feat for feat, psi_val in drift_train_val.items()
        if psi_val > threshold
    ]
    # Features with high drift between train and test
    high_drift_test = [
        feat for feat, psi_val in drift_train_test.items()
        if psi_val > threshold
    ]
    
    # Log warnings separately
    if high_drift_val:
        logger.warning(
            "⚠ High PSI detected (train_vs_val): %s",
            high_drift_val
        ) 

    if high_drift_test:
        logger.warning(
            "⚠ High PSI detected (train_vs_test): %s",
            high_drift_test
        )

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"].get("tracking_uri", "mlruns")) #Sets where MLflow stores experiments,Save experiments locally inside ./mlruns/
    mlflow.set_experiment(cfg["mlflow"].get("experiment_name", "fraud_detection"))

    with mlflow.start_run(run_name=f"xgb_{run_ts}") as run:
        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

        # Log config & drift artifacts
        mlflow.log_artifact(config_path)  # Uploads configuration file
        mlflow.log_dict(drift_train_val,  "drift/psi_train_val.json")  #Saves drift report as JSON artifact
        mlflow.log_dict(drift_train_test, "drift/psi_train_test.json")
        mlflow.log_param("high_drift_val_features", str(high_drift_val)) 
        mlflow.log_param("high_drift_test_features", str(high_drift_test)) #Logs dangerous drift features

        # ── 1. Optuna tuning ─────────────────────────────────────────────────
        if cfg["optuna"].get("enabled", True):   #Checks if tuning is enabled
            logger.info("Starting Optuna hyperparameter search …")
            best_params, best_cv_score = run_optuna(X_train, y_train, cfg)
            mlflow.log_params(best_params)                                 #Stores parameters in MLflow
            mlflow.log_metric("optuna_best_cv_auprc", best_cv_score)       #Logs best tuning metric
        else:
            # Fall back to config values
            best_params = {                                           # if optuna disabled Use parameters from config file
                "n_estimators":     cfg["training"]["n_estimators"],
                "max_depth":        cfg["training"]["max_depth"],
                "learning_rate":    cfg["training"]["learning_rate"],
                "scale_pos_weight": cfg["training"].get("scale_pos_weight", 577),
            }
            mlflow.log_params(best_params)

        # ── Build final model ─────────────────────────────────────────────────
        model = xgb.XGBClassifier(
            **best_params,
            eval_metric="aucpr",       # Very important for this imbalanced fraud dataset
            random_state=42,           # Makes results reproducible
            n_jobs=-1,                 # Use all CPU cores
        )

        # ── 4. Cross-Validation ───────────────────────────────────────────────
        n_splits = cfg["cross_validation"].get("n_splits", 5)
        cv_stats = cross_validate_model(model, X_train, y_train, n_splits=n_splits)
        mlflow.log_metrics(cv_stats)

        # ── Final fit (full train set, early-stop on val) ─────────────────────
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],   #Tracks performance during training
            verbose=50,                           #Prints logs every 50 boosting rounds
        )

        # ── 2. Threshold Tuning ───────────────────────────────────────────────
        val_probs = model.predict_proba(X_val)[:, 1]          #It selects ONLY the fraud probability (class 1)
        val_auprc = average_precision_score(y_val, val_probs)  # model evaluation on validation data
        logger.info("Validation AUPRC: %.4f", val_auprc)
        mlflow.log_metric("val_auprc", val_auprc)           # stores the metric in MLflow so i can compare experiments later (experiment tracking)
        
        ## Model decision optimization 
        beta = cfg["training"].get("fbeta_beta", 0.5)   # loads the beta value for the F-beta score
        threshold = tune_threshold(val_probs, y_val, beta=beta)
        mlflow.log_param("decision_threshold", threshold)   # Stores the chosen threshold in MLflow

        # ── Test-set evaluation (run only once) ───────────────────────────────
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= threshold).astype(int)

        roc_auc    = roc_auc_score(y_test, test_probs)
        avg_prec   = average_precision_score(y_test, test_probs)
        precision  = precision_score(y_test, test_preds, zero_division=0)
        report     = classification_report(y_test, test_preds, output_dict=True)
        cm         = confusion_matrix(y_test, test_preds)

        test_metrics = {
            "test_roc_auc":    roc_auc,
            "test_auprc":      avg_prec,
            "test_precision":  precision,
            "test_recall":     report["1"]["recall"],
            "test_f1":         report["1"]["f1-score"],
        }
        mlflow.log_metrics(test_metrics)

        logger.info("\n" + classification_report(y_test, test_preds))
        logger.info("ROC-AUC: %.4f", roc_auc)
        logger.info("AUPRC:   %.4f", avg_prec)
        logger.info("Confusion matrix:\n%s", cm)

        # ── 5. Feature Importance ─────────────────────────────────────────────
        fi_path = log_feature_importance(model, feature_names)
        mlflow.log_artifact(str(fi_path), artifact_path="feature_importance")

        # ── CI/CD gate ────────────────────────────────────────────────────────
        min_prec = cfg["training"]["min_precision"]
        if precision < min_prec:
            mlflow.set_tag("deploy_status", "BLOCKED")
            raise ValueError(
                f"Precision {precision:.3f} below minimum {min_prec}. "
                "Deployment blocked."
            )

        # ── 6. Model Versioning + Registry ───────────────────────────────────
        model_path  = cfg["model"]["path"]
        version_tag, _ = save_model(model, model_path)
        meta_path   = write_meta(
            model_path, version_tag, run_id, run_ts,
            threshold, test_metrics, cv_stats,
        )
        register_mlflow(
            model, meta_path,
            registered_model_name=cfg["mlflow"].get("registered_model_name", "FraudXGB"),
            version_tag=version_tag,
        )

        logger.info("✅ Training complete — version=%s | run_id=%s", version_tag, run_id)

    return model, test_metrics
