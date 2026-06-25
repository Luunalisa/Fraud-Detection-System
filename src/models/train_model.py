
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyperparameter Tuning — Optuna
# ──────────────────────────────────────────────────────────────────────────────

def _make_objective(X_train, y_train, cfg: dict):
    """Return an Optuna objective that runs StratifiedKFold AUPRC."""
    cv_cfg = cfg["cross_validation"]
    n_splits = cv_cfg.get("n_splits", 5)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":        trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth":           trial.suggest_int("max_depth", 3, 10),
            "learning_rate":       trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":           trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":    trial.suggest_int("min_child_weight", 1, 20),
            "gamma":               trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":           trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":          trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":    cfg["training"].get("scale_pos_weight", 577),
            "eval_metric":         "aucpr",
            "random_state":        42,
            "n_jobs":              -1,
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores: list[float] = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            #X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            #y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            probs = model.predict_proba(X_va)[:, 1]
            fold_scores.append(average_precision_score(y_va, probs))

            # Optuna pruning — stop unpromising trials early
            trial.report(np.mean(fold_scores), fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def run_optuna(X_train, y_train, cfg: dict) -> dict[str, Any]:
    """Run Optuna study and return best hyperparameters."""
    n_trials = cfg["optuna"].get("n_trials", 50)
    timeout  = cfg["optuna"].get("timeout_seconds", None)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name="fraud_xgb_tuning",
    )
    study.optimize(
        _make_objective(X_train, y_train, cfg),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )
    logger.info(
        "Optuna finished — best AUPRC: %.4f | best params: %s",
        study.best_value,
        study.best_params,
    )
    return study.best_params, study.best_value
