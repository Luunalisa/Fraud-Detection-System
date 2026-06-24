
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    VarianceThreshold,
    mutual_info_classif,
)

from src.utils.logger import get_logger

from src.utils.config import load_config

logger = get_logger(__name__)


class FeatureSelector:

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        self.cfg    = load_config(config_path)
        self.fs_cfg = self.cfg["feature_selection"]
        self.bl_cfg = self.cfg["balancing"]
        self.target = self.cfg["data"]["target_column"]
        self.rs     = self.cfg["project"]["random_state"]

        # Populated after fit_select()
        self.selected_features_: List[str]      = []
        self.feature_scores_:    Dict[str, Dict] = {}
        self.removed_variance_:  List[str]       = []
        self.removed_corr_:      List[str]       = []

    def fit_select(
        self, train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        
        logger.info("=== Feature Selection FIT (training data only) ===")

        # Separate features and target
        X = train_df.drop(columns=[self.target], errors="ignore")
        y = train_df[self.target]

        n_input = X.shape[1]
        logger.info(f"Input:  {X.shape[0]:,} rows x {n_input} features")

        # ── Selection steps (all fitted on X_train only) ──────────────
        X = self._remove_low_variance(X)
        X = self._remove_high_correlation(X)
        X = self._rfe_selection(X, y)      # final elimination
        self._mutual_info_scores(X, y)     # scores for reporting — no filtering

        self.selected_features_ = X.columns.tolist()

        logger.info(
            f"Output: {len(self.selected_features_)} features selected "
            f"from {n_input} input features"
        )
        logger.info(f"Selected: {self.selected_features_}")
        return X, self.selected_features_

    def _balance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        
        strategy = self.bl_cfg.get("strategy", "smote")
        sampling = self.bl_cfg.get("sampling_strategy", "auto")
        k        = int(self.bl_cfg.get("k_neighbors", 5))

        if strategy == "none":
            logger.info("Balancing skipped — strategy: none")
            return X_train, y_train

        logger.info(
            f"Before {strategy.upper()}: "
            f"{y_train.value_counts().to_dict()}"
        )

        if strategy == "smote":
            sampler = SMOTE(
                sampling_strategy=sampling,
                k_neighbors=k,
                random_state=self.rs,
                #n_jobs=-1,
            )
        elif strategy == "adasyn":
            sampler = ADASYN(
                sampling_strategy=sampling,
                n_neighbors=k,
                random_state=self.rs,
                #n_jobs=-1,
            )
        else:
            raise ValueError(
                f"Unknown balancing strategy: '{strategy}'. "
                f"Choose one of: smote | adasyn | none"
            )

        X_res, y_res = sampler.fit_resample(X_train, y_train)

        # fit_resample() returns numpy arrays — restore column names and Series
        X_bal = pd.DataFrame(X_res, columns=X_train.columns)
        y_bal = pd.Series(y_res, name=self.target)

        logger.info(
            f"After  {strategy.upper()}: "
            f"{y_bal.value_counts().to_dict()}"
        )
        return X_bal, y_bal

    def log_summary(
        self,
        X_train_bal: pd.DataFrame,
        X_val:       pd.DataFrame,
        X_test:      pd.DataFrame,
        y_train_bal: pd.Series,
        y_val:       pd.Series,
        y_test:      pd.Series,
    ) -> None:
        
        logger.info("─" * 62)
        logger.info("SPLIT & BALANCE SUMMARY")
        logger.info(
            f"  Train (balanced) : {len(X_train_bal):>8,} rows | "
            f"fraud: {int(y_train_bal.sum()):>6,} "
            f"({y_train_bal.mean():.4%})"
        )
        logger.info(
            f"  Validation       : {len(X_val):>8,} rows | "
            f"fraud: {int(y_val.sum()):>6,} "
            f"({y_val.mean():.4%})"
        )
        logger.info(
            f"  Test             : {len(X_test):>8,} rows | "
            f"fraud: {int(y_test.sum()):>6,} "
            f"({y_test.mean():.4%})"
        )
        logger.info(f"  Features selected  : {len(self.selected_features_)}")
        logger.info(f"  Removed (variance) : {self.removed_variance_ or 'none'}")
        logger.info(f"  Removed (corr)     : {self.removed_corr_ or 'none'}")
        logger.info("─" * 62)

    def save_results(
        self,
        X_train_bal: pd.DataFrame,
        X_val:       pd.DataFrame,
        X_test:      pd.DataFrame,
        y_train_bal: pd.Series,
        y_val:       pd.Series,
        y_test:      pd.Series,
        base_path:   str = "data/processed/",
    ) -> None:
        
        out = Path(base_path)
        out.mkdir(parents=True, exist_ok=True)

        # ── Feature DataFrames ──────────────────────────────────────
        for name, df in [
            ("X_train_bal", X_train_bal),
            ("X_val",       X_val),
            ("X_test",      X_test),
        ]:
            # fit_resample can return numpy arrays in some versions
            if isinstance(df, np.ndarray):
                df = pd.DataFrame(df, columns=self.selected_features_)
            df.to_parquet(out / f"{name}.parquet", index=False)
            logger.info(f"Saved {name}.parquet  shape: {df.shape}")

        # ── Target Series ───────────────────────────────────────────
        for name, s in [
            ("y_train_bal", y_train_bal),
            ("y_val",       y_val),
            ("y_test",      y_test),
        ]:
            # parquet requires a DataFrame — wrap Series first
            pd.Series(s).to_frame(name=self.target).to_parquet(
                out / f"{name}.parquet", index=False
            )
            logger.info(f"Saved {name}.parquet")

        # ── Metadata JSON ───────────────────────────────────────────
        with open(out / "selected_features.json", "w") as f:
            json.dump(self.selected_features_, f, indent=2)

        mi_scores = self.feature_scores_.get("mutual_info", {})
        with open(out / "mi_scores.json", "w") as f:
            json.dump(mi_scores, f, indent=2)

        logger.info(f"All splits and metadata saved -> {out.resolve()}")


    ## --------------- Feature Selection and Feaature Scores -----------------------


    def _remove_low_variance(self, X: pd.DataFrame) -> pd.DataFrame:
      
        threshold = self.fs_cfg.get("variance_threshold", 0.01)
        num_cols  = X.select_dtypes(include=np.number).columns.tolist()

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X[num_cols])

        # get_support() returns a boolean array — True = keep, False = remove
        kept    = np.array(num_cols)[selector.get_support()].tolist()
        removed = [c for c in num_cols if c not in kept]
        self.removed_variance_ = removed

        if removed:
            logger.info(
                f"Variance filter (threshold={threshold}): "
                f"removed {len(removed)} column(s) -> {removed}"
            )
            X = X.drop(columns=removed)
        else:
            logger.info(
                f"Variance filter (threshold={threshold}): "
                f"no columns removed"
            )

        return X

    def _remove_high_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
    
        threshold = self.fs_cfg.get("correlation_threshold", 0.95)
        num_cols  = X.select_dtypes(include=np.number).columns

        # Full absolute correlation matrix
        corr  = X[num_cols].corr().abs()

        # Keep only the upper triangle (k=1 excludes the diagonal).
        # The matrix is symmetric so checking one triangle is enough.
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        # Drop any column where at least one correlation exceeds threshold
        to_drop = [
            col for col in upper.columns
            if any(upper[col] > threshold)
        ]
        self.removed_corr_ = to_drop

        if to_drop:
            logger.info(
                f"Correlation filter (threshold={threshold}): "
                f"removed {len(to_drop)} column(s) -> {to_drop}"
            )
            X = X.drop(columns=to_drop)
        else:
            logger.info(
                f"Correlation filter (threshold={threshold}): "
                f"no columns removed"
            )

        return X

    def _mutual_info_scores(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
    
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            logger.warning("No numeric columns available for MI scoring.")
            return {}

        mi = mutual_info_classif(
            X[num_cols].fillna(0),   # fillna avoids crash on any remaining nulls
            y,
            discrete_features=False,  # all our features are continuous
            random_state=self.rs,
        )

        mi_scores = dict(zip(num_cols, mi.round(5)))
        self.feature_scores_["mutual_info"] = mi_scores

        top5 = sorted(
            mi_scores.items(), key=lambda x: x[1], reverse=True
        )[:5]
        logger.info(f"Top-5 MI features: {top5}")

        return mi_scores

    def _rfe_selection(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
       
        method     = self.fs_cfg.get("method", "rfe")
        n_features = self.fs_cfg.get("n_features", 20)
        num_cols   = X.select_dtypes(include=np.number).columns.tolist()

        if not num_cols:
            logger.warning("No numeric columns for feature selection — skipping.")
            return X

        # ── Method 1: RFE  ──────────────────
        if method == "rfe":

            estimator = RandomForestClassifier(
                n_estimators=50,   # 50 trees — enough for reliable importance
                max_depth=5,       # shallow trees — fast, avoids overfitting
                random_state=self.rs,
                n_jobs=-1,         # use all CPU cores
            )

            rfe = RFE(
                estimator=estimator,
                n_features_to_select=min(n_features, len(num_cols)),
                step=5,            # remove 5 weakest features per iteration
            )

            rfe.fit(X[num_cols].fillna(0), y)

            # rfe.support_ is a boolean array: True = kept, False = removed
            kept = np.array(num_cols)[rfe.support_].tolist()

            # Store importances from the final fitted estimator (on kept features)
            if hasattr(rfe.estimator_, "feature_importances_"):
                fi = dict(
                    zip(num_cols, rfe.estimator_.feature_importances_.round(5))
                )
                self.feature_scores_["rfe_importance"] = fi

            # Preserve any non-numeric columns unchanged
            other_cols = [c for c in X.columns if c not in num_cols]
            X = X[kept + other_cols]

            logger.info(
                f"RFE: selected {len(kept)} features "
                f"from {len(num_cols)} numeric columns"
            )

        # ── Method 2: Importance threshold (faster, less rigorous) ───
        elif method == "importance":

            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.rs,
                n_jobs=-1,
            )
            rf.fit(X[num_cols].fillna(0), y)

            fi   = pd.Series(rf.feature_importances_, index=num_cols)
            kept = fi.nlargest(n_features).index.tolist()

            self.feature_scores_["importance"] = fi.to_dict()
            X = X[kept]

            logger.info(
                f"Importance selection: kept top {len(kept)} features"
            )

        # ── Method 3: None — keep all (for experiments/debugging) ───
        else:
            logger.info(
                f"Feature selection method '{method}' — "
                f"keeping all {len(num_cols)} features"
            )

        return X
