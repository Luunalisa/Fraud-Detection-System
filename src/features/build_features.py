
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import joblib
from pathlib import Path 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    PowerTransformer,
)

from src.utils.logger import get_logger

from src.utils.config import load_config


logger = get_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        self.cfg      = load_config(config_path)
        self.fe_cfg   = self.cfg["feature_engineering"]
        self.pp_cfg   = self.cfg["preprocessing"]
        self.target   = self.cfg["data"]["target_column"]

        self._scaler: Optional[object]           = None
        self._power_transformer: Optional[object] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._skewed_cols: List[str]             = []
        self._scale_cols: List[str]              = []
        self._engineered_cols: List[str]         = []
        self.feature_names_out_: List[str]       = []
        
        # Amount feature saved stats (set during fit)
        self._amount_bins_       = None   # bin edges from qcut
        self._amount_high_thresh_ = None  # 95th percentile
        self._amount_mean_       = None   # mean for zscore
        self._amount_std_        = None   # std for zscore


    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        
        logger.info("=== Feature Engineering FIT ===")
        X = X.copy()

        # 1. Build new features first (no fitting needed)
        X = self._time_features(X)
        X = self._amount_features(X)
        if self.fe_cfg.get("interaction_features", False):
            X = self._interaction_features(X)

        # 2. Encode categoricals
        X = self._encode_categoricals(X, fit=True)

        # 3. Identify and fit skewness correction   #for Linear models
        #X = self._fit_skewness(X)

        # 4. Fit scaler on designated columns
        self._fit_scaler(X)

        self.feature_names_out_ = X.columns.tolist()
        logger.info(f"Feature engineering fitted. Output features: {len(self.feature_names_out_)}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        
        logger.info("=== Feature Engineering TRANSFORM ===")
        X = X.copy()

        X = self._time_features(X)
        X = self._amount_features(X)
        #if self.fe_cfg.get("interaction_features", False):            # for Linear models
            #X = self._interaction_features(X)
        X = self._encode_categoricals(X, fit=False)
        #X = self._apply_skewness(X)    # for Linear models
        X = self._apply_scaler(X)

        # Reorder to match fitted order
        for col in self.feature_names_out_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[[c for c in self.feature_names_out_ if c in X.columns]]
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)
    

    ##----------- Feature builders --------------------
    

    def _time_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fe_cfg.get("time_features", True):
            return X
        if "Time" not in X.columns:
            return X

        X["hour_of_day"]    = (X["Time"] % 86400) // 3600
        X["day_of_week"]    = (X["Time"] // 86400) % 7
        X["is_night"]       = ((X["hour_of_day"] >= 22) | (X["hour_of_day"] <= 6)).astype(int)
        X["is_weekend"]     = (X["day_of_week"] >= 5).astype(int)
        X["is_business_hr"] = ((X["hour_of_day"] >= 9) & (X["hour_of_day"] <= 17)).astype(int)

        # Sine/cosine encoding preserves cyclical nature for linear models 
        # together sin + cos represent hour as a point on a circle
        X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)  #Convert hour into a circle to capture cyclic behavior
        X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)

        new_cols = [
            "hour_of_day", "day_of_week", "is_night",
            "is_weekend", "is_business_hr", "hour_sin", "hour_cos"
        ]
        self._engineered_cols.extend(new_cols)
        # Drop raw Time AFTER extracting everything useful from it
        X = X.drop(columns=["Time"])
        logger.info(f"Time features created: {new_cols}")
        return X
    
    def _amount_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fe_cfg.get("amount_features", True):
            return X
        if "Amount" not in X.columns:
            return X

        fitting = self._amount_bins_ is None

        # ── Learn during fit, reuse during transform ───────────────
        if fitting:
            self._amount_mean_        = X["Amount"].mean()
            self._amount_std_         = X["Amount"].std()
            self._amount_high_thresh_ = X["Amount"].quantile(0.95)
            clip_upper                = X["Amount"].quantile(0.99)

            _, self._amount_bins_ = pd.qcut(
                X["Amount"].clip(upper=clip_upper),
                q=5, labels=False, duplicates="drop", retbins=True
            )
            self._amount_bins_[0]  = -float("inf")
            self._amount_bins_[-1] =  float("inf")

        # ── Apply ──────────────────────────────────────────────────
        X["log_amount"]    = np.log1p(X["Amount"])
        X["amount_zscore"] = (X["Amount"] - self._amount_mean_) / (self._amount_std_ + 1e-9)
        X["amount_sqrt"]   = np.sqrt(X["Amount"].clip(lower=0))

        X["amount_bin"] = pd.cut(
            X["Amount"],
            bins=self._amount_bins_,
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        ).astype(int)

        X["is_high_value"] = (X["Amount"] > self._amount_high_thresh_).astype(int)

        new_cols = [
            "log_amount", "amount_zscore", "amount_sqrt",
            "amount_bin", "is_high_value"
        ]

        if fitting:
            self._engineered_cols.extend(new_cols)
            logger.info(f"Amount features created: {new_cols}")

        return X

    def _interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:   #Its good to use with linear models, these benefit the most because they cannot automatically learn interactions.
        
        pairs = [("V1", "V2"), ("V3", "V4"), ("V14", "V17")]
        for a, b in pairs:
            if a in X.columns and b in X.columns:
                X[f"{a}_x_{b}"] = X[a] * X[b]
        if "log_amount" in X.columns and "V14" in X.columns:
            X["log_amount_x_V14"] = X["log_amount"] * X["V14"]
        logger.info("Interaction features created.")
        return X

    
    ## -------------------- Encoding ---------------------
    

    def _encode_categoricals(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            return X

        for col in cat_cols:
            n_unique = X[col].nunique()
            if n_unique == 2:
                # Binary → label encode
                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self._label_encoders[col] = le
                else:
                    le = self._label_encoders.get(col)
                    if le:
                        X[col] = le.transform(X[col].astype(str))
            elif n_unique <= 15:
                # Low cardinality → one-hot encode
                if fit:
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                    self._label_encoders[f"_ohe_{col}"] = dummies.columns.tolist()
                else:
                    expected_cols = self._label_encoders.get(f"_ohe_{col}", [])
                    dummies       = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    for ec in expected_cols:
                        if ec not in dummies.columns:
                            dummies[ec] = 0
                    X = pd.concat([X.drop(columns=[col]), dummies[expected_cols]], axis=1)
            else:
                # High cardinality → frequency encoding
                if fit:
                    freq = X[col].value_counts(normalize=True).to_dict()
                    self._label_encoders[f"_freq_{col}"] = freq
                else:
                    freq = self._label_encoders.get(f"_freq_{col}", {})
                X[col] = X[col].map(freq if fit else self._label_encoders.get(f"_freq_{col}", {})).fillna(0)

        logger.info(f"Categorical encoding applied to {len(cat_cols)} column(s).")
        return X

 
    ## --------------------- Skewness # Good for linear models ---------------------------
 

    def _fit_skewness(self, X: pd.DataFrame) -> pd.DataFrame:                 
        
        num_cols = X.select_dtypes(include=np.number).columns
        num_cols = [c for c in num_cols if c != self.target]

        skewness = X[num_cols].skew().abs()
        self._skewed_cols = skewness[skewness > 1.0].index.tolist()

        if self._skewed_cols:
            logger.info(f"Skewed columns (|skew|>1): {self._skewed_cols}")
            self._power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
            X[self._skewed_cols]    = self._power_transformer.fit_transform(X[self._skewed_cols])
        else:
            logger.info("No columns with |skew| > 1.0 — skewness step skipped.")

        return X

    def _apply_skewness(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._skewed_cols and self._power_transformer:
            cols_present = [c for c in self._skewed_cols if c in X.columns]
            X[cols_present] = self._power_transformer.transform(X[cols_present])
        return X

    ## ------------------------ Scaling ------------------------------------
    

    def _fit_scaler(self, X: pd.DataFrame) -> None:  # for train x
        scale_cols = [
            c for c in self.pp_cfg.get("scale_columns", []) if c in X.columns
        ]
        self._scale_cols = scale_cols
        if not scale_cols:
            return

        scaler_name = self.pp_cfg.get("scaler", "standard")
        scaler_map  = {
            "standard": StandardScaler(),
            "minmax":   MinMaxScaler(),
            "robust":   RobustScaler(),
        }
        self._scaler = scaler_map.get(scaler_name, StandardScaler())
        self._scaler.fit(X[scale_cols])
        logger.info(f"Scaler ({scaler_name}) fitted on: {scale_cols}")

    def _apply_scaler(self, X: pd.DataFrame) -> pd.DataFrame:  # for val/test x
        if self._scaler and self._scale_cols:
            cols_present = [c for c in self._scale_cols if c in X.columns]
            X[cols_present] = self._scaler.transform(X[cols_present])
        return X

    ## ------------------ Persistence -----------------------

    def save_scaler(self, path: str = "data/processed/scaler.joblib") -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path)
        logger.info(f"Scaler saved → {path}")

    def save_feature_names(self, path: str = "artifacts/feature_engineering/feature_names.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.feature_names_out_, f, indent=2)
        logger.info(f"Feature names saved → {path}")
        
    def save_feature_engineer(self, path: str = "artifacts/feature_engineering/feature_engineer.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"FeatureEngineer saved → {path}")
    
