

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DASHBOARD_DIR = Path("data/dashboard")


def _save_json(obj: dict, name: str) -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = DASHBOARD_DIR / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  [dashboard] Saved {name}")


def _save_csv(df: pd.DataFrame, name: str, n: int = 5000) -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    sample = df.sample(min(n, len(df)), random_state=42) if len(df) > n else df.copy()
    sample.to_csv(DASHBOARD_DIR / name, index=False)
    print(f"  [dashboard] Saved {name}  ({len(sample):,} rows)")


def save_all(results: dict) -> None:
   
    print("\n[DASHBOARD] Saving artifacts to data/dashboard/ ...")

    df_raw      = results.get("df")
    train_clean = results.get("train_clean")
    X_train_raw = results.get("X_train")
    X_tr_eng    = results.get("X_train_sel")
    y_train_cl  = results.get("y_train_clean")
    y_train_bal = results.get("y_train_bal")
    X_train_bal = results.get("X_train_bal")
    X_val       = results.get("X_val")
    X_test      = results.get("X_test")
    y_val       = results.get("y_val")
    y_test      = results.get("y_test")
    selector    = results.get("selector")
    overview    = results.get("overview", {})
    selected    = results.get("selected_features", [])

    # 1. Overview
    _save_json(overview, "overview.json")

    # 2. Class distribution
    if df_raw is not None and "Class" in df_raw.columns:
        vc = df_raw["Class"].value_counts().sort_index()
        _save_json({
            "counts":    {str(k): int(v) for k, v in vc.items()},
            "pct":       {str(k): round(float(v / len(df_raw) * 100), 4)
                          for k, v in vc.items()},
            "imbalance": round(float(vc.iloc[0]) / max(float(vc.iloc[-1]), 1), 1),
            "total":     int(len(df_raw)),
        }, "class_dist.json")

    # 3. Missing values
    if df_raw is not None:
        null_pct = (df_raw.isnull().mean() * 100).round(3)
        _save_json({
            "missing_pct":     null_pct.to_dict(),
            "total_missing":   int(df_raw.isnull().sum().sum()),
            "cols_with_nulls": int((null_pct > 0).sum()),
        }, "missing.json")

    # 4. Split stats
    if all(v is not None for v in [X_train_bal, X_val, X_test]):
        _save_json({
            "n_train":          int(len(X_train_bal)),
            "n_val":            int(len(X_val)),
            "n_test":           int(len(X_test)),
            "fraud_train":      int(pd.Series(y_train_bal).sum()) if y_train_bal is not None else 0,
            "fraud_val":        int(pd.Series(y_val).sum())       if y_val       is not None else 0,
            "fraud_test":       int(pd.Series(y_test).sum())      if y_test      is not None else 0,
            "fraud_rate_train": float(pd.Series(y_train_bal).mean()) if y_train_bal is not None else 0,
            "fraud_rate_val":   float(pd.Series(y_val).mean())       if y_val       is not None else 0,
            "fraud_rate_test":  float(pd.Series(y_test).mean())      if y_test      is not None else 0,
        }, "split_stats.json")

    # 5. SMOTE stats
    if y_train_cl is not None and y_train_bal is not None:
        before = pd.Series(y_train_cl).value_counts().sort_index()
        after  = pd.Series(y_train_bal).value_counts().sort_index()
        _save_json({
            "before":       {str(k): int(v) for k, v in before.items()},
            "after":        {str(k): int(v) for k, v in after.items()},
            "before_total": int(len(y_train_cl)),
            "after_total":  int(len(y_train_bal)),
        }, "smote_stats.json")

    # 6. MI scores + RFE ranking
    if selector is not None:
        mi = selector.feature_scores_.get("mutual_info", {})
        _save_json({str(k): float(v) for k, v in mi.items()}, "mi_scores.json")

        rfe_imp = selector.feature_scores_.get("rfe_importance", {})
        if rfe_imp:
            sorted_feats = sorted(rfe_imp.items(), key=lambda x: x[1], reverse=True)
            rfe_ranking  = {feat: int(i + 1) for i, (feat, _) in enumerate(sorted_feats)}
            _save_json({
                "ranking":    rfe_ranking,
                "importance": {k: float(v) for k, v in rfe_imp.items()},
                "n_selected": int(len(selected)),
            }, "rfe_ranking.json")

    # 7. Selected features
    _save_json(list(selected), "selected_features.json")

    # 8. Skewness before / after
    if X_train_raw is not None and X_tr_eng is not None:
        df_b   = pd.DataFrame(X_train_raw)
        df_a   = pd.DataFrame(X_tr_eng)
        num_b  = df_b.select_dtypes(include=np.number)
        num_a  = df_a.select_dtypes(include=np.number)
        common = list(set(num_b.columns) & set(num_a.columns))
        _save_json({
            "before": {c: round(float(num_b[c].skew()), 4) for c in common},
            "after":  {c: round(float(num_a[c].skew()), 4) for c in common},
        }, "skewness.json")

    # 9. Raw data sample
    if df_raw is not None:
        _save_csv(df_raw, "df_raw_sample.csv", n=5000)

    # 10. Train clean sample
    if train_clean is not None:
        _save_csv(train_clean, "df_train_sample.csv", n=5000)

    # 11. Amount data
    if df_raw is not None and "Amount" in df_raw.columns:
        amount_df = df_raw[["Amount"]].copy()
        amount_df["log_amount"] = np.log1p(amount_df["Amount"])
        if "Class" in df_raw.columns:
            amount_df["Class"] = df_raw["Class"].values
        _save_csv(amount_df, "amount_data.csv", n=10000)

    # 12. Time data
    if df_raw is not None and "Time" in df_raw.columns:
        time_df = df_raw[["Time"]].copy()
        time_df["hour"] = (time_df["Time"] % 86400) // 3600
        if "Class" in df_raw.columns:
            time_df["Class"] = df_raw["Class"].values
        _save_csv(time_df, "time_data.csv", n=len(df_raw))

    # 13. Correlation matrix (train)
    if train_clean is not None:
        num_cols = train_clean.select_dtypes(include=np.number).columns[:20]
        corr     = train_clean[num_cols].corr().round(4)
        out_path = DASHBOARD_DIR / "correlation_matrix.csv"
        corr.to_csv(out_path)
        print(f"  [dashboard] Saved correlation_matrix.csv")

    total_files = len(list(DASHBOARD_DIR.glob("*")))
    print(f"[DASHBOARD] Done — {total_files} files saved to data/dashboard/\n")
