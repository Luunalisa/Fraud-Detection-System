
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.data_ingestion     import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.features.build_features  import FeatureEngineer
from src.features.feature_selection import FeatureSelector
from src.visualization.eda_plots  import EDAVisualizer
from src.utils.logger             import get_logger
from src.utils.config             import load_config

from src.visualization.save_dashboard_artifacts import save_all

logger = get_logger(__name__)


def run_pipeline(config_path: str = "configs/data_config.yaml") -> dict:
    t0  = time.time()
    cfg = load_config(config_path)

    # ── 1. Ingest ────────────────────────────────────────────────
    logger.info("\n[1/7] DATA INGESTION")
    ingestor   = DataIngestion(config_path)
    df_raw     = ingestor.load()
    overview   = ingestor.overview_report(df_raw)
    ingestor.print_overview(df_raw)
    ingestor.save_interim(df_raw)


    # ── 2. SPLIT FIRST — before ANY fitting ──────────────────────
    logger.info("\n[3/7] SPLIT FIRST (before fitting anything)")
    target    = cfg["data"]["target_column"]
    #drop_cols = cfg["data"].get("drop_columns", [])
    rs        = cfg["project"]["random_state"]

    X_raw = df_raw.drop(columns=[target], errors="ignore")
    y_raw = df_raw[target]

    test_size = cfg["split"]["test_size"]
    val_size  = cfg["split"]["val_size"]
    stratify  = y_raw if cfg["split"].get("stratify", True) else None

    # First cut: test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_raw,
        test_size=test_size,
        random_state=rs,
        stratify=stratify
    )
    # Second cut: validation from remaining
    val_ratio = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=rs,
        stratify=y_temp if stratify is not None else None
    )

    logger.info(f"Train: {X_train_raw.shape} | "
                f"Val: {X_val_raw.shape} | "
                f"Test: {X_test.shape}")

    # ── 3. Preprocess — fit on train ONLY ────────────────────────
    logger.info("\n[4/7] DATA PREPROCESSING")

    # Add target back temporarily so preprocessor can reference it
    train_df = X_train_raw.copy()
    train_df[target] = y_train.values
    val_df   = X_val_raw.copy()
    val_df[target]   = y_val.values
    test_df  = X_test.copy()
    test_df[target]  = y_test.values

    preprocessor = DataPreprocessor(config_path)
    train_clean  = preprocessor.fit_transform(train_df)  # FIT on train only
    val_clean    = preprocessor.transform(val_df)         # APPLY to val
    test_clean   = preprocessor.transform(test_df)        # APPLY to test

    preprocessor.save(train_clean, "data/interim/train_clean.parquet")
    logger.info("Preprocessor fitted on train only — applied to val and test")
    
    preprocessor.save_preprocessor()

    # ── 4. EDA on train only (no leakage from val/test) ──────────
    logger.info("\n[5/7] EDA VISUALIZATIONS (train set only)")
    viz    = EDAVisualizer(save_dir=cfg["paths"]["figures"])
    y_train_clean   = train_clean[target]

    viz.plot_overview_table(train_clean)
    viz.plot_missing_heatmap(train_clean)
    viz.plot_class_distribution(y_train_clean)
    viz.plot_feature_histograms(train_clean, target=target)
    viz.plot_boxplots(train_clean, target=target)
    viz.plot_correlation_heatmap(train_clean)
    viz.plot_feature_target_correlation(train_clean, target=target)
    viz.plot_amount_distribution(train_clean)
    viz.plot_time_analysis(train_clean)
    #viz.plot_eda_dashboard(train_clean, y=y_train_clean)

    # ── 5. Feature Engineering — fit on train ONLY ───────────────
    logger.info("\n[6/7] FEATURE ENGINEERING")

    X_train_clean = train_clean.drop(columns=[target], errors="ignore")
    X_val_clean  = val_clean.drop(columns=[target],   errors="ignore")
    X_test_clean = test_clean.drop(columns=[target],  errors="ignore")

    fe          = FeatureEngineer(config_path)
    X_tr_eng    = fe.fit_transform(X_train_clean)   # FIT on train only
    X_v_eng     = fe.transform(X_val_clean)        # APPLY to val
    X_te_eng    = fe.transform(X_test_clean)       # APPLY to test

    fe.save_scaler(cfg["paths"]["scaler"])
    fe.save_feature_names()
    fe.save_feature_engineer()

    viz.plot_skewness(X_train_clean, X_tr_eng)

    # ── 6. Feature Selection + SMOTE — train only ────────────────
    #logger.info("\n[7/7] FEATURE SELECTION + SMOTE")
    logger.info("\n[7/7] SMOTE")
    selector = FeatureSelector(config_path)

    # Recombine for selector (it separates X and y internally)
            #train_eng       = X_tr_eng.copy()
            #train_eng[target] = y_train_clean.values

    selector = FeatureSelector(config_path)

    # Feature selection fitted on train only
           #X_tr_sel, selected_features = selector.fit_select(train_eng)

    # Apply same feature selection to val and test
           #X_v_sel  = X_v_eng[selected_features]
           #X_te_sel = X_te_eng[selected_features]

    # SMOTE on train only
    #X_train_bal, y_train_bal = selector._balance(X_tr_sel, y_train_clean)   # with feature selection
    X_train_bal, y_train_bal = selector._balance(X_tr_eng, y_train_clean)     # without feature selection

    
    viz.plot_smote_comparison(y_train_clean, pd.Series(y_train_bal))
    viz.plot_split_donut(
        n_train=len(X_train_bal), n_val=len(X_v_eng), n_test=len(X_te_eng),   # n_val=len(X_v_sel), n_test=len(X_te_sel), replace with engineered data
        fraud_train=int(y_train_bal.sum()),
        fraud_val=int(y_val.sum()),
        fraud_test=int(y_test.sum()),
    )

    mi = selector.feature_scores_.get("mutual_info", {})
    if mi:
        viz.plot_feature_importance(mi, title="Mutual Information (top 20)")       
    

    # Save processed data 
    selector.save_results(
        X_train_bal, X_v_eng, X_te_eng,   #X_v_sel, X_te_sel,
        y_train_bal, y_val, y_test,
        base_path="data/processed/",
    )


    # ── Build results dict ────────────────────────────────────────
    results = dict(
        df              = df_raw,
        train_df        = train_df,
        val_df          = val_df,
        test_df         = test_df,
        train_clean     = train_clean,
        val_clean       = val_clean,
        test_clean      = test_clean,
        X_train         = X_train_clean,   # before engineering (for skewness before)
        #X_train_sel     = X_tr_sel,        # after engineering + selection (skewness after)
        X_train_bal     = X_train_bal,
        #X_val           = X_v_sel,
        #X_test          = X_te_sel,
        y_train         = y_train,
        y_train_clean   = y_train_clean,
        y_train_bal     = y_train_bal,
        y_val           = y_val,
        y_test          = y_test,
        #selected_features = selected_features,
        overview        = overview,
        fe              = fe,
        selector        = selector,
    )

    #selector.save_results(results, "data/processed/")

    # ── Save dashboard artifacts ──────────────────────────────────
    save_all(results)


    elapsed = time.time() - t0
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"Train (balanced): {len(X_train_bal):,} ")
  

    return results

if __name__ == "__main__":
    run_pipeline()