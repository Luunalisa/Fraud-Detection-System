"""
dashboard/app.py
─────────────────────────────────────────────────────────────────────────────
Professional EDA Dashboard — Fraud Detection Pipeline
Built with Dash + Plotly + Matplotlib (figures encoded as base64 images).

Tabs
────
1  Dataset Overview EDA
2  Modeling EDA (Train Only)
3  Preprocessing & Feature Engineering Validation
4  Feature Selection

Run
───
    pip install dash dash-bootstrap-components plotly pandas numpy seaborn scikit-learn imbalanced-learn matplotlib
    python dashboard/app.py

Then open  http://127.0.0.1:8050
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import io
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")           # headless backend — must be set before pyplot import

import dash
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dash import dcc, html
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif

# Local import — adjust path as needed
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.visualization.eda_plots import EDAVisualizer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DEMO DATA (replace this section with your real dataset loading logic)
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_data():
    """
    Generates a synthetic Credit-Card-Fraud-like dataset.
    Replace with:   df = pd.read_csv("data/creditcard.csv")
    """
    np.random.seed(42)
    n = 10_000
    n_fraud = 200

    # PCA-like features V1–V28
    pca_cols = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    df = pd.DataFrame(pca_cols)
    df["Amount"] = np.random.exponential(scale=90, size=n)
    df["Time"]   = np.linspace(0, 172_800, n)           # 2 days in seconds
    df["Class"]  = 0
    fraud_idx    = np.random.choice(n, n_fraud, replace=False)
    df.loc[fraud_idx, "Class"] = 1
    # Make fraud slightly different
    df.loc[fraud_idx, "V1"]  -= 3
    df.loc[fraud_idx, "V4"]  += 4
    df.loc[fraud_idx, "Amount"] *= 0.4

    return df


def _make_splits(df):
    from sklearn.model_selection import train_test_split
    y  = df["Class"]
    X  = df.drop(columns=["Class"])
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15 / 0.85, stratify=y_trainval, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _make_smote_data(y_train):
    """Simulate SMOTE output without running actual SMOTE."""
    fraud_count = y_train.sum()
    legit_count = (y_train == 0).sum()
    y_after = pd.concat([
        pd.Series([0] * legit_count),
        pd.Series([1] * legit_count),   # fully balanced
    ], ignore_index=True)
    return y_after


def _make_log_transformed(df):
    df2 = df.copy()
    for col in df2.select_dtypes(include=np.number).columns:
        if col == "Class":
            continue
        min_val = df2[col].min()
        if min_val >= 0:
            df2[col] = np.log1p(df2[col])
    return df2


# ─────────────────────────────────────────────────────────────────────────────
# 1.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

VIZ = EDAVisualizer(save_dir=None)        # no disk saving; we use base64


def _fig_to_b64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _img(src: str, style: dict | None = None) -> html.Img:
    default = {"width": "100%", "borderRadius": "8px",
               "boxShadow": "0 2px 8px rgba(0,0,0,.08)"}
    if style:
        default.update(style)
    return html.Img(src=src, style=default)


def _card(children, title: str = "", color: str = "white") -> dbc.Card:
    header = [dbc.CardHeader(
        html.H6(title, className="mb-0 fw-bold text-primary"),
        style={"background": "#F8FAFF", "borderBottom": "1px solid #E5E7EB"},
    )] if title else []
    return dbc.Card(
        header + [dbc.CardBody(children)],
        style={"borderRadius": "10px", "border": "1px solid #E5E7EB",
               "background": color, "marginBottom": "18px"},
    )


def _section_title(text: str) -> html.Div:
    return html.Div(
        html.H5(text, className="mb-3 fw-bold",
                style={"color": "#1E3A5F", "borderBottom": "2px solid #2563EB",
                       "paddingBottom": "6px"}),
        className="mt-4 mb-2",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PRE-RENDER ALL FIGURES  (at startup — not on-demand)
# ─────────────────────────────────────────────────────────────────────────────

print("⏳  Loading data and rendering figures …")

# ── data ──────────────────────────────────────────────────────────────────────
DF     = _make_demo_data()
TARGET = "Class"
y_full = DF[TARGET]

X_train, X_val, X_test, y_train, y_val, y_test = _make_splits(DF)
df_train = pd.concat([X_train, y_train], axis=1)

y_after_smote = _make_smote_data(y_train)
df_log        = _make_log_transformed(DF.drop(columns=[TARGET]))

# ── tab 1: Dataset Overview EDA ───────────────────────────────────────────────
b64_overview  = _fig_to_b64(VIZ.plot_overview_table(DF))
b64_missing   = _fig_to_b64(VIZ.plot_missing_heatmap(DF))
b64_classdist = _fig_to_b64(VIZ.plot_class_distribution(y_full))
b64_hist_raw  = _fig_to_b64(VIZ.plot_feature_histograms(
    DF, target=TARGET, split_by_class=False))
b64_box_raw   = _fig_to_b64(VIZ.plot_boxplots(
    DF, target=TARGET, split_by_class=False))
b64_amount    = _fig_to_b64(VIZ.plot_amount_distribution(DF))
b64_hourly    = _fig_to_b64(VIZ.plot_time_analysis(DF, show_fraud_rate=False))

# ── tab 2: Modeling EDA (train only) ──────────────────────────────────────────
b64_corr_heat = _fig_to_b64(VIZ.plot_correlation_heatmap(df_train))
b64_ft_corr   = _fig_to_b64(VIZ.plot_feature_target_correlation(df_train, target=TARGET))
b64_hist_cls  = _fig_to_b64(VIZ.plot_feature_histograms(
    df_train, target=TARGET, split_by_class=True))
b64_box_cls   = _fig_to_b64(VIZ.plot_boxplots(
    df_train, target=TARGET, split_by_class=True))
b64_fraud_hr  = _fig_to_b64(VIZ.plot_time_analysis(df_train, show_fraud_rate=True))

# ── tab 3: Preprocessing & Feature Engineering ────────────────────────────────
b64_donut     = _fig_to_b64(VIZ.plot_split_donut(
    n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
    fraud_train=y_train.sum(), fraud_val=y_val.sum(), fraud_test=y_test.sum(),
))
b64_smote     = _fig_to_b64(VIZ.plot_smote_comparison(y_train, y_after_smote))

# For skewness, use top 8 high-skew numeric cols
skew_cols = (
    DF.drop(columns=[TARGET])
    .select_dtypes(include=np.number)
    .skew()
    .abs()
    .sort_values(ascending=False)
    .head(8)
    .index.tolist()
)
b64_skew  = _fig_to_b64(VIZ.plot_skewness(
    DF.drop(columns=[TARGET]),
    df_log,
    cols=skew_cols,
))
b64_logtr = _fig_to_b64(VIZ.plot_log_transform_comparison(
    DF.drop(columns=[TARGET]),
    df_log,
    cols=skew_cols[:6],
))

# ── tab 4: Feature Selection ──────────────────────────────────────────────────
feature_cols = [c for c in X_train.columns]
X_tr_arr     = X_train[feature_cols].values
y_tr_arr     = y_train.values

# Mutual Information
mi_scores = mutual_info_classif(X_tr_arr, y_tr_arr, random_state=42)
mi_dict   = dict(zip(feature_cols, mi_scores))
b64_mi    = _fig_to_b64(VIZ.plot_feature_importance(
    mi_dict, title="Mutual Information Scores (Feature Importance)", top_n=20))

# RFE (fast: use 10 estimators, 15 features for demo)
rfe_estimator = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
rfe           = RFE(rfe_estimator, n_features_to_select=15, step=3)
rfe.fit(X_tr_arr, y_tr_arr)
rfe_ranking   = dict(zip(feature_cols, rfe.ranking_.tolist()))
b64_rfe       = _fig_to_b64(VIZ.plot_rfe_ranking(
    rfe_ranking,
    n_features_selected=15,
    title="Recursive Feature Elimination — Feature Ranking",
))

print("✅  All figures rendered.")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  KPI CARDS  (top summary row)
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, icon: str = "", color: str = "#2563EB") -> dbc.Col:
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Div(icon, style={"fontSize": "1.6rem", "marginBottom": "4px"}),
                html.H4(value, className="mb-0 fw-bold", style={"color": color}),
                html.Small(label, className="text-muted"),
            ], className="text-center py-3"),
        ], style={"borderRadius": "10px", "border": "none",
                  "boxShadow": "0 2px 12px rgba(37,99,235,.12)"}),
        xs=6, sm=4, md=2,
    )


fraud_pct = y_full.mean() * 100
kpi_row = dbc.Row([
    _kpi("Total Samples",  f"{len(DF):,}",          "🗃️",  "#2563EB"),
    _kpi("Features",       str(DF.shape[1] - 1),    "📐",  "#0F6E56"),
    _kpi("Fraud Cases",    f"{y_full.sum():,}",      "🚨",  "#DC2626"),
    _kpi("Fraud Rate",     f"{fraud_pct:.3f}%",      "📊",  "#D97706"),
    _kpi("Train Rows",     f"{len(y_train):,}",      "🏋️",  "#7C3AED"),
    _kpi("RFE Selected",   "15 features",            "🎯",  "#0F6E56"),
], className="g-3 mb-4")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TAB LAYOUTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
tab1_layout = html.Div([
    _section_title("📋  Dataset Overview"),
    dbc.Row([
        dbc.Col(_card(_img(b64_overview),  "Overview Table"),       md=5),
        dbc.Col(_card(_img(b64_classdist), "Class Distribution"),   md=7),
    ], className="g-3"),

    _section_title("🔍  Missing Value Analysis"),
    _card(_img(b64_missing), "Missing Values Heatmap"),

    _section_title("📈  Feature Distributions (all data)"),
    _card(_img(b64_hist_raw), "Numeric Feature Histograms"),
    _card(_img(b64_box_raw),  "Boxplots — Outlier Detection"),

    _section_title("💰  Amount & Time"),
    dbc.Row([
        dbc.Col(_card(_img(b64_amount),  "Amount Distribution (Raw vs log1p)"), md=7),
        dbc.Col(_card(_img(b64_hourly),  "Transactions per Hour"),              md=5),
    ], className="g-3"),
])

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
tab2_layout = html.Div([
    dbc.Alert(
        [html.I(className="bi bi-info-circle me-2"),
         "All plots in this tab use the training set only to prevent data leakage."],
        color="info", className="mb-3", style={"fontSize": "0.88rem"},
    ),

    _section_title("🔗  Correlation Analysis"),
    dbc.Row([
        dbc.Col(_card(_img(b64_corr_heat), "Correlation Heatmap"),          md=7),
        dbc.Col(_card(_img(b64_ft_corr),   "Feature–Target Correlations"),  md=5),
    ], className="g-3"),

    _section_title("📊  Class-Stratified Distributions"),
    _card(_img(b64_hist_cls), "Histograms — Legit vs Fraud"),
    _card(_img(b64_box_cls),  "Boxplots — Legit vs Fraud"),

    _section_title("🕐  Temporal Fraud Patterns"),
    _card(_img(b64_fraud_hr), "Fraud Rate by Hour of Day"),
])

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
tab3_layout = html.Div([
    _section_title("✂️  Train / Validation / Test Split"),
    _card(_img(b64_donut), "Split Donut & Statistics"),

    _section_title("⚖️  SMOTE Oversampling"),
    _card(_img(b64_smote), "Class Balance Before vs After SMOTE"),

    _section_title("📐  Skewness Correction"),
    _card(_img(b64_skew), "Skewness Before & After Yeo-Johnson / log Transform"),

    _section_title("🔄  Log Transform — Feature-Level Comparison"),
    _card(_img(b64_logtr), "Distribution Before & After Transformation"),
])

# ── Tab 4 ─────────────────────────────────────────────────────────────────────
tab4_layout = html.Div([
    _section_title("🏆  Recursive Feature Elimination"),
    _card(_img(b64_rfe, {"maxHeight": "600px", "objectFit": "contain"}),
          "RFE Feature Ranking (rank 1 = selected)"),

    _section_title("🧠  Mutual Information Scores"),
    _card(_img(b64_mi, {"maxHeight": "600px", "objectFit": "contain"}),
          "Mutual Information with Target (Feature Importance)"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 5.  APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css",
    ],
    title="Fraud Detection — EDA Dashboard",
)

NAVBAR = dbc.Navbar(
    dbc.Container([
        html.Span("🔍", style={"fontSize": "1.4rem", "marginRight": "10px"}),
        dbc.NavbarBrand("Fraud Detection — EDA Dashboard",
                        className="fw-bold fs-5"),
        dbc.Badge("v1.0", color="primary", pill=True, className="ms-2"),
    ], fluid=True),
    color="white",
    dark=False,
    sticky="top",
    style={"borderBottom": "2px solid #2563EB",
           "boxShadow": "0 2px 10px rgba(0,0,0,.07)"},
)

TAB_STYLE = {
    "fontWeight": "600",
    "color":      "#6B7280",
    "padding":    "12px 20px",
    "border":     "none",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "color":           "#2563EB",
    "borderBottom":    "3px solid #2563EB",
    "backgroundColor": "#F0F7FF",
}

app.layout = html.Div([
    NAVBAR,
    dbc.Container([
        html.Br(),
        # ── KPI row ──────────────────────────────────────────────────────────
        kpi_row,
        # ── Tabs ─────────────────────────────────────────────────────────────
        dcc.Tabs(
            id="main-tabs",
            value="tab-1",
            children=[
                dcc.Tab(label="📋 Dataset Overview EDA",
                        value="tab-1",
                        style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="🔬 Modeling EDA (Train Only)",
                        value="tab-2",
                        style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="⚙️ Preprocessing & Feature Engineering",
                        value="tab-3",
                        style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="🎯 Feature Selection",
                        value="tab-4",
                        style=TAB_STYLE, selected_style=TAB_SELECTED),
            ],
            style={"marginBottom": "0"},
        ),
        html.Div(id="tab-content",
                 style={"padding": "20px 0", "minHeight": "400px"}),
    ], fluid=True, style={"maxWidth": "1400px"}),

    # ── Footer ────────────────────────────────────────────────────────────────
    html.Footer(
        dbc.Container(
            html.Small("Fraud Detection EDA Dashboard · Built with Dash & Matplotlib",
                       className="text-muted"),
            fluid=True,
        ),
        style={"borderTop": "1px solid #E5E7EB",
               "padding": "16px 0", "marginTop": "40px", "textAlign": "center"},
    ),
], style={"backgroundColor": "#F8FAFC", "minHeight": "100vh"})


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

from dash.dependencies import Input, Output

@app.callback(Output("tab-content", "children"),
              Input("main-tabs", "value"))
def render_tab(tab: str):
    if tab == "tab-1":
        return tab1_layout
    if tab == "tab-2":
        return tab2_layout
    if tab == "tab-3":
        return tab3_layout
    if tab == "tab-4":
        return tab4_layout
    return html.Div("Unknown tab")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
