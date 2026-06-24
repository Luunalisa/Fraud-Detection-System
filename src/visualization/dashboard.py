"""
src/visualization/dashboard.py
--------------------------------
Professional interactive EDA dashboard — 4 tabs.

DESIGN PRINCIPLE:
-----------------
All static charts call EDAVisualizer methods from eda_plots.py directly.
Matplotlib figures are converted to base64 PNG and embedded as html.Img.
Only the interactive dropdown-driven charts use Plotly (callbacks cannot
regenerate matplotlib figures efficiently on each user interaction).

Tab 1  Dataset Overview EDA
Tab 2  Modeling EDA (Train Only)
Tab 3  Preprocessing & Feature Engineering Validation
Tab 4  Feature Selection

Run:
    python -m src.visualization.dashboard
    # Opens at http://127.0.0.1:8050

Requirements:
    pip install dash dash-bootstrap-components plotly matplotlib seaborn
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required for Dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path fix ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.visualization.eda_plots import EDAVisualizer

# ─────────────────────────────────────────────────────────────────
# COLOUR SYSTEM
# ─────────────────────────────────────────────────────────────────
C = dict(
    bg      = "#0F172A",
    card    = "#1E293B",
    card2   = "#162032",
    border  = "#2D3F55",
    blue    = "#3B82F6",
    blue_dk = "#1D4ED8",
    green   = "#10B981",
    red     = "#EF4444",
    amber   = "#F59E0B",
    purple  = "#8B5CF6",
    cyan    = "#06B6D4",
    text    = "#F1F5F9",
    muted   = "#94A3B8",
    legit   = "#10B981",
    fraud   = "#EF4444",
    line    = "#334155",
)

# Plotly base layout — for the interactive dropdown charts only
PLOTLY_BASE = dict(
    paper_bgcolor = C["card"],
    plot_bgcolor  = C["bg"],
    font          = dict(color=C["text"], family="Inter, sans-serif", size=11),
    margin        = dict(l=50, r=30, t=55, b=45),
    hoverlabel    = dict(bgcolor=C["card2"], font_color=C["text"],
                         bordercolor=C["border"]),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"]),
    xaxis         = dict(gridcolor=C["line"], linecolor=C["border"],
                         zerolinecolor=C["border"]),
    yaxis         = dict(gridcolor=C["line"], linecolor=C["border"],
                         zerolinecolor=C["border"]),
)


def PL(fig: go.Figure, **kw) -> go.Figure:
    """Apply Plotly base layout + overrides."""
    fig.update_layout(**{**PLOTLY_BASE, **kw})
    return fig


# ─────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────
DASH_DIR = ROOT / "data" / "dashboard"


def jload(name: str) -> dict:
    p = DASH_DIR / name
    return json.load(open(p)) if p.exists() else {}


def cload(name: str) -> pd.DataFrame:
    p = DASH_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# Load all artifacts once at startup
OV    = jload("overview.json")
CD    = jload("class_dist.json")
MJ    = jload("missing.json")
SS    = jload("split_stats.json")
SM    = jload("smote_stats.json")
MI    = jload("mi_scores.json")
RFE   = jload("rfe_ranking.json")
SKEW  = jload("skewness.json")
SELF  = jload("selected_features.json")    # list of feature names

DF_RAW  = cload("df_raw_sample.csv")       # 5000-row sample of raw data
DF_TR   = cload("df_train_sample.csv")     # 5000-row sample of train_clean
AMT     = cload("amount_data.csv")         # Amount + log_amount + Class
TIME_D  = cload("time_data.csv")           # hour + Class
CORR_DF = cload("correlation_matrix.csv")  # correlation matrix (train)

NUM_RAW = [c for c in DF_RAW.select_dtypes(include=np.number).columns
           if c != "Class"] if not DF_RAW.empty else []
NUM_TR  = [c for c in DF_TR.select_dtypes(include=np.number).columns
           if c != "Class"] if not DF_TR.empty else []

# ─────────────────────────────────────────────────────────────────
# EDAVisualizer instance — all static charts call this
# save_dir=None so no PNG files are written to disk
# ─────────────────────────────────────────────────────────────────
VIZ = EDAVisualizer(save_dir=None)

# ─────────────────────────────────────────────────────────────────
# MATPLOTLIB → BASE64 BRIDGE
# Converts any plt.Figure from eda_plots into an html.Img
# ─────────────────────────────────────────────────────────────────

def mpl_to_b64(fig: plt.Figure, dpi: int = 130) -> str:
    """Render a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight",
                facecolor="#1E293B",   # match dashboard card colour
                edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def mpl_img(fig: plt.Figure, dpi: int = 130) -> html.Img:
    """Return an html.Img tag from a matplotlib figure."""
    return html.Img(
        src=mpl_to_b64(fig, dpi=dpi),
        style={"width": "100%", "borderRadius": "8px",
               "display": "block"},
    )


# ─────────────────────────────────────────────────────────────────
# PRE-BUILD ALL STATIC MATPLOTLIB FIGURES AT STARTUP
# Each calls the corresponding EDAVisualizer method directly.
# ─────────────────────────────────────────────────────────────────

print("[Dashboard] Building static figures from eda_plots.py ...")

# ── Tab 1 static figures ──────────────────────────────────────────

# 1. Overview table
IMG_OVERVIEW = mpl_img(VIZ.plot_overview_table(DF_RAW)) \
    if not DF_RAW.empty else None

# 2. Missing values heatmap
IMG_MISSING = mpl_img(VIZ.plot_missing_heatmap(DF_RAW)) \
    if not DF_RAW.empty else None

# 3. Class distribution
if not DF_RAW.empty and "Class" in DF_RAW.columns:
    IMG_CLASS_DIST = mpl_img(VIZ.plot_class_distribution(DF_RAW["Class"]))
else:
    IMG_CLASS_DIST = None

# 8. Amount distribution (raw vs log) — no class split for Tab 1
if not DF_RAW.empty and "Amount" in DF_RAW.columns:
    IMG_AMOUNT = mpl_img(VIZ.plot_amount_distribution(DF_RAW,
                                                       split_by_class=False))
else:
    IMG_AMOUNT = None

# 9. Time analysis — transactions per hour only (no fraud rate for Tab 1)
if not DF_RAW.empty and "Time" in DF_RAW.columns:
    IMG_TIME_TXN = mpl_img(VIZ.plot_time_analysis(DF_RAW,
                                                    show_fraud_rate=False))
else:
    IMG_TIME_TXN = None

# ── Tab 2 static figures (train data only) ────────────────────────

# 6. Correlation heatmap
IMG_CORR = mpl_img(VIZ.plot_correlation_heatmap(DF_TR)) \
    if not DF_TR.empty else None

# 7. Feature-target correlation
IMG_FT_CORR = mpl_img(
    VIZ.plot_feature_target_correlation(DF_TR, target="Class", top_n=20)
) if not DF_TR.empty and "Class" in DF_TR.columns else None

# 9b. Fraud rate by hour (train data — Modeling tab)
if not DF_TR.empty and "Time" in DF_TR.columns and "Class" in DF_TR.columns:
    IMG_FRAUD_HOUR = mpl_img(VIZ.plot_time_analysis(DF_TR,
                                                      show_fraud_rate=True))
elif not TIME_D.empty and "Class" in TIME_D.columns:
    IMG_FRAUD_HOUR = mpl_img(VIZ.plot_time_analysis(TIME_D,
                                                      show_fraud_rate=True))
else:
    IMG_FRAUD_HOUR = None

# ── Tab 3 static figures ──────────────────────────────────────────

# 11. Split donut
_nt  = SS.get("n_train", 0)
_nv  = SS.get("n_val",   0)
_nte = SS.get("n_test",  0)
if _nt or _nv or _nte:
    IMG_DONUT = mpl_img(VIZ.plot_split_donut(
        n_train=_nt, n_val=_nv, n_test=_nte,
        fraud_train=SS.get("fraud_train", 0),
        fraud_val=  SS.get("fraud_val",   0),
        fraud_test= SS.get("fraud_test",  0),
    ))
else:
    IMG_DONUT = None

# 10. SMOTE comparison
_sb = SM.get("before", {})
_sa = SM.get("after",  {})
if _sb and _sa:
    y_before = pd.Series(
        [0] * _sb.get("0", _sb.get(0, 0)) +
        [1] * _sb.get("1", _sb.get(1, 0))
    )
    y_after = pd.Series(
        [0] * _sa.get("0", _sa.get(0, 0)) +
        [1] * _sa.get("1", _sa.get(1, 0))
    )
    IMG_SMOTE = mpl_img(VIZ.plot_smote_comparison(y_before, y_after))
else:
    IMG_SMOTE = None

# 13. Skewness before / after
_skew_b = SKEW.get("before", {})
_skew_a = SKEW.get("after",  {})
if _skew_b and _skew_a:
    df_skew_b = pd.DataFrame([_skew_b])
    df_skew_a = pd.DataFrame([_skew_a])
    IMG_SKEWNESS = mpl_img(VIZ.plot_skewness(df_skew_b, df_skew_a))
else:
    IMG_SKEWNESS = None

# 15. Log transform comparison
if not AMT.empty and "Amount" in AMT.columns and "log_amount" in AMT.columns:
    IMG_LOG_TRANSFORM = mpl_img(VIZ.plot_log_transform_comparison(
        df_raw=AMT[["Amount"]],
        df_transformed=AMT[["log_amount"]].rename(
            columns={"log_amount": "Amount"}),
        cols=["Amount"],
    ))
else:
    IMG_LOG_TRANSFORM = None

# ── Tab 4 static figures ──────────────────────────────────────────

# 12. Feature importance (MI scores)
if MI:
    IMG_MI = mpl_img(VIZ.plot_feature_importance(
        MI,
        title="Mutual Information Scores — Top 20 (post-RFE)",
        top_n=20,
    ))
else:
    IMG_MI = None

# 14. RFE ranking
_rrank = RFE.get("ranking", {})
_n_sel = RFE.get("n_selected", len(SELF) if isinstance(SELF, list) else 0)
if _rrank:
    IMG_RFE = mpl_img(VIZ.plot_rfe_ranking(
        ranking=_rrank,
        n_features_selected=_n_sel,
    ))
else:
    IMG_RFE = None

print("[Dashboard] All static figures built.\n")

# ─────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────
CARD_STYLE = {
    "backgroundColor": C["card"],
    "border":          f"1px solid {C['border']}",
    "borderRadius":    "12px",
    "padding":         "20px",
    "marginBottom":    "18px",
    "boxShadow":       "0 2px 8px rgba(0,0,0,0.3)",
}
SEC_STYLE = {
    "color":         C["muted"],
    "fontSize":      "11px",
    "fontWeight":    "600",
    "letterSpacing": "0.08em",
    "textTransform": "uppercase",
    "marginBottom":  "14px",
    "paddingBottom": "6px",
    "borderBottom":  f"1px solid {C['border']}",
}
TAB_STYLE = {
    "backgroundColor": C["card"],
    "color":           C["muted"],
    "border":          f"1px solid {C['border']}",
    "borderRadius":    "8px 8px 0 0",
    "padding":         "10px 22px",
    "fontSize":        "13px",
    "fontWeight":      "500",
}
TAB_SEL = {
    **TAB_STYLE,
    "backgroundColor": C["blue_dk"],
    "color":           "white",
    "borderTop":       f"2px solid {C['blue']}",
}
DD_STYLE = {
    "backgroundColor": C["card2"],
    "color":           C["text"],
    "border":          f"1px solid {C['border']}",
    "borderRadius":    "7px",
    "fontSize":        "12px",
}
TABLE_BASE = dict(
    style_header={
        "backgroundColor": C["blue_dk"], "color": "white",
        "fontWeight": "bold", "fontSize": "12px",
        "border": f"1px solid {C['border']}",
    },
    style_data={
        "backgroundColor": C["card"], "color": C["text"],
        "borderColor": C["border"], "fontSize": "12px",
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": C["bg"]},
    ],
    style_table={"overflowX": "auto"},
)


def kpi(label, value, color=None, sub=""):
    return dbc.Col(html.Div([
        html.P(label, style={"color": C["muted"], "fontSize": "10px",
                              "marginBottom": "4px", "textTransform": "uppercase",
                              "letterSpacing": "0.07em"}),
        html.H4(value, style={"color": color or C["blue"],
                               "fontWeight": "800", "marginBottom": "2px",
                               "fontSize": "21px"}),
        html.P(sub, style={"color": C["muted"], "fontSize": "10px",
                            "marginBottom": "0"}),
    ], style={**CARD_STYLE, "padding": "14px 18px", "marginBottom": "0"}),
    xs=12, sm=6, md=4, lg=3)


def sec(title):
    return html.P(title, style=SEC_STYLE)


def card(*children):
    return html.Div(list(children), style=CARD_STYLE)


def G(gid, fig=None, h=360):
    """Plotly graph — used only for interactive dropdown charts."""
    kw = {"id": gid,
          "config": {"displayModeBar": True,
                     "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                     "displaylogo": False},
          "style": {"height": f"{h}px"}}
    if fig is not None:
        kw["figure"] = fig
    return dcc.Graph(**kw)


def DD(dd_id, opts, val=None):
    return dcc.Dropdown(
        id=dd_id,
        options=[{"label": o, "value": o} for o in opts],
        value=val or (opts[0] if opts else None),
        style=DD_STYLE, className="mb-3", clearable=False,
    )


def img_card(title: str, img: html.Img | None) -> html.Div:
    """Wrap a matplotlib image in a dashboard card."""
    return card(
        sec(title),
        img if img is not None else html.P(
            "Data not available — run the pipeline first.",
            style={"color": C["muted"], "fontSize": "12px",
                   "fontStyle": "italic"}
        ),
    )


# ─────────────────────────────────────────────────────────────────
# SELECTED FEATURES TABLE ROWS
# ─────────────────────────────────────────────────────────────────
SEL_ROWS = []
for _i, _f in enumerate(SELF if isinstance(SELF, list) else [], 1):
    SEL_ROWS.append({
        "Rank":     _i,
        "Feature":  _f,
        "MI Score": f"{MI.get(_f, 0):.5f}" if MI else "—",
        "Type":     ("Engineered" if any(k in _f for k in
                     ["log_", "amount_", "hour", "is_", "day_",
                      "sqrt", "bin", "zscore", "sin", "cos"])
                     else "PCA / original"),
    })

# ─────────────────────────────────────────────────────────────────
# TAB BUILDERS
# ─────────────────────────────────────────────────────────────────

def tab1():
    n_rows = OV.get("n_rows", 0)
    n_cols = OV.get("n_cols", 0)
    mem    = OV.get("memory_mb", "—")
    n_dup  = OV.get("n_duplicates", 0)
    miss   = MJ.get("total_missing", 0)
    imb    = CD.get("imbalance", "—")

    return html.Div([
        # KPI cards
        dbc.Row([
            kpi("Total Rows",    f"{n_rows:,}" if n_rows else "—", C["blue"]),
            kpi("Columns",       str(n_cols),                       C["purple"]),
            kpi("Memory",        f"{mem} MB",                       C["cyan"]),
            kpi("Duplicates",    f"{n_dup:,}",
                C["red"] if n_dup else C["green"]),
            kpi("Imbalance",     f"{imb}:1",                        C["red"],
                sub="legit : fraud"),
            kpi("Missing Cells", f"{miss:,}",
                C["green"] if miss == 0 else C["red"]),
        ], className="g-2 mb-3"),

        # 1. Overview table — from eda_plots.plot_overview_table()
        img_card("Dataset Overview", IMG_OVERVIEW),

        # 2. Missing values — from eda_plots.plot_missing_heatmap()
        img_card("Missing Values Analysis", IMG_MISSING),

        # 3. Class distribution — from eda_plots.plot_class_distribution()
        img_card("Class Distribution", IMG_CLASS_DIST),

        # 4. Histograms (no class split) — interactive Plotly dropdown
        # eda_plots.plot_feature_histograms() only works for full grid,
        # here we use Plotly for per-feature interactivity
        card(
            sec("Feature Histogram — All Data  (select a feature)"),
            DD("dd-hist-raw", NUM_RAW),
            G("g-hist-raw", h=340),
        ),

        # 5. Boxplots (no class split) — interactive Plotly dropdown
        card(
            sec("Boxplot — Outlier Detection  (select a feature)"),
            DD("dd-box-raw", NUM_RAW),
            G("g-box-raw", h=340),
        ),

        # 8. Amount distribution — from eda_plots.plot_amount_distribution()
        img_card("Amount Distribution — Raw vs log1p Transform", IMG_AMOUNT),

        # 9. Transactions per hour — from eda_plots.plot_time_analysis()
        img_card("Transactions per Hour of Day", IMG_TIME_TXN),
    ])


def tab2():
    return html.Div([
        # Leakage warning
        html.Div([
            html.Span(
                "⚠️  All charts in this tab use training data only. "
                "Val and test are never visualised to prevent data leakage.",
                style={"color": C["amber"], "fontSize": "12px",
                       "fontWeight": "500"}),
        ], style={**CARD_STYLE, "padding": "10px 18px",
                  "borderLeft": f"3px solid {C['amber']}",
                  "marginBottom": "18px"}),

        # 6. Correlation heatmap — from eda_plots.plot_correlation_heatmap()
        img_card("Feature Correlation Matrix (Train Set)", IMG_CORR),

        # 7. Feature-target correlation — from eda_plots.plot_feature_target_correlation()
        img_card("Feature–Target Correlations (Train Set)", IMG_FT_CORR),

        # 4b. Histograms by class — interactive Plotly
        card(
            sec("Feature Distribution by Class  (select a feature)"),
            DD("dd-hist-cls", NUM_TR),
            G("g-hist-cls", h=340),
        ),

        # 5b. Boxplots by class — interactive Plotly
        card(
            sec("Boxplot by Class  (select a feature)"),
            DD("dd-box-cls", NUM_TR),
            G("g-box-cls", h=340),
        ),

        # 9b. Fraud rate by hour — from eda_plots.plot_time_analysis()
        img_card("Fraud Rate by Hour of Day (Train Set)", IMG_FRAUD_HOUR),
    ])


def tab3():
    _nt  = SS.get("n_train", 0)
    _nv  = SS.get("n_val",   0)
    _nte = SS.get("n_test",  0)

    split_rows = [
        {"Set": "Train (balanced)", "Rows": f"{_nt:,}",
         "Fraud": f"{SS.get('fraud_train',0):,}",
         "Fraud rate": f"{SS.get('fraud_rate_train',0):.4%}"},
        {"Set": "Validation", "Rows": f"{_nv:,}",
         "Fraud": f"{SS.get('fraud_val',0):,}",
         "Fraud rate": f"{SS.get('fraud_rate_val',0):.4%}"},
        {"Set": "Test", "Rows": f"{_nte:,}",
         "Fraud": f"{SS.get('fraud_test',0):,}",
         "Fraud rate": f"{SS.get('fraud_rate_test',0):.4%}"},
    ]

    return html.Div([
        # 11. Split donut + stats table side by side
        dbc.Row([
            dbc.Col(
                # plot_split_donut() called above → IMG_DONUT
                img_card("Data Split — 70 / 15 / 15", IMG_DONUT),
                md=6),
            dbc.Col(card(
                sec("Split Statistics"),
                dash_table.DataTable(
                    data=split_rows,
                    columns=[{"name": c, "id": c}
                              for c in ["Set", "Rows", "Fraud", "Fraud rate"]],
                    **TABLE_BASE,
                ),
                html.Br(),
                html.P("✓  Stratified — fraud rate preserved in all sets.",
                       style={"color": C["green"], "fontSize": "11px",
                              "marginBottom": "4px"}),
                html.P("✓  SMOTE on train only — val/test stay natural.",
                       style={"color": C["green"], "fontSize": "11px",
                              "marginBottom": "0"}),
            ), md=6),
        ], className="g-3"),

        # 10. SMOTE comparison — from eda_plots.plot_smote_comparison()
        img_card("SMOTE Oversampling — Train Set Only", IMG_SMOTE),

        # 13. Skewness — from eda_plots.plot_skewness()
        img_card("Skewness Before vs After Engineering", IMG_SKEWNESS),

        # 15. Log transform — from eda_plots.plot_log_transform_comparison()
        #     + interactive Plotly selector for raw vs log
        card(
            sec("Log Transform Comparison  (select column)"),
            DD("dd-log",
               [c for c in ["Amount", "log_amount"]
                if not AMT.empty and c in AMT.columns]),
            G("g-log-tr", h=340),
        ),

        html.Div([
            html.P("ℹ️  The chart above uses Plotly for interactivity. "
                   "The full before/after comparison is shown below "
                   "using eda_plots.plot_log_transform_comparison():",
                   style={"color": C["muted"], "fontSize": "11px",
                          "marginBottom": "10px"}),
            IMG_LOG_TRANSFORM if IMG_LOG_TRANSFORM else html.P(
                "Amount data not available.",
                style={"color": C["muted"], "fontStyle": "italic",
                       "fontSize": "12px"}),
        ], style=CARD_STYLE),
    ])


def tab4():
    n_total = len(MI) if MI else 0
    n_sel   = len(SELF) if isinstance(SELF, list) else 0

    return html.Div([
        dbc.Row([
            kpi("Input Features",    str(n_total),         C["blue"],
                sub="after engineering"),
            kpi("Selected Features", str(n_sel),           C["green"],
                sub="survived all filters"),
            kpi("Removed",           str(n_total - n_sel), C["amber"],
                sub="variance + corr + RFE"),
            kpi("Method",            "RFE + RF",           C["purple"],
                sub="RandomForest estimator"),
        ], className="g-2 mb-3"),

        # 12. Feature importance (MI) — from eda_plots.plot_feature_importance()
        img_card(
            "Mutual Information Scores — Top 20 (post-RFE, "
            "confirms final selected features)",
            IMG_MI,
        ),

        # 14. RFE ranking — from eda_plots.plot_rfe_ranking()
        img_card("RFE Feature Ranking", IMG_RFE),

        # Selected features table
        card(
            sec("Final Selected Features"),
            dash_table.DataTable(
                data=SEL_ROWS,
                columns=[{"name": c, "id": c}
                          for c in ["Rank", "Feature", "MI Score", "Type"]],
                sort_action="native",
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": C["bg"]},
                    {"if": {"filter_query": '{Type} = "Engineered"'},
                     "color": C["cyan"]},
                ],
                style_header=TABLE_BASE["style_header"],
                style_data=TABLE_BASE["style_data"],
                style_table=TABLE_BASE["style_table"],
                page_size=15,
            ),
        ),
    ])


# ─────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Fraud Detection — EDA Dashboard",
)

app.layout = html.Div(
    style={"backgroundColor": C["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter', sans-serif", "color": C["text"]},
    children=[

        # Header
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3(
                            [html.Span("🔍 "), "Fraud Detection EDA Dashboard"],
                            style={"margin": "0", "color": "white",
                                   "fontWeight": "800", "fontSize": "18px"}),
                        html.P(
                            "Credit Card Transactions — Preprocessing & EDA  "
                            "| Static charts use eda_plots.py directly",
                            style={"margin": "0",
                                   "color": "rgba(255,255,255,0.6)",
                                   "fontSize": "11px"}),
                    ]),
                    dbc.Col([
                        html.Div([
                            html.Span(
                                f"{OV.get('n_rows', 284807):,} transactions",
                                style={"color": "rgba(255,255,255,0.8)",
                                       "fontSize": "11px",
                                       "marginRight": "16px"}),
                            html.Span("0.173% fraud rate",
                                      style={"color": C["red"],
                                             "fontSize": "11px",
                                             "fontWeight": "700"}),
                        ], style={"textAlign": "right", "paddingTop": "8px"}),
                    ], className="ms-auto", width="auto"),
                ], align="center"),
            ], fluid=True, style={"padding": "12px 28px"}),
        ], style={
            "background":
                f"linear-gradient(135deg, {C['blue_dk']} 0%, {C['blue']} 100%)",
            "boxShadow": "0 3px 16px rgba(0,0,0,0.5)",
        }),

        # Tabs
        dbc.Container([
            dcc.Tabs(
                id="tabs",
                value="t1",
                children=[
                    dcc.Tab(label="📊  Dataset Overview EDA",
                            value="t1", style=TAB_STYLE,
                            selected_style=TAB_SEL),
                    dcc.Tab(label="🧪  Modeling EDA",
                            value="t2", style=TAB_STYLE,
                            selected_style=TAB_SEL),
                    dcc.Tab(label="⚙️  Preprocessing & FE",
                            value="t3", style=TAB_STYLE,
                            selected_style=TAB_SEL),
                    dcc.Tab(label="🎯  Feature Selection",
                            value="t4", style=TAB_STYLE,
                            selected_style=TAB_SEL),
                ],
                style={"marginTop": "22px"},
                colors={"border": C["border"],
                        "primary": C["blue"],
                        "background": C["card"]},
            ),
            html.Div(id="tab-content",
                     style={"paddingTop": "22px", "paddingBottom": "50px"}),
        ], fluid=True, style={"padding": "0 28px"}),
    ],
)


# ─────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────

@app.callback(Output("tab-content", "children"),
              Input("tabs", "value"))
def render_tab(t):
    return {"t1": tab1, "t2": tab2,
            "t3": tab3, "t4": tab4}.get(t, lambda: html.Div())()


# ── Interactive: histogram all data (Tab 1) ───────────────────────
@app.callback(Output("g-hist-raw", "figure"),
              Input("dd-hist-raw", "value"))
def cb_hist_raw(col):
    if not col or DF_RAW.empty or col not in DF_RAW.columns:
        return go.Figure()
    vals = DF_RAW[col].dropna()
    fig  = PL(go.Figure([go.Histogram(
        x=vals, nbinsx=55,
        marker_color=C["blue"], opacity=0.85)]),
        title=f"Distribution — {col}",
        xaxis_title=col, yaxis_title="Count")
    sk = float(vals.skew())
    fig.add_annotation(
        text=(f"skew={sk:.3f}   mean={vals.mean():.3f}   "
              f"median={vals.median():.3f}   std={vals.std():.3f}"),
        xref="paper", yref="paper", x=0.5, y=1.07, showarrow=False,
        font=dict(color=C["red"] if abs(sk) > 1 else C["green"], size=11))
    return fig


# ── Interactive: boxplot all data (Tab 1) ────────────────────────
@app.callback(Output("g-box-raw", "figure"),
              Input("dd-box-raw", "value"))
def cb_box_raw(col):
    if not col or DF_RAW.empty or col not in DF_RAW.columns:
        return go.Figure()
    vals   = DF_RAW[col].dropna()
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr    = q3 - q1
    n_out  = int(((vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)).sum())
    fig    = PL(go.Figure([go.Box(
        y=vals, name=col,
        marker_color=C["blue"], line_color=C["blue"],
        fillcolor="rgba(59,130,246,0.2)", boxmean=True)]),
        title=f"Boxplot — {col}", yaxis_title=col)
    fig.add_annotation(
        text=(f"Q1={q1:.3f}   Q3={q3:.3f}   IQR={iqr:.3f}   "
              f"Outliers={n_out:,} ({n_out/len(vals):.2%})"),
        xref="paper", yref="paper", x=0.5, y=1.07, showarrow=False,
        font=dict(color=C["muted"], size=11))
    return fig


# ── Interactive: histogram by class (Tab 2) ──────────────────────
@app.callback(Output("g-hist-cls", "figure"),
              Input("dd-hist-cls", "value"))
def cb_hist_cls(col):
    if not col or DF_TR.empty or col not in DF_TR.columns:
        return go.Figure()
    fig = go.Figure()
    if "Class" in DF_TR.columns:
        for cls, color, lbl in [(0, C["legit"],  "Legit"),
                                  (1, C["fraud"], "Fraud")]:
            vals = DF_TR[DF_TR["Class"] == cls][col].dropna()
            fig.add_trace(go.Histogram(
                x=vals, nbinsx=55, name=lbl,
                marker_color=color, opacity=0.65,
                histnorm="probability density"))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(x=DF_TR[col].dropna(),
                                    nbinsx=55, marker_color=C["blue"],
                                    opacity=0.85))
    PL(fig, title=f"{col} — Distribution by Class (density normalised)",
       xaxis_title=col, yaxis_title="Density")
    return fig


# ── Interactive: boxplot by class (Tab 2) ────────────────────────
@app.callback(Output("g-box-cls", "figure"),
              Input("dd-box-cls", "value"))
def cb_box_cls(col):
    if not col or DF_TR.empty or col not in DF_TR.columns:
        return go.Figure()
    fig = go.Figure()
    if "Class" in DF_TR.columns:
        for cls, color, lbl in [(0, C["legit"],  "Legit"),
                                  (1, C["fraud"], "Fraud")]:
            vals = DF_TR[DF_TR["Class"] == cls][col].dropna()
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_trace(go.Box(
                y=vals, name=lbl,
                marker_color=color, line_color=color, boxmean=True,
                fillcolor=f"rgba({r},{g},{b},0.2)"))
    else:
        fig.add_trace(go.Box(y=DF_TR[col].dropna(), name=col,
                              marker_color=C["blue"]))
    PL(fig, title=f"{col} — Boxplot by Class", yaxis_title=col)
    return fig


# ── Interactive: log transform (Tab 3) ───────────────────────────
@app.callback(Output("g-log-tr", "figure"),
              Input("dd-log", "value"))
def cb_log_tr(col):
    if not col or AMT.empty or col not in AMT.columns:
        return go.Figure()
    vals  = AMT[col].dropna()
    color = C["red"] if col == "Amount" else C["green"]
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig   = make_subplots(rows=1, cols=2,
                           subplot_titles=["Histogram", "Box Plot"])
    fig.add_trace(go.Histogram(x=vals, nbinsx=60,
                                marker_color=color, opacity=0.82),
                  row=1, col=1)
    fig.add_trace(go.Box(y=vals, marker_color=color, boxmean=True,
                          fillcolor=f"rgba({r},{g},{b},0.2)",
                          line_color=color),
                  row=1, col=2)
    sk = float(vals.skew())
    fig.add_annotation(
        text=f"skew = {sk:.3f}   mean = {vals.mean():.4f}",
        xref="paper", yref="paper", x=0.5, y=1.07, showarrow=False,
        font=dict(color=C["red"] if abs(sk) > 1 else C["green"], size=12))
    PL(fig, title=f"Transform Comparison — {col}", showlegend=False)
    fig.update_xaxes(gridcolor=C["line"])
    fig.update_yaxes(gridcolor=C["line"])
    return fig


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 58)
    print("  Fraud Detection — EDA Dashboard")
    print("  Open:  http://127.0.0.1:8050")
    print("  Stop:  Ctrl+C")
    print("═" * 58 + "\n")
    app.run(debug=True, port=8050, host="0.0.0.0")
