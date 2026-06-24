
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path fix so imports work from any working directory ───────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

## ------------------ COLOUR SYSTEM -----------------

C = dict(
    bg       = "#0F172A",
    card     = "#1E293B",
    card2    = "#162032",
    border   = "#2D3F55",
    blue     = "#3B82F6",
    blue_dk  = "#1D4ED8",
    green    = "#10B981",
    red      = "#EF4444",
    amber    = "#F59E0B",
    purple   = "#8B5CF6",
    cyan     = "#06B6D4",
    text     = "#F1F5F9",
    muted    = "#94A3B8",
    legit    = "#10B981",
    fraud    = "#EF4444",
    line     = "#334155",
)

PLOTLY_BASE = dict(
    paper_bgcolor = C["card"],
    plot_bgcolor  = C["bg"],
    font          = dict(color=C["text"], family="Inter, sans-serif", size=11),
    margin        = dict(l=50, r=30, t=50, b=40),
    hoverlabel    = dict(bgcolor=C["card2"], font_color=C["text"],
                         bordercolor=C["border"]),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                         font_color=C["text"]),
    xaxis         = dict(gridcolor=C["line"], linecolor=C["border"],
                         zerolinecolor=C["border"]),
    yaxis         = dict(gridcolor=C["line"], linecolor=C["border"],
                         zerolinecolor=C["border"]),
)


def L(fig: go.Figure, **kw) -> go.Figure:
    """Apply base layout + any overrides."""
    fig.update_layout(**{**PLOTLY_BASE, **kw})
    return fig


## -------------------- DATA LOADERS -------------------

DASH_DIR = ROOT / "data" / "dashboard"


def jload(name: str) -> dict:
    p = DASH_DIR / name
    return json.load(open(p)) if p.exists() else {}


def cload(name: str) -> pd.DataFrame:
    p = DASH_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ── Load once at startup ──────────────────────────────────────────
OV    = jload("overview.json")
CD    = jload("class_dist.json")
MJ    = jload("missing.json")
SS    = jload("split_stats.json")
SM    = jload("smote_stats.json")
MI    = jload("mi_scores.json")
RFE   = jload("rfe_ranking.json")
SKEW  = jload("skewness.json")
SELF  = jload("selected_features.json")   # list

DF_RAW   = cload("df_raw_sample.csv")
DF_TR    = cload("df_train_sample.csv")
AMT      = cload("amount_data.csv")
TIME_D   = cload("time_data.csv")
CORR_CSV = cload("correlation_matrix.csv")

NUM_RAW = [c for c in DF_RAW.select_dtypes(include=np.number).columns
           if c != "Class"] if not DF_RAW.empty else []
NUM_TR  = [c for c in DF_TR.select_dtypes(include=np.number).columns
           if c != "Class"] if not DF_TR.empty else []


## ------------------------- UI COMPONENTS ------------------------------


CARD = {
    "backgroundColor": C["card"],
    "border":          f"1px solid {C['border']}",
    "borderRadius":    "12px",
    "padding":         "20px",
    "marginBottom":    "18px",
    "boxShadow":       "0 2px 8px rgba(0,0,0,0.3)",
}

SEC = {
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

TABLE_STYLE = dict(
    style_header={
        "backgroundColor": C["blue_dk"],
        "color":           "white",
        "fontWeight":      "bold",
        "fontSize":        "12px",
        "border":          f"1px solid {C['border']}",
    },
    style_data={
        "backgroundColor": C["card"],
        "color":           C["text"],
        "borderColor":     C["border"],
        "fontSize":        "12px",
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"},
         "backgroundColor": C["bg"]},
    ],
    style_table={"overflowX": "auto"},
)


def kpi(label: str, value: str, color: str = C["blue"], sub: str = "") -> dbc.Col:
    return dbc.Col(html.Div([
        html.P(label, style={"color": C["muted"], "fontSize": "10px",
                              "marginBottom": "4px", "textTransform": "uppercase",
                              "letterSpacing": "0.07em"}),
        html.H4(value, style={"color": color, "fontWeight": "800",
                               "marginBottom": "2px", "fontSize": "22px"}),
        html.P(sub,   style={"color": C["muted"], "fontSize": "10px",
                              "marginBottom": "0"}),
    ], style={**CARD, "padding": "14px 18px", "marginBottom": "0"}),
    xs=12, sm=6, md=4, lg=2)


def section(title: str) -> html.P:
    return html.P(title, style=SEC)


def card(*children) -> html.Div:
    return html.Div(list(children), style=CARD)


def graph(fig_id: str, fig: go.Figure = None,
          height: int = 360) -> dcc.Graph:
    kwargs = {"id": fig_id,
              "config": {"displayModeBar": True,
                         "modeBarButtonsToRemove": ["select2d","lasso2d"],
                         "displaylogo": False},
              "style":  {"height": f"{height}px"}}
    if fig is not None:
        kwargs["figure"] = fig
    return dcc.Graph(**kwargs)


def dropdown(dd_id: str, options: list, value=None) -> dcc.Dropdown:
    return dcc.Dropdown(
        id=dd_id,
        options=[{"label": o, "value": o} for o in options],
        value=value or (options[0] if options else None),
        style=DD_STYLE,
        className="mb-3",
        clearable=False,
    )


## ----------------------- STATIC FIGURES (built once at startup) --------------------------


# Class distribution 
counts = CD.get("counts", {"0": 0, "1": 0})
pcts   = CD.get("pct",    {"0": 0, "1": 0})
_c0    = counts.get("0", counts.get(0, 0))
_c1    = counts.get("1", counts.get(1, 0))
_p0    = pcts.get("0",   pcts.get(0,   0))
_p1    = pcts.get("1",   pcts.get(1,   0))

FIG_CLASS_BAR = L(go.Figure([go.Bar(
    x=["Legit (0)", "Fraud (1)"],
    y=[_c0, _c1],
    marker_color=[C["legit"], C["fraud"]],
    text=[f"{_c0:,}<br>{_p0:.3f}%", f"{_c1:,}<br>{_p1:.3f}%"],
    textposition="outside",
    textfont=dict(size=11, color=C["text"]),
)]),
    title="Class Counts", yaxis_title="Count",
    margin=dict(l=50, r=30, t=50, b=40))

FIG_CLASS_PIE = L(go.Figure([go.Pie(
    labels=["Legit", "Fraud"],
    values=[_c0, _c1],
    marker_colors=[C["legit"], C["fraud"]],
    hole=0.52,
    textinfo="label+percent",
    textfont=dict(size=11),
    pull=[0, 0.1],
)]),
    title="Class Proportion",
    margin=dict(l=10, r=10, t=50, b=10))

#  Missing values 
_miss = MJ.get("missing_pct", {})
if _miss:
    _ms = pd.Series(_miss).sort_values(ascending=False).head(20)
    FIG_MISSING = L(go.Figure([go.Bar(
        x=_ms.values, y=_ms.index, orientation="h",
        marker_color=[C["red"] if v > 30 else
                      C["amber"] if v > 5 else C["green"]
                      for v in _ms.values],
    )]),
        title="Missing % per Column", xaxis_title="Missing %")
else:
    FIG_MISSING = go.Figure()
    FIG_MISSING.add_annotation(text="✓  No missing values",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(size=20, color=C["green"]))
    L(FIG_MISSING, title="Missing Values")

# Amount distribution
if not AMT.empty:
    FIG_AMOUNT = make_subplots(rows=1, cols=2,
                                subplot_titles=["Raw Amount", "log1p(Amount)"])
    FIG_AMOUNT.add_trace(
        go.Histogram(x=AMT["Amount"], nbinsx=60,
                     marker_color=C["blue"], opacity=0.85,
                     name="Raw Amount"), row=1, col=1)
    if "log_amount" in AMT.columns:
        FIG_AMOUNT.add_trace(
            go.Histogram(x=AMT["log_amount"], nbinsx=60,
                         marker_color=C["green"], opacity=0.85,
                         name="log1p"), row=1, col=2)
    L(FIG_AMOUNT, title="Amount — Raw vs log1p Transform",
      showlegend=False)
    FIG_AMOUNT.update_xaxes(gridcolor=C["line"])
    FIG_AMOUNT.update_yaxes(gridcolor=C["line"])

    # Skew annotations
    sk_raw = round(float(AMT["Amount"].skew()), 2)
    sk_log = round(float(AMT["log_amount"].skew()), 2) if "log_amount" in AMT.columns else 0
    FIG_AMOUNT.add_annotation(text=f"skew={sk_raw}", x=0.22, y=1.05,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(color=C["red"], size=11))
    FIG_AMOUNT.add_annotation(text=f"skew={sk_log}", x=0.78, y=1.05,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(color=C["green"], size=11))
else:
    FIG_AMOUNT = go.Figure()
    L(FIG_AMOUNT, title="Amount data not found")

#  Transactions per hour 
if not TIME_D.empty:
    _hrly  = TIME_D.groupby("hour").size().reset_index(name="count")
    FIG_TXN_HOUR = L(go.Figure([go.Bar(
        x=_hrly["hour"], y=_hrly["count"],
        marker_color=C["blue"], opacity=0.85,
    )]),
        title="Transactions per Hour",
        xaxis_title="Hour of Day", yaxis_title="Count")
    FIG_TXN_HOUR.add_vrect(x0=22, x1=24, fillcolor=C["red"],
                            opacity=0.09, line_width=0)
    FIG_TXN_HOUR.add_vrect(x0=0, x1=6, fillcolor=C["red"],
                            opacity=0.09, line_width=0,
                            annotation_text="Night (22–06)",
                            annotation_font_color=C["red"],
                            annotation_position="top left")
else:
    FIG_TXN_HOUR = go.Figure()
    L(FIG_TXN_HOUR, title="Time data not found")

#  Correlation heatmap 
if not CORR_CSV.empty:
    _cm = CORR_CSV.copy()
    if _cm.columns[0].startswith("Unnamed"):
        _cm = _cm.set_index(_cm.columns[0])
    FIG_CORR = L(go.Figure([go.Heatmap(
        z=_cm.values,
        x=_cm.columns.tolist(),
        y=_cm.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="ρ", tickfont=dict(color=C["text"])),
        text=np.round(_cm.values, 2),
        texttemplate="%{text}", textfont=dict(size=7),
        hoverongaps=False,
    )]),
        title="Feature Correlation Matrix (Train Set)",
        height=620,
        xaxis=dict(tickangle=45, tickfont=dict(size=8), gridcolor=C["line"]),
        yaxis=dict(tickfont=dict(size=8), gridcolor=C["line"]))
else:
    FIG_CORR = go.Figure()
    L(FIG_CORR, title="Correlation matrix not found")

#  Feature-target correlation 
if not DF_TR.empty and "Class" in DF_TR.columns:
    _nc  = [c for c in DF_TR.select_dtypes(include=np.number).columns
            if c != "Class"]
    _ct  = DF_TR[_nc].corrwith(DF_TR["Class"]).dropna()
    _ct  = _ct.reindex(_ct.abs().sort_values(ascending=False).index[:20])
    FIG_FT_CORR = L(go.Figure([go.Bar(
        x=_ct.values[::-1], y=_ct.index[::-1], orientation="h",
        marker_color=[C["fraud"] if v < 0 else C["blue"]
                      for v in _ct.values[::-1]],
        text=[f"{v:.3f}" for v in _ct.values[::-1]],
        textposition="outside", textfont=dict(color=C["text"], size=9),
    )]),
        title="Feature–Target Correlations (Train Set)",
        xaxis_title="Pearson ρ with Class", height=500)
    FIG_FT_CORR.add_vline(x=0, line_color=C["muted"], line_width=1)
else:
    FIG_FT_CORR = go.Figure()
    L(FIG_FT_CORR, title="Train data not found")

#  Fraud rate by hour
if not TIME_D.empty and "Class" in TIME_D.columns:
    _fh = (TIME_D.groupby("hour")["Class"]
           .mean().reset_index()
           .rename(columns={"Class": "fr"}))
    _fh["pct"] = _fh["fr"] * 100
    FIG_FRAUD_HOUR = L(go.Figure([
        go.Scatter(x=_fh["hour"], y=_fh["pct"],
                   mode="lines+markers",
                   line=dict(color=C["fraud"], width=2.5),
                   marker=dict(size=6, color=C["fraud"]),
                   fill="tozeroy",
                   fillcolor="rgba(239,68,68,0.12)")
    ]),
        title="Fraud Rate by Hour (Train Set)",
        xaxis_title="Hour", yaxis_title="Fraud Rate (%)")
    FIG_FRAUD_HOUR.add_vrect(x0=22, x1=24, fillcolor=C["red"],
                              opacity=0.09, line_width=0)
    FIG_FRAUD_HOUR.add_vrect(x0=0, x1=6, fillcolor=C["red"],
                              opacity=0.09, line_width=0,
                              annotation_text="Night window",
                              annotation_font_color=C["red"],
                              annotation_position="top left")
else:
    FIG_FRAUD_HOUR = go.Figure()
    L(FIG_FRAUD_HOUR, title="Time data not found")

#  Split donut 
_nt  = SS.get("n_train", 0)
_nv  = SS.get("n_val",   0)
_nte = SS.get("n_test",  0)
_tot = _nt + _nv + _nte
FIG_DONUT = L(go.Figure([go.Pie(
    labels=[f"Train  {_nt:,}", f"Val  {_nv:,}", f"Test  {_nte:,}"],
    values=[_nt, _nv, _nte],
    marker_colors=[C["green"], C["blue"], C["amber"]],
    hole=0.56,
    textinfo="label+percent",
    textfont=dict(size=11),
    pull=[0.04, 0, 0],
)]),
    title="70 / 15 / 15 Split", showlegend=True)
FIG_DONUT.add_annotation(
    text=f"{_tot:,}<br>total", x=0.5, y=0.5, showarrow=False,
    font=dict(size=13, color=C["text"]))

# SMOTE comparison 
_sb = SM.get("before", {"0": 0, "1": 0})
_sa = SM.get("after",  {"0": 0, "1": 0})
_bt = SM.get("before_total", 1)
_at = SM.get("after_total",  1)

FIG_SMOTE = make_subplots(rows=1, cols=2,
                           subplot_titles=["Before SMOTE", "After SMOTE"])
for _col, (_d, _tot_s) in enumerate([(_sb, _bt), (_sa, _at)], start=1):
    _keys = sorted(_d.keys())
    _vals = [_d[k] for k in _keys]
    _lbls = ["Legit" if str(k) == "0" else "Fraud" for k in _keys]
    _cols = [C["legit"] if str(k) == "0" else C["fraud"] for k in _keys]
    FIG_SMOTE.add_trace(
        go.Bar(x=_lbls, y=_vals, marker_color=_cols,
               text=[f"{v:,}<br>({v/_tot_s:.1%})" for v in _vals],
               textposition="outside",
               textfont=dict(color=C["text"], size=10)),
        row=1, col=_col)
L(FIG_SMOTE, title="SMOTE Oversampling — Train Set Only", showlegend=False)
FIG_SMOTE.update_xaxes(gridcolor=C["line"])
FIG_SMOTE.update_yaxes(gridcolor=C["line"])

#  Skewness before / after 
_sb2 = SKEW.get("before", {})
_sa2 = SKEW.get("after",  {})
_com = sorted(set(_sb2) & set(_sa2),
              key=lambda c: _sb2.get(c, 0), reverse=True)[:15]

if _com:
    FIG_SKEW = L(go.Figure([
        go.Bar(name="Before", x=_com,
               y=[_sb2.get(c, 0) for c in _com],
               marker_color=C["red"], opacity=0.82),
        go.Bar(name="After",  x=_com,
               y=[_sa2.get(c, 0) for c in _com],
               marker_color=C["green"], opacity=0.82),
    ]),
        title="Skewness Before vs After Engineering",
        barmode="group",
        xaxis_tickangle=-40,
        yaxis_title="|Skewness|")
    FIG_SKEW.add_hline(y=1, line_dash="dash", line_color=C["muted"],
                        annotation_text="|skew|=1",
                        annotation_font_color=C["muted"])
else:
    FIG_SKEW = go.Figure()
    L(FIG_SKEW, title="Skewness data not found")

# MI scores 
if MI:
    _mi  = pd.Series(MI).sort_values(ascending=False).head(20)
    _sel = set(SELF) if isinstance(SELF, list) else set()
    _mc  = [C["green"] if f in _sel else C["muted"]
            for f in _mi.index]
    FIG_MI = L(go.Figure([go.Bar(
        x=_mi.values[::-1], y=_mi.index[::-1],
        orientation="h",
        marker=dict(color=_mc[::-1],
                    line=dict(color=C["border"], width=0.5)),
        text=[f"{v:.4f}" for v in _mi.values[::-1]],
        textposition="outside",
        textfont=dict(color=C["text"], size=9),
    )]),
        title="Mutual Information Scores — Top 20 (green = selected by RFE)",
        xaxis_title="MI Score", height=520)
else:
    FIG_MI = go.Figure()
    L(FIG_MI, title="MI scores not found")

#  RFE ranking 
_rrank = RFE.get("ranking",    {})
_n_sel = RFE.get("n_selected", len(SELF) if isinstance(SELF, list) else 0)

if _rrank:
    _rs  = pd.Series(_rrank).sort_values()
    _rc  = [C["green"] if v == 1 else
            C["amber"] if v <= 3 else C["muted"]
            for v in _rs.values]
    FIG_RFE = L(go.Figure([go.Bar(
        x=_rs.values[::-1], y=_rs.index[::-1],
        orientation="h",
        marker=dict(color=_rc[::-1],
                    line=dict(color=C["border"], width=0.5)),
        text=[f"rank {v}" for v in _rs.values[::-1]],
        textposition="outside",
        textfont=dict(color=C["text"], size=9),
    )]),
        title="RFE Feature Ranking  (rank 1 = selected)",
        xaxis_title="RFE Rank",
        height=max(420, len(_rs) * 24))
    FIG_RFE.add_vline(x=1.5, line_dash="dash", line_color=C["red"],
                      annotation_text=f"Selected ({_n_sel})",
                      annotation_font_color=C["red"])
else:
    FIG_RFE = go.Figure()
    L(FIG_RFE, title="RFE ranking not found")


## -------------------- SPLIT STATS TABLE DATA --------------------------


SPLIT_ROWS = [
    {"Set": "Train (balanced)",
     "Rows":       f"{SS.get('n_train',0):,}",
     "Fraud":      f"{SS.get('fraud_train',0):,}",
     "Fraud rate": f"{SS.get('fraud_rate_train',0):.4%}"},
    {"Set": "Validation",
     "Rows":       f"{SS.get('n_val',0):,}",
     "Fraud":      f"{SS.get('fraud_val',0):,}",
     "Fraud rate": f"{SS.get('fraud_rate_val',0):.4%}"},
    {"Set": "Test",
     "Rows":       f"{SS.get('n_test',0):,}",
     "Fraud":      f"{SS.get('fraud_test',0):,}",
     "Fraud rate": f"{SS.get('fraud_rate_test',0):.4%}"},
]


## ---------------------- SELECTED FEATURES TABLE DATA -----------------------

SEL_ROWS = []
for i, feat in enumerate(SELF if isinstance(SELF, list) else [], 1):
    SEL_ROWS.append({
        "Rank":     i,
        "Feature":  feat,
        "MI Score": f"{MI.get(feat, 0):.5f}" if MI else "—",
        "Type":     ("Engineered" if any(k in feat for k in
                     ["log_", "amount_", "hour", "is_", "day_",
                      "sqrt", "bin", "zscore", "sin", "cos"])
                     else "PCA / original"),
    })


## ------------------ APP INIT ----------------------------


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Fraud Detection — EDA Dashboard",
)


## -------------------- TAB CONTENT BUILDERS --------------------


def tab1() -> html.Div:
    n_rows = OV.get("n_rows", 0)
    n_cols = OV.get("n_cols", 0)
    mem    = OV.get("memory_mb", "—")
    n_dup  = OV.get("n_duplicates", 0)
    miss   = MJ.get("total_missing", 0)
    imb    = CD.get("imbalance", "—")

    ov_rows = [
        {"Metric": "Shape",            "Value": f"{n_rows:,} rows × {n_cols} columns"},
        {"Metric": "Memory",           "Value": f"{mem} MB"},
        {"Metric": "Numeric cols",     "Value": str(OV.get("n_numeric", "—"))},
        {"Metric": "Categorical cols", "Value": str(OV.get("n_categorical", 0))},
        {"Metric": "Missing cells",    "Value": f"{miss:,}"},
        {"Metric": "Duplicate rows",   "Value": f"{n_dup:,}"},
        {"Metric": "Problem type",     "Value": OV.get("problem_type", "Binary Classification")},
        {"Metric": "Target column",    "Value": OV.get("target", "Class")},
    ]

    return html.Div([
        # KPI row
        dbc.Row([
            kpi("Total Rows",      f"{n_rows:,}" if n_rows else "—", C["blue"]),
            kpi("Columns",         str(n_cols),                       C["purple"]),
            kpi("Memory",          f"{mem} MB",                       C["cyan"]),
            kpi("Duplicates",      f"{n_dup:,}",
                C["red"] if n_dup else C["green"]),
            kpi("Imbalance",       f"{imb}:1",                        C["red"],
                sub="legit : fraud"),
            kpi("Missing Cells",   f"{miss:,}",
                C["green"] if miss == 0 else C["red"]),
        ], className="g-2 mb-3"),

        # Overview table + class distribution side by side
        dbc.Row([
            dbc.Col(card(
                section("Dataset Overview"),
                dash_table.DataTable(
                    data=ov_rows,
                    columns=[{"name": c, "id": c} for c in ["Metric", "Value"]],
                    **TABLE_STYLE,
                ),
            ), md=5),
            dbc.Col(card(
                section("Class Distribution"),
                dbc.Row([
                    dbc.Col(graph("cbar", FIG_CLASS_BAR, height=300), md=6),
                    dbc.Col(graph("cpie", FIG_CLASS_PIE, height=300), md=6),
                ]),
            ), md=7),
        ], className="g-3"),

        # Missing values
        card(section("Missing Values Analysis"),
             graph("gmiss", FIG_MISSING, height=280)),

        # Interactive histogram (no class split)
        card(section("Feature Histogram — All Data"),
             dropdown("dd-hist-raw", NUM_RAW),
             graph("g-hist-raw", height=320)),

        # Interactive boxplot (no class split)
        card(section("Boxplot — Outlier Detection"),
             dropdown("dd-box-raw", NUM_RAW),
             graph("g-box-raw", height=320)),

        # Amount distribution
        card(section("Amount Distribution — Raw vs log1p"),
             graph("gamt", FIG_AMOUNT, height=320)),

        # Transactions per hour
        card(section("Transactions per Hour of Day"),
             graph("gtxn", FIG_TXN_HOUR, height=300)),
    ])


def tab2() -> html.Div:
    return html.Div([
        # Warning banner
        html.Div([
            html.Span("⚠️  All charts in this tab use training data only. "
                      "Val and test are never visualised to prevent data leakage.",
                      style={"color": C["amber"], "fontSize": "12px",
                             "fontWeight": "500"}),
        ], style={**CARD, "padding": "10px 18px",
                  "borderLeft": f"3px solid {C['amber']}",
                  "marginBottom": "18px"}),

        # Correlation heatmap
        card(section("Feature Correlation Matrix"),
             graph("gcorr", FIG_CORR, height=620)),

        # Feature-target correlation
        card(section("Feature–Target Correlations"),
             graph("gftcorr", FIG_FT_CORR, height=480)),

        # Histogram by class
        card(section("Feature Distribution by Class"),
             dropdown("dd-hist-cls", NUM_TR),
             graph("g-hist-cls", height=340)),

        # Boxplot by class
        card(section("Boxplot by Class"),
             dropdown("dd-box-cls", NUM_TR),
             graph("g-box-cls", height=340)),

        # Fraud rate by hour
        card(section("Fraud Rate by Hour"),
             graph("gfraud-hr", FIG_FRAUD_HOUR, height=320)),
    ])


def tab3() -> html.Div:
    return html.Div([
        # Split donut + stats table
        dbc.Row([
            dbc.Col(card(
                section("Data Split — 70 / 15 / 15"),
                graph("gdonut", FIG_DONUT, height=340),
            ), md=5),
            dbc.Col(card(
                section("Split Statistics"),
                dash_table.DataTable(
                    data=SPLIT_ROWS,
                    columns=[{"name": c, "id": c}
                              for c in ["Set", "Rows", "Fraud", "Fraud rate"]],
                    **TABLE_STYLE,
                ),
                html.Br(),
                html.P("✓  Stratified split — fraud rate preserved in all sets.",
                       style={"color": C["green"], "fontSize": "11px",
                              "marginBottom": "4px"}),
                html.P("✓  SMOTE applied to train only — val/test remain natural.",
                       style={"color": C["green"], "fontSize": "11px",
                              "marginBottom": "0"}),
            ), md=7),
        ], className="g-3"),

        # SMOTE comparison
        card(section("SMOTE Oversampling — Train Set Only"),
             graph("gsmote", FIG_SMOTE, height=360)),

        # Skewness before / after
        card(section("Skewness Before vs After Engineering"),
             graph("gskew", FIG_SKEW, height=380)),

        # Log transform comparison (interactive)
        card(section("Log Transform Comparison"),
             dropdown("dd-log",
                      [c for c in ["Amount", "log_amount"]
                       if not AMT.empty and c in AMT.columns]),
             graph("g-log-tr", height=340)),
    ])


def tab4() -> html.Div:
    n_total = len(MI) if MI else 0
    n_sel   = len(SELF) if isinstance(SELF, list) else 0

    return html.Div([
        # KPI row
        dbc.Row([
            kpi("Input Features",    str(n_total),          C["blue"],
                sub="after engineering"),
            kpi("Selected Features", str(n_sel),            C["green"],
                sub="survived all filters"),
            kpi("Removed",           str(n_total - n_sel),  C["amber"],
                sub="variance + corr + RFE"),
            kpi("Method",            "RFE + RF",            C["purple"],
                sub="RandomForest estimator"),
        ], className="g-2 mb-3"),

        # MI scores
        card(section("Mutual Information — Feature Importance (post-RFE)"),
             graph("gmi", FIG_MI,
                   height=max(380, n_total * 22))),

        # RFE ranking
        card(section("RFE Feature Ranking"),
             graph("grfe", FIG_RFE,
                   height=max(420, len(_rrank) * 24))),

        # Selected features table
        card(section("Final Selected Features"),
             dash_table.DataTable(
                 data=SEL_ROWS,
                 columns=[{"name": c, "id": c}
                           for c in ["Rank", "Feature", "MI Score", "Type"]],
                 sort_action="native",
                 style_data_conditional=[
                     {"if": {"row_index": "odd"},
                      "backgroundColor": C["bg"]},
                     {"if": {"filter_query": '{Type} = "Engineered"'},
                      "color": C["cyan"]},
                 ],
                 **{k: v for k, v in TABLE_STYLE.items()
                    if k != "style_data_conditional"},
                 page_size=15,
             )),
    ])


## --------------------------- APP LAYOUT --------------------------

app.layout = html.Div(
    style={"backgroundColor": C["bg"],
           "minHeight": "100vh",
           "fontFamily": "'Inter', sans-serif",
           "color": C["text"]},
    children=[

        # ── Header ───────────────────────────────────────────────
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3(
                            [html.Span("🔍 ", style={"marginRight": "6px"}),
                             "Fraud Detection EDA"],
                            style={"margin": "0", "color": "white",
                                   "fontWeight": "800", "fontSize": "19px"}),
                        html.P("Credit Card Transactions — Preprocessing Dashboard",
                               style={"margin": "0",
                                      "color": "rgba(255,255,255,0.65)",
                                      "fontSize": "11px"}),
                    ]),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(html.Div([
                                html.Span(f"{OV.get('n_rows',284807):,} rows",
                                          style={"color": "rgba(255,255,255,0.8)",
                                                 "fontSize": "11px",
                                                 "marginRight": "16px"}),
                                html.Span("0.173% fraud",
                                          style={"color": C["red"],
                                                 "fontSize": "11px",
                                                 "fontWeight": "700"}),
                            ], style={"textAlign": "right",
                                      "paddingTop": "10px"})),
                        ]),
                    ], className="ms-auto", width="auto"),
                ], align="center"),
            ], fluid=True, style={"padding": "13px 28px"}),
        ], style={"background": f"linear-gradient(135deg, {C['blue_dk']}, {C['blue']})",
                  "boxShadow": "0 3px 16px rgba(0,0,0,0.5)",
                  "marginBottom": "0"}),

        # ── Tab bar ───────────────────────────────────────────────
        dbc.Container([
            dcc.Tabs(
                id="tabs",
                value="t1",
                children=[
                    dcc.Tab(label="📊  Dataset Overview EDA",
                            value="t1", style=TAB_STYLE, selected_style=TAB_SEL),
                    dcc.Tab(label="🧪  Modeling EDA",
                            value="t2", style=TAB_STYLE, selected_style=TAB_SEL),
                    dcc.Tab(label="⚙️  Preprocessing & FE",
                            value="t3", style=TAB_STYLE, selected_style=TAB_SEL),
                    dcc.Tab(label="🎯  Feature Selection",
                            value="t4", style=TAB_STYLE, selected_style=TAB_SEL),
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


## ----------------------- CALLBACKS ------------------------


@app.callback(Output("tab-content", "children"),
              Input("tabs", "value"))
def render_tab(t: str):
    if t == "t1": return tab1()
    if t == "t2": return tab2()
    if t == "t3": return tab3()
    if t == "t4": return tab4()
    return html.Div("Not found")


# ── Tab 1 — histogram (all data) ──────────────────────────────────
@app.callback(Output("g-hist-raw", "figure"),
              Input("dd-hist-raw", "value"))
def cb_hist_raw(col):
    if not col or DF_RAW.empty or col not in DF_RAW.columns:
        return go.Figure()
    vals = DF_RAW[col].dropna()
    fig  = L(go.Figure([go.Histogram(
        x=vals, nbinsx=55,
        marker_color=C["blue"], opacity=0.85,
    )]),
        title=f"Distribution — {col}",
        xaxis_title=col, yaxis_title="Count")
    sk = float(vals.skew())
    fig.add_annotation(
        text=f"skew = {sk:.3f}   |   mean = {vals.mean():.2f}   |   "
             f"median = {vals.median():.2f}",
        xref="paper", yref="paper", x=0.5, y=1.07,
        showarrow=False,
        font=dict(color=C["red"] if abs(sk) > 1 else C["green"], size=11))
    return fig


# ── Tab 1 — boxplot (all data) ────────────────────────────────────
@app.callback(Output("g-box-raw", "figure"),
              Input("dd-box-raw", "value"))
def cb_box_raw(col):
    if not col or DF_RAW.empty or col not in DF_RAW.columns:
        return go.Figure()
    vals = DF_RAW[col].dropna()
    fig  = L(go.Figure([go.Box(
        y=vals, name=col,
        marker_color=C["blue"],
        line_color=C["blue"],
        fillcolor="rgba(59,130,246,0.2)",
        boxmean=True,
    )]),
        title=f"Boxplot — {col}", yaxis_title=col)
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr    = q3 - q1
    outliers = ((vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)).sum()
    fig.add_annotation(
        text=f"Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  "
             f"Outliers={outliers:,} ({outliers/len(vals):.2%})",
        xref="paper", yref="paper", x=0.5, y=1.07,
        showarrow=False,
        font=dict(color=C["muted"], size=11))
    return fig


# ── Tab 2 — histogram by class ────────────────────────────────────
@app.callback(Output("g-hist-cls", "figure"),
              Input("dd-hist-cls", "value"))
def cb_hist_cls(col):
    if not col or DF_TR.empty or col not in DF_TR.columns:
        return go.Figure()
    fig = go.Figure()
    if "Class" in DF_TR.columns:
        for cls, color, lbl in [(0, C["legit"], "Legit"),
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
    L(fig, title=f"{col} — Distribution by Class (density)",
      xaxis_title=col, yaxis_title="Density")
    return fig


# ── Tab 2 — boxplot by class ──────────────────────────────────────
@app.callback(Output("g-box-cls", "figure"),
              Input("dd-box-cls", "value"))
def cb_box_cls(col):
    if not col or DF_TR.empty or col not in DF_TR.columns:
        return go.Figure()
    fig = go.Figure()
    if "Class" in DF_TR.columns:
        for cls, color, lbl in [(0, C["legit"], "Legit"),
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
    L(fig, title=f"{col} — Boxplot by Class", yaxis_title=col)
    return fig


# ── Tab 3 — log transform comparison ─────────────────────────────
@app.callback(Output("g-log-tr", "figure"),
              Input("dd-log", "value"))
def cb_log_tr(col):
    if not col or AMT.empty or col not in AMT.columns:
        return go.Figure()
    vals  = AMT[col].dropna()
    color = C["red"] if col == "Amount" else C["green"]
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=["Histogram", "Box Plot"])
    fig.add_trace(
        go.Histogram(x=vals, nbinsx=60,
                     marker_color=color, opacity=0.82, name=col),
        row=1, col=1)
    fig.add_trace(
        go.Box(y=vals, name=col, marker_color=color,
               fillcolor=f"rgba({r},{g},{b},0.2)",
               line_color=color, boxmean=True),
        row=1, col=2)
    sk = float(vals.skew())
    fig.add_annotation(
        text=f"skew = {sk:.3f}   |   mean = {vals.mean():.4f}",
        xref="paper", yref="paper", x=0.5, y=1.07,
        showarrow=False,
        font=dict(color=C["red"] if abs(sk) > 1 else C["green"], size=12))
    L(fig, title=f"Log Transform — {col}", showlegend=False)
    fig.update_xaxes(gridcolor=C["line"])
    fig.update_yaxes(gridcolor=C["line"])
    return fig


## ----------------- ENTRY POINT -------------------


if __name__ == "__main__":
    print("\n" + "═" * 58)
    print("  Fraud Detection — EDA Dashboard")
    print("  Open in browser:  http://127.0.0.1:8050")
    print("  Ctrl+C to stop")
    print("═" * 58 + "\n")
    app.run(debug=True, port=8050, host="0.0.0.0")
