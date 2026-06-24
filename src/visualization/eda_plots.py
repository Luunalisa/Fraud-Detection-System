

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

## --------------- Style ----------------------

PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#0F6E56",
    "danger":    "#DC2626",
    "warn":      "#D97706",
    "neutral":   "#6B7280",
    "light":     "#F3F4F6",
    "fraud":     "#EF4444",
    "legit":     "#10B981",
}
FRAUD_PALETTE = [PALETTE["legit"], PALETTE["fraud"]]

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAFA",
    "axes.edgecolor":    "#E5E7EB",
    "axes.linewidth":    0.8,
    "grid.color":        "#F3F4F6",
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
})


def _save(fig: plt.Figure, name: str, save_dir: Optional[str]) -> None:
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")


class EDAVisualizer:
    """Produce all EDA and preprocessing visualizations."""

    def __init__(self, save_dir: Optional[str] = "reports/figures") -> None:
        self.save_dir = save_dir

    
    ## --------------- 1. Dataset overview table ----------------------------
    

    def plot_overview_table(
        self, df: pd.DataFrame, title: str = "Dataset Overview"
    ) -> plt.Figure:
        rows = [
            ["Shape",            f"{df.shape[0]:,} rows × {df.shape[1]} columns"],
            ["Memory",           f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB"],
            ["Numeric cols",     str(df.select_dtypes(include=np.number).shape[1])],
            ["Categorical cols", str(df.select_dtypes(include=["object", "category"]).shape[1])],
            ["Missing cells",    f"{df.isnull().sum().sum():,}"],
            ["Duplicate rows",   f"{df.duplicated().sum():,}"],
        ]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")
        tbl = ax.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#E5E7EB")
            if r == 0:
                cell.set_facecolor(PALETTE["primary"])
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#EFF6FF")
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
        _save(fig, "01_overview_table", self.save_dir)
        return fig

    ## ------------------- 2. Missing values heatmap ----------------------
   

    def plot_missing_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        null_pct = df.isnull().mean() * 100
        missing  = null_pct[null_pct > 0].sort_values(ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Missing Value Analysis", fontsize=13, fontweight="bold")

        if missing.empty:
            ax1.text(0.5, 0.5, "✓ No missing values",
                     ha="center", va="center", fontsize=14,
                     color=PALETTE["secondary"], fontweight="bold",
                     transform=ax1.transAxes)
            ax1.set_axis_off()
        else:
            colors = [PALETTE["danger"] if v > 30 else PALETTE["warn"] for v in missing.values]
            bars   = ax1.barh(missing.index, missing.values, color=colors, edgecolor="white")
            ax1.axvline(30, color=PALETTE["danger"], linestyle="--", lw=1, label="30% threshold")
            ax1.axvline(5,  color=PALETTE["warn"],   linestyle="--", lw=1, label="5% threshold")
            ax1.set_xlabel("Missing %")
            ax1.set_title("Missing % per Column")
            ax1.legend()
            for bar, val in zip(bars, missing.values):
                ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", fontsize=8)

        sample = df.sample(min(1000, len(df)), random_state=42).isnull().T
        cmap   = LinearSegmentedColormap.from_list("miss", ["#EFF6FF", "#DC2626"])
        ax2.imshow(sample.values, aspect="auto", cmap=cmap, interpolation="none")
        ax2.set_yticks(range(len(df.columns)))
        ax2.set_yticklabels(df.columns, fontsize=7)
        ax2.set_xlabel("Sample observations")
        ax2.set_title("Missingness pattern (1,000-row sample)")
        ax2.set_xticks([])

        plt.tight_layout()
        _save(fig, "02_missing_heatmap", self.save_dir)
        return fig

    
    ## --------------------- 3. Class distribution ----------------------
    

    def plot_class_distribution(
        self, y: pd.Series, class_names: Optional[Dict] = None
    ) -> plt.Figure:
        names = class_names or {0: "Legit", 1: "Fraud"}
        vc    = y.value_counts().sort_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle("Class Distribution (Target Variable)", fontsize=13, fontweight="bold")

        colors = [PALETTE["legit"], PALETTE["fraud"]]
        bars   = ax1.bar([names.get(k, k) for k in vc.index], vc.values,
                         color=colors[:len(vc)], edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, vc.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + vc.max() * 0.01,
                     f"{val:,}\n({val/len(y):.3%})", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
        ax1.set_title("Absolute counts")
        ax1.set_ylabel("Count")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        wedges, texts, autotexts = ax2.pie(
            vc.values,
            labels=[names.get(k, k) for k in vc.index],
            colors=colors[:len(vc)],
            autopct="%1.3f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            explode=[0, 0.12] if len(vc) > 1 else [0],
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")
        ax2.set_title("Proportion")

        ratio = vc.max() / vc.min() if vc.min() > 0 and len(vc) > 1 else 0
        fig.text(0.5, -0.02,
                 f"Imbalance ratio: {ratio:.0f}:1  |  Minority class: {vc.min()/len(y):.3%}",
                 ha="center", fontsize=9, color=PALETTE["danger"], fontweight="bold")

        plt.tight_layout()
        _save(fig, "03_class_distribution", self.save_dir)
        return fig

    
    ## ---------------------- 4. Feature histograms -----------------------------------
    #             split_by_class=False → plain histograms (EDA tab)
    #             split_by_class=True  → per-class overlay (Modeling tab)
    

    def plot_feature_histograms(
        self,
        df: pd.DataFrame,
        target: str = "Class",
        max_cols: int = 12,
        ncols: int = 4,
        split_by_class: bool = True,
    ) -> plt.Figure:
        num_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if c != target][:max_cols]
        nrows = int(np.ceil(len(num_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
        axes = np.array(axes).flatten()
        title_suffix = "by Class" if split_by_class else "(all data)"
        fig.suptitle(f"Numeric Feature Distributions {title_suffix}",
                     fontsize=13, fontweight="bold")

        for i, col in enumerate(num_cols):
            ax = axes[i]
            if split_by_class and target in df.columns:
                for cls, color, label in [(0, PALETTE["legit"], "Legit"),
                                           (1, PALETTE["fraud"], "Fraud")]:
                    subset = df[df[target] == cls][col].dropna()
                    ax.hist(subset, bins=40, alpha=0.55, color=color,
                            label=label, density=True, edgecolor="none")
                ax.legend(fontsize=7)
            else:
                ax.hist(df[col].dropna(), bins=40, color=PALETTE["primary"],
                        alpha=0.7, edgecolor="none")

            skew = float(df[col].skew())
            ax.set_title(f"{col}\nskew={skew:.2f}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_visible(False)

        for j in range(len(num_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        fname = "04_feature_histograms_by_class" if split_by_class else "04_feature_histograms"
        _save(fig, fname, self.save_dir)
        return fig

    
    ## --------------------------- 5. Boxplots --------------------------
    #              split_by_class=False → plain (EDA tab)
    #              split_by_class=True  → per-class (Modeling tab)


    def plot_boxplots(
        self,
        df: pd.DataFrame,
        target: str = "Class",
        top_n: int = 10,
        split_by_class: bool = True,
    ) -> plt.Figure:
        num_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if c != target][:top_n]
        ncols = 5
        nrows = int(np.ceil(len(num_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
        axes = np.array(axes).flatten()
        title_suffix = "by Class" if split_by_class else "— All Data"
        fig.suptitle(f"Boxplots — Outlier Detection {title_suffix}",
                     fontsize=13, fontweight="bold")

        for i, col in enumerate(num_cols):
            ax = axes[i]
            if split_by_class and target in df.columns:
                data   = [df[df[target] == 0][col].dropna(),
                          df[df[target] == 1][col].dropna()]
                labels = ["Legit", "Fraud"]
                colors = [PALETTE["legit"], PALETTE["fraud"]]
            else:
                data   = [df[col].dropna()]
                labels = [col]
                colors = [PALETTE["primary"]]

            bp = ax.boxplot(
                data, labels=labels,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 0.8},
                flierprops={"marker": ".", "markersize": 2,
                             "alpha": 0.4, "markerfacecolor": PALETTE["danger"]},
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)

            ax.set_title(col, fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=8)

        for j in range(len(num_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        fname = "05_boxplots_by_class" if split_by_class else "05_boxplots"
        _save(fig, fname, self.save_dir)
        return fig

    
    ## ----------------------- 6. Correlation heatmap -------------------
    

    def plot_correlation_heatmap(
        self, df: pd.DataFrame, max_cols: int = 20
    ) -> plt.Figure:
        num_cols = df.select_dtypes(include=np.number).columns[:max_cols]
        corr     = df[num_cols].corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr, mask=mask, cmap=cmap, center=0,
            vmin=-1, vmax=1,
            annot=len(num_cols) <= 15,
            fmt=".2f", annot_kws={"size": 7},
            square=True, linewidths=0.3,
            cbar_kws={"shrink": 0.7},
            ax=ax,
        )
        ax.set_title("Feature Correlation Matrix (lower triangle)", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
        plt.tight_layout()
        _save(fig, "06_correlation_heatmap", self.save_dir)
        return fig
    

    ## --------------------- 7. Feature–target correlation ---------------------
    

    def plot_feature_target_correlation(
        self, df: pd.DataFrame, target: str = "Class", top_n: int = 20
    ) -> plt.Figure:
        num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
        corr     = df[num_cols].corrwith(df[target]).dropna()
        corr     = corr.reindex(corr.abs().sort_values(ascending=False).index[:top_n])

        fig, ax = plt.subplots(figsize=(9, 6))
        colors  = [PALETTE["fraud"] if v < 0 else PALETTE["primary"] for v in corr.values]
        bars    = ax.barh(corr.index[::-1], corr.values[::-1],
                          color=colors[::-1], edgecolor="white", height=0.7)
        ax.axvline(0, color="#374151", linewidth=0.8)
        ax.set_xlabel("Pearson correlation with target")
        ax.set_title(f"Top {top_n} Feature Correlations with '{target}'",
                     fontsize=13, fontweight="bold")

        legend_patches = [
            mpatches.Patch(color=PALETTE["fraud"],   label="Negative correlation"),
            mpatches.Patch(color=PALETTE["primary"], label="Positive correlation"),
        ]
        ax.legend(handles=legend_patches, loc="lower right")

        for bar, val in zip(bars, corr.values[::-1]):
            ax.text(
                val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=7.5,
            )
        ax.grid(axis="x", alpha=0.4)
        plt.tight_layout()
        _save(fig, "07_feature_target_corr", self.save_dir)
        return fig
    

    ## ------------------------ 8. Amount distribution raw vs log -------------------------
   

    def plot_amount_distribution(
        self, df: pd.DataFrame, split_by_class: bool = False
    ) -> plt.Figure:
        if "Amount" not in df.columns:
            print("Column 'Amount' not found — skipping.")
            return plt.figure()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Amount Feature — Raw Distribution", fontsize=13, fontweight="bold")

        target  = "Class" if "Class" in df.columns else None

        # Left: raw histogram
        axes[0].hist(df["Amount"].dropna(), bins=60, color=PALETTE["primary"],
                     alpha=0.75, edgecolor="none")
        axes[0].set_title("Raw Amount")
        axes[0].set_ylabel("Frequency")
        skew = float(df["Amount"].skew())
        axes[0].text(0.97, 0.95, f"skew={skew:.2f}", transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=9,
                     color=PALETTE["danger"] if abs(skew) > 1 else PALETTE["secondary"])

        # Right: log1p histogram
        log_vals = np.log1p(df["Amount"].dropna())
        axes[1].hist(log_vals, bins=60, color=PALETTE["secondary"],
                     alpha=0.75, edgecolor="none")
        axes[1].set_title("log1p(Amount)")
        axes[1].set_ylabel("Frequency")
        skew2 = float(log_vals.skew())
        axes[1].text(0.97, 0.95, f"skew={skew2:.2f}", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=9,
                     color=PALETTE["danger"] if abs(skew2) > 1 else PALETTE["secondary"])

        plt.tight_layout()
        _save(fig, "08_amount_distribution", self.save_dir)
        return fig

    
    ## ---------------------- 9. Time / hourly analysis ------------------------
    #             show_fraud_rate controls which panel renders
    

    def plot_time_analysis(
        self, df: pd.DataFrame, show_fraud_rate: bool = True
    ) -> plt.Figure:
        if "Time" not in df.columns:
            print("Column 'Time' not found — skipping.")
            return plt.figure()

        df = df.copy()
        df["hour"] = (df["Time"] % 86400) // 3600
        target     = "Class" if "Class" in df.columns else None

        if show_fraud_rate and target:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        else:
            fig, ax1 = plt.subplots(figsize=(8, 4))
        fig.suptitle("Temporal Analysis", fontsize=13, fontweight="bold")

        # Hourly volume
        hourly = df.groupby("hour").size()
        ax1.bar(hourly.index, hourly.values, color=PALETTE["primary"], alpha=0.75, edgecolor="none")
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("Transaction count")
        ax1.set_title("Transactions per Hour")
        ax1.set_xticks(range(0, 24, 2))
        ax1.axvspan(22, 24, alpha=0.08, color=PALETTE["danger"], label="Night (22–06)")
        ax1.axvspan(0,  6,  alpha=0.08, color=PALETTE["danger"])
        ax1.legend()

        if show_fraud_rate and target and target in df.columns:
            fraud_rate = df.groupby("hour")[target].mean() * 100
            ax2.plot(fraud_rate.index, fraud_rate.values,
                     color=PALETTE["fraud"], linewidth=2, marker="o", markersize=4)
            ax2.fill_between(fraud_rate.index, fraud_rate.values,
                             alpha=0.15, color=PALETTE["fraud"])
            ax2.set_xlabel("Hour of day")
            ax2.set_ylabel("Fraud rate (%)")
            ax2.set_title("Fraud Rate by Hour of Day")
            ax2.set_xticks(range(0, 24, 2))
            ax2.axvspan(22, 24, alpha=0.08, color=PALETTE["danger"], label="Night window")
            ax2.axvspan(0,  6,  alpha=0.08, color=PALETTE["danger"])
            ax2.legend()

        plt.tight_layout()
        fname = "09_time_analysis_full" if show_fraud_rate else "09_time_analysis"
        _save(fig, fname, self.save_dir)
        return fig

    ## ------------------------ 10. SMOTE comparison ------------------- 
    

    def plot_smote_comparison(
        self,
        y_before: pd.Series,
        y_after:  pd.Series,
        class_names: Optional[Dict] = None,
    ) -> plt.Figure:
        names = class_names or {0: "Legit", 1: "Fraud"}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle("SMOTE — Before vs After Oversampling (Train set only)",
                     fontsize=13, fontweight="bold")

        for ax, y, title in zip(axes, [y_before, y_after],
                                 ["Before SMOTE", "After SMOTE"]):
            vc     = y.value_counts().sort_index()
            colors = [PALETTE["legit"], PALETTE["fraud"]]
            bars   = ax.bar([names.get(k, k) for k in vc.index], vc.values,
                            color=colors[:len(vc)], edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + vc.max() * 0.01,
                        f"{val:,}\n({val/len(y):.1%})",
                        ha="center", va="bottom", fontsize=9)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Count")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        plt.tight_layout()
        _save(fig, "10_smote_comparison", self.save_dir)
        return fig

    ## ------------------------ 11. Split donut ---------------------

    
    def plot_split_donut(
        self,
        n_train: int, n_val: int, n_test: int,
        fraud_train: int, fraud_val: int, fraud_test: int,
    ) -> plt.Figure:
        total = n_train + n_val + n_test
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Train / Validation / Test Split", fontsize=13, fontweight="bold")

        sizes  = [n_train, n_val, n_test]
        labels = [
            f"Train\n{n_train:,} ({n_train/total:.0%})",
            f"Validation\n{n_val:,} ({n_val/total:.0%})",
            f"Test\n{n_test:,} ({n_test/total:.0%})",
        ]
        colors = ["#10B981", "#3B82F6", "#F59E0B"]
        wedges, _ = ax1.pie(sizes, labels=labels, colors=colors,
                             startangle=90,
                             wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2})
        ax1.text(0, 0, f"{total:,}\ntotal", ha="center", va="center",
                 fontsize=12, fontweight="bold", color="#374151")
        ax1.set_title("Data partitioning", fontweight="bold")

        sets      = ["Train", "Validation", "Test"]
        counts    = [fraud_train, fraud_val, fraud_test]
        ns        = [n_train, n_val, n_test]
        rates     = [f"{c/n:.4%}" for c, n in zip(counts, ns)]
        cell_data = list(zip(
            [f"{n:,}" for n in ns],
            [f"{c:,}" for c in counts],
            rates,
        ))
        tbl = ax2.table(
            cellText=cell_data,
            rowLabels=sets,
            colLabels=["Rows", "Fraud", "Fraud rate"],
            cellLoc="center", rowLoc="center",
            loc="center", bbox=[0, 0.1, 1, 0.8],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#E5E7EB")
            if r == 0:
                cell.set_facecolor("#1E40AF")
                cell.set_text_props(color="white", fontweight="bold")
            elif c == -1:
                cell.set_facecolor(colors[r - 1] if r > 0 else "#fff")
                cell.set_text_props(fontweight="bold")
        ax2.axis("off")
        ax2.set_title("Split statistics", fontweight="bold")

        plt.tight_layout()
        _save(fig, "11_split_donut", self.save_dir)
        return fig


    ## ---------------------- 12. Feature importance (MI scores) --------------------
    
    def plot_feature_importance(
        self,
        scores: Dict[str, float],
        title: str = "Feature Importance (Mutual Information)",
        top_n: int = 20,
    ) -> plt.Figure:
        series = pd.Series(scores).sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
        cmap    = plt.cm.Blues(np.linspace(0.35, 0.85, len(series)))
        bars    = ax.barh(series.index[::-1], series.values[::-1],
                          color=cmap[::-1], edgecolor="none", height=0.7)
        for bar, val in zip(bars, series.values[::-1]):
            ax.text(bar.get_width() + series.max() * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Score")
        ax.grid(axis="x", alpha=0.4)
        plt.tight_layout()
        _save(fig, "12_feature_importance", self.save_dir)
        return fig

    ## ------------------- 13. Skewness before / after ----------------------
    

    def plot_skewness(
        self,
        df_before: pd.DataFrame,
        df_after:  pd.DataFrame,
        cols: Optional[List[str]] = None,
    ) -> plt.Figure:
        num_before = df_before.select_dtypes(include=np.number)
        num_after  = df_after.select_dtypes(include=np.number)
        common     = list(set(num_before.columns) & set(num_after.columns))
        if cols:
            common = [c for c in cols if c in common]

        skew_before = num_before[common].skew().abs().sort_values(ascending=False).head(15)
        skew_after  = num_after[common].skew().abs().reindex(skew_before.index)

        x   = np.arange(len(skew_before))
        w   = 0.38
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w / 2, skew_before.values, width=w, label="Before transform",
               color=PALETTE["danger"],    alpha=0.75, edgecolor="none")
        ax.bar(x + w / 2, skew_after.values,  width=w, label="After transform",
               color=PALETTE["secondary"], alpha=0.75, edgecolor="none")
        ax.axhline(1, color="#374151", linestyle="--", linewidth=0.8, label="|skew|=1 threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(skew_before.index, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("|Skewness|")
        ax.set_title("Skewness Reduction (Yeo-Johnson / log transform)",
                     fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        _save(fig, "13_skewness", self.save_dir)
        return fig

    ## ---------------------- 14. Log-transform comparison  --------------------------
    #             Shows raw vs transformed side-by-side for
    #             every column pair supplied.

    def plot_log_transform_comparison(
        self,
        df_raw:         pd.DataFrame,
        df_transformed: pd.DataFrame,
        cols: Optional[List[str]] = None,
        max_cols: int = 6,
    ) -> plt.Figure:
        
        num_common = list(
            set(df_raw.select_dtypes(include=np.number).columns) &
            set(df_transformed.select_dtypes(include=np.number).columns)
        )
        if cols:
            num_common = [c for c in cols if c in num_common]
        num_common = num_common[:max_cols]

        n   = len(num_common)
        fig, axes = plt.subplots(n, 2, figsize=(10, n * 2.5))
        if n == 1:
            axes = [axes]
        fig.suptitle("Log / Yeo-Johnson Transform — Before vs After",
                     fontsize=13, fontweight="bold")

        for i, col in enumerate(num_common):
            ax_raw, ax_tr = axes[i]
            raw_vals = df_raw[col].dropna()
            tr_vals  = df_transformed[col].dropna()

            ax_raw.hist(raw_vals, bins=50, color=PALETTE["danger"],   alpha=0.7, edgecolor="none")
            ax_tr.hist( tr_vals,  bins=50, color=PALETTE["secondary"], alpha=0.7, edgecolor="none")

            skew_r = float(raw_vals.skew())
            skew_t = float(tr_vals.skew())
            ax_raw.set_title(f"{col}  |  raw  skew={skew_r:.2f}", fontsize=9)
            ax_tr.set_title( f"{col}  |  transformed  skew={skew_t:.2f}", fontsize=9)
            for ax in (ax_raw, ax_tr):
                ax.yaxis.set_visible(False)
                ax.tick_params(labelsize=7)

        plt.tight_layout()
        _save(fig, "15_log_transform_comparison", self.save_dir)
        return fig

    
    ## ------------------- 15. RFE ranking plot ----------------------
    
    def plot_rfe_ranking(
        self,
        ranking: Dict[str, int],
        n_features_selected: Optional[int] = None,
        title: str = "Recursive Feature Elimination — Feature Ranking",
    ) -> plt.Figure:
        
        series = pd.Series(ranking).sort_values()

        colors = [
            PALETTE["secondary"] if r == 1 else
            (PALETTE["warn"]     if r <= 3 else PALETTE["neutral"])
            for r in series.values
        ]

        fig, ax = plt.subplots(figsize=(10, max(5, len(series) * 0.3)))
        bars = ax.barh(series.index[::-1], series.values[::-1],
                       color=colors[::-1], edgecolor="none", height=0.7)

        if n_features_selected is not None:
            ax.axvline(1.5, color=PALETTE["danger"], linestyle="--",
                       linewidth=1.2, label=f"Selected ({n_features_selected} features)")
            ax.legend()

        for bar, val in zip(bars, series.values[::-1]):
            ax.text(bar.get_width() + 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    f"rank {val}", va="center", fontsize=7.5)

        ax.set_xlabel("RFE Rank  (1 = selected)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.4)

        legend_patches = [
            mpatches.Patch(color=PALETTE["secondary"], label="Selected (rank 1)"),
            mpatches.Patch(color=PALETTE["warn"],      label="Rank 2–3"),
            mpatches.Patch(color=PALETTE["neutral"],   label="Eliminated"),
        ]
        ax.legend(handles=legend_patches, loc="lower right")

        plt.tight_layout()
        _save(fig, "14_rfe_ranking", self.save_dir)
        return fig
