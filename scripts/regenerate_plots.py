"""Regenerate thesis-quality plots from results/summary.csv.

Improvements over the default plots:
  - Short model labels (readable in PDF)
  - Grouped by model with track as color
  - fontsize 12-13, DPI 300
  - Horizontal bars where label density is high
  - Output: results/thesis_*.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

RESULTS_DIR = Path("results")
OUT_PREFIX = "thesis_"

MODEL_SHORT: dict[str, str] = {
    "claude-opus-4-6":   "Claude Opus",
    "claude-sonnet-4-6": "Claude Sonnet",
    "claude-haiku-4-5":  "Claude Haiku",
    "gpt-4o":            "GPT-4o",
    "gpt-4o-mini":       "GPT-4o-mini",
    "llama3.2":          "Llama 3.2",
}

TRACK_COLOR: dict[str, str] = {
    "blind":  "#4C72B0",
    "kg":     "#55A868",
    "hybrid": "#C44E52",
}

TRACK_LABEL: dict[str, str] = {
    "blind":  "Blind",
    "kg":     "KG (Wikidata)",
    "hybrid": "Hybrid (KG + Wikipedia)",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#F7F7F7",
    "figure.facecolor": "white",
})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _short(name: str) -> str:
    return MODEL_SHORT.get(name, name)


def _save(fig: plt.Figure, name: str) -> None:
    path = RESULTS_DIR / f"{OUT_PREFIX}{name}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def _pct_formatter(x, _):
    return f"{x:.0%}"


# ------------------------------------------------------------------
# 1. Hallucination rates  (grouped by model, colour = track)
# ------------------------------------------------------------------

def plot_hallucination_rates(df: pd.DataFrame) -> None:
    models = df["model"].unique().tolist()
    tracks = df["track"].unique().tolist()

    n_models = len(models)
    n_tracks = len(tracks)
    width = 0.22
    x = np.arange(n_models)
    offsets = np.linspace(-(n_tracks - 1) / 2, (n_tracks - 1) / 2, n_tracks) * width

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (col, title) in enumerate([
        ("hallucination_rate",          "Overall Hallucination Rate"),
    ]):
        ax = axes[0]
        for t_idx, track in enumerate(tracks):
            sub = df[df["track"] == track].set_index("model")
            vals = [sub.loc[m, col] if m in sub.index else float("nan") for m in models]
            bars = ax.bar(
                x + offsets[t_idx], vals, width,
                label=TRACK_LABEL.get(track, track),
                color=TRACK_COLOR.get(track, "#888"),
                alpha=0.9,
            )
            for bar, v in zip(bars, vals):
                if not math.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + 0.008,
                        f"{v:.1%}",
                        ha="center", va="bottom", fontsize=9,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in models], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Hallucination Rate")
        ax.set_ylim(0, min(1.0, df["hallucination_rate"].max() * 1.35))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.35)

    # Right subplot: SUPPORTS vs REFUTES breakdown (blind track only)
    ax2 = axes[1]
    blind = df[df["track"] == "blind"].set_index("model")
    sup_vals  = [blind.loc[m, "supports_hallucination_rate"] if m in blind.index else float("nan") for m in models]
    ref_vals  = [blind.loc[m, "refutes_hallucination_rate"]  if m in blind.index else float("nan") for m in models]
    b1 = ax2.bar(x - width / 2, sup_vals, width, label="SUPPORTS claims", color="seagreen", alpha=0.9)
    b2 = ax2.bar(x + width / 2, ref_vals, width, label="REFUTES claims",  color="tomato",   alpha=0.9)
    for bars in [b1, b2]:
        for bar, v in zip(bars, sup_vals if bars is b1 else ref_vals):
            if not math.isnan(v):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.008,
                    f"{v:.1%}",
                    ha="center", va="bottom", fontsize=9,
                )
    ax2.set_xticks(x)
    ax2.set_xticklabels([_short(m) for m in models], rotation=25, ha="right")
    ax2.set_title("SUPPORTS vs REFUTES Hallucination (Blind track)")
    ax2.set_ylabel("Hallucination Rate")
    ax2.set_ylim(0, min(1.0, max(sup_vals + ref_vals) * 1.35))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", alpha=0.35)

    fig.tight_layout(pad=2.0)
    _save(fig, "hallucination_rates.png")


# ------------------------------------------------------------------
# 2. Statistical metrics  (Precision/Recall/F1 + Kappa/MCC)
# ------------------------------------------------------------------

def plot_statistical_metrics(df: pd.DataFrame) -> None:
    models = df["model"].unique().tolist()
    tracks = df["track"].unique().tolist()

    n_models = len(models)
    n_tracks = len(tracks)
    width = 0.22
    x = np.arange(n_models)
    offsets = np.linspace(-(n_tracks - 1) / 2, (n_tracks - 1) / 2, n_tracks) * width

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metrics, title in [
        (ax1, [("precision_hallucination", "Precision"), ("recall_hallucination", "Recall"), ("f1_hallucination", "F1")], "Precision / Recall / F1"),
        (ax2, [("cohen_kappa", "Cohen κ"), ("mcc", "MCC")], "Agreement Metrics (Kappa & MCC)"),
    ]:
        n_metrics = len(metrics)
        m_width = width / n_metrics * 1.8
        m_offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * m_width * 0.9

        for t_idx, track in enumerate(tracks):
            sub = df[df["track"] == track].set_index("model")
            base_offset = offsets[t_idx]
            for m_idx, (col, lbl) in enumerate(metrics):
                vals = [sub.loc[m, col] if m in sub.index else float("nan") for m in models]
                final_offsets = base_offset + m_offsets[m_idx]
                bars = ax.bar(
                    x + final_offsets, vals, m_width,
                    color=TRACK_COLOR.get(track, "#888"),
                    alpha=0.55 + 0.22 * m_idx,
                    label=f"{TRACK_LABEL.get(track, track)} – {lbl}" if t_idx == 0 else None,
                )

        # Add track legend patches instead
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=TRACK_COLOR[t], label=TRACK_LABEL.get(t, t))
            for t in tracks if t in TRACK_COLOR
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        # Metric labels on x-axis (show per model group)
        # Use short model names
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in models], rotation=25, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.35)

        # Metric sub-labels below x-axis
        metric_names = [lbl for _, lbl in metrics]
        for xi, model in enumerate(models):
            for m_idx, (col, lbl) in enumerate(metrics):
                base_offset = offsets[0]  # use first track for position
                xpos = xi + m_offsets[m_idx]
                ax.text(xpos, -0.09, lbl, ha="center", va="top",
                        fontsize=7, color="#555", transform=ax.get_xaxis_transform())

    fig.tight_layout(pad=2.0)
    _save(fig, "statistical_metrics.png")


# ------------------------------------------------------------------
# 3. NLI label distribution  (one subplot per track)
# ------------------------------------------------------------------

def plot_nli_distribution(df: pd.DataFrame) -> None:
    tracks = df["track"].unique().tolist()
    models_ordered = df["model"].unique().tolist()

    fig, axes = plt.subplots(1, len(tracks), figsize=(6 * len(tracks), 6), sharey=True)
    if len(tracks) == 1:
        axes = [axes]

    for ax, track in zip(axes, tracks):
        sub = df[df["track"] == track].copy()
        sub["label"] = sub["model"].map(_short)
        sub = sub.set_index("label")
        # reorder by models_ordered
        ordered_labels = [_short(m) for m in models_ordered if _short(m) in sub.index]
        sub = sub.loc[ordered_labels]

        ent  = sub["nli_entailment_pct"].values
        neu  = sub["nli_neutral_pct"].values
        cont = sub["nli_contradiction_pct"].values
        y = np.arange(len(sub))

        ax.barh(y, ent,  height=0.6, label="Entailment",    color="#55A868", alpha=0.9)
        ax.barh(y, neu,  height=0.6, left=ent,              label="Neutral",      color="gold",     alpha=0.9)
        ax.barh(y, cont, height=0.6, left=ent + neu,        label="Contradiction", color="tomato",   alpha=0.9)

        for i, (e, n, c) in enumerate(zip(ent, neu, cont)):
            for val, offset, col in [(e, e/2, "white"), (n, e+n/2, "#555"), (c, e+n+c/2, "white")]:
                if val > 0.04:
                    ax.text(offset, i, f"{val:.0%}", ha="center", va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(sub.index, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_title(f"NLI Distribution — {TRACK_LABEL.get(track, track)}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
        ax.grid(axis="x", alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout(pad=2.0)
    _save(fig, "nli_distribution.png")


# ------------------------------------------------------------------
# 4. Cost efficiency
# ------------------------------------------------------------------

def plot_cost_efficiency(df: pd.DataFrame) -> None:
    tracks  = df["track"].unique().tolist()
    models  = df["model"].unique().tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: cost per 1k samples grouped bar (cloud models only)
    cloud_df = df[df["total_cost_usd"].notna() & (df["total_cost_usd"] > 0)].copy()
    cloud_models = cloud_df["model"].unique().tolist()
    n_tracks = len(tracks)
    width = 0.22
    x = np.arange(len(cloud_models))
    offsets = np.linspace(-(n_tracks-1)/2, (n_tracks-1)/2, n_tracks) * width

    for t_idx, track in enumerate(tracks):
        sub = cloud_df[cloud_df["track"] == track].set_index("model")
        vals = [sub.loc[m, "total_cost_usd"] if m in sub.index else float("nan") for m in cloud_models]
        bars = ax1.bar(
            x + offsets[t_idx], vals, width,
            label=TRACK_LABEL.get(track, track),
            color=TRACK_COLOR.get(track, "#888"),
            alpha=0.9,
        )
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.005,
                    f"${v:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([_short(m) for m in cloud_models], rotation=25, ha="right")
    ax1.set_title("Total Cost per Run (300 samples)")
    ax1.set_ylabel("Cost (USD)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.35)

    # Right: scatter hallucination rate vs cost per 1k (blind track only)
    blind = df[df["track"] == "blind"].copy()
    for _, row in blind.iterrows():
        hr = row["hallucination_rate"]
        cps = row.get("cost_per_sample_usd")
        cost_1k = float(cps) * 1000 if pd.notna(cps) and cps else 0.0
        is_local = cost_1k == 0.0
        marker = "^" if is_local else "o"
        color  = "#AAAAAA" if is_local else TRACK_COLOR["blind"]
        ax2.scatter(cost_1k, hr, marker=marker, s=120, color=color, zorder=3)
        offset_x = 2 if cost_1k > 0 else 0.2
        ax2.annotate(
            _short(row["model"]),
            (cost_1k, hr),
            textcoords="offset points", xytext=(6, 4),
            fontsize=11,
        )

    ax2.set_xlabel("Cost per 1 000 samples (USD)", fontsize=12)
    ax2.set_ylabel("Hallucination Rate", fontsize=12)
    ax2.set_title("Cost vs. Hallucination Rate (Blind track)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax2.grid(alpha=0.3)

    fig.tight_layout(pad=2.0)
    _save(fig, "cost_efficiency.png")


# ------------------------------------------------------------------
# 5. Latency & tokens
# ------------------------------------------------------------------

def plot_latency_tokens(df: pd.DataFrame) -> None:
    models = df["model"].unique().tolist()
    tracks = df["track"].unique().tolist()

    n_models = len(models)
    n_tracks = len(tracks)
    width = 0.22
    x = np.arange(n_models)
    offsets = np.linspace(-(n_tracks-1)/2, (n_tracks-1)/2, n_tracks) * width

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: avg latency per model × track
    for t_idx, track in enumerate(tracks):
        sub = df[df["track"] == track].set_index("model")
        vals = [sub.loc[m, "avg_latency_s"] if m in sub.index else float("nan") for m in models]
        bars = ax1.bar(
            x + offsets[t_idx], vals, width,
            label=TRACK_LABEL.get(track, track),
            color=TRACK_COLOR.get(track, "#888"),
            alpha=0.9,
        )
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.05,
                    f"{v:.1f}s",
                    ha="center", va="bottom", fontsize=8,
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([_short(m) for m in models], rotation=25, ha="right")
    ax1.set_title("Average Latency per Sample")
    ax1.set_ylabel("Latency (s)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.35)

    # Right: avg tokens per sample × track (stacked input/output)
    for t_idx, track in enumerate(tracks):
        sub = df[df["track"] == track].set_index("model")
        avg_in  = [sub.loc[m, "total_input_tokens"]  / 300 if m in sub.index else 0 for m in models]
        avg_out = [sub.loc[m, "total_output_tokens"] / 300 if m in sub.index else 0 for m in models]
        xi = x + offsets[t_idx]
        b_in  = ax2.bar(xi, avg_in,  width, color=TRACK_COLOR.get(track, "#888"), alpha=0.9, label=f"{TRACK_LABEL.get(track, track)} – input")
        b_out = ax2.bar(xi, avg_out, width, bottom=avg_in, color=TRACK_COLOR.get(track, "#888"), alpha=0.45, label=f"{TRACK_LABEL.get(track, track)} – output")
        for xi_val, vi, vo in zip(xi, avg_in, avg_out):
            total = vi + vo
            if total > 0:
                ax2.text(xi_val, total + 2, f"{total:.0f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([_short(m) for m in models], rotation=25, ha="right")
    ax2.set_title("Average Tokens per Sample (input + output)")
    ax2.set_ylabel("Tokens")
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=TRACK_COLOR[t], label=TRACK_LABEL.get(t, t)) for t in tracks if t in TRACK_COLOR]
    ax2.legend(handles=legend_els)
    ax2.grid(axis="y", alpha=0.35)

    fig.tight_layout(pad=2.0)
    _save(fig, "latency_tokens.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = RESULTS_DIR / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    print("Generating thesis plots...")
    plot_hallucination_rates(df)
    plot_statistical_metrics(df)
    plot_nli_distribution(df)
    plot_cost_efficiency(df)
    plot_latency_tokens(df)
    print("Done. Plots saved to results/thesis_*.png")
