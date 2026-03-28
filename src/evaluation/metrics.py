"""Evaluation metrics and report generation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

console = Console()

# Global plot style
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#F7F7F7",
    "figure.facecolor": "white",
})

# Token cost per 1k tokens in USD; None = local/free model
TOKEN_COST_PER_1K: dict[str, dict[str, Optional[float]]] = {
    "claude-opus-4-6":   {"input": 0.015,   "output": 0.075},
    "claude-sonnet-4-6": {"input": 0.003,   "output": 0.015},
    "claude-haiku-4-5":  {"input": 0.0008,  "output": 0.004},
    "gpt-4o":            {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":       {"input": 0.00015, "output": 0.0006},
    "llama3.2":          {"input": None,    "output": None},
}


# ------------------------------------------------------------------
# Per-sample result
# ------------------------------------------------------------------

@dataclass
class SampleResult:
    sample_id: int
    claim: str
    fever_label: str             # SUPPORTS | REFUTES
    llm_response: str
    llm_error: Optional[str]
    nli_label: str               # ENTAILMENT | NEUTRAL | CONTRADICTION
    nli_entailment: float
    nli_neutral: float
    nli_contradiction: float
    hallucination: Optional[bool]  # True | False | None (ambiguous)
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


# ------------------------------------------------------------------
# Aggregated metrics for one LLM
# ------------------------------------------------------------------

@dataclass
class LLMMetrics:
    model_name: str
    provider: str
    n_total: int = 0
    n_evaluated: int = 0       # excludes API errors and NEUTRAL
    n_hallucinations: int = 0
    n_correct: int = 0
    n_neutral: int = 0
    n_errors: int = 0

    # Per-label breakdown
    n_supports_total: int = 0
    n_supports_hallucination: int = 0
    n_refutes_total: int = 0
    n_refutes_hallucination: int = 0

    # NLI distribution
    nli_entailment_count: int = 0
    nli_neutral_count: int = 0
    nli_contradiction_count: int = 0

    sample_results: list[SampleResult] = field(default_factory=list, repr=False)

    # Private accumulators (populated for every sample, including errors)
    _latency_list: list[float] = field(default_factory=list, repr=False)
    _input_tokens_list: list[int] = field(default_factory=list, repr=False)
    _output_tokens_list: list[int] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Existing rate properties
    # ------------------------------------------------------------------

    @property
    def hallucination_rate(self) -> float:
        """Fraction of evaluated samples where hallucination was detected."""
        if self.n_evaluated == 0:
            return float("nan")
        return self.n_hallucinations / self.n_evaluated

    @property
    def accuracy(self) -> float:
        """Fraction of evaluated samples that were correct (not hallucinating)."""
        if self.n_evaluated == 0:
            return float("nan")
        return self.n_correct / self.n_evaluated

    @property
    def supports_hallucination_rate(self) -> float:
        if self.n_supports_total == 0:
            return float("nan")
        return self.n_supports_hallucination / self.n_supports_total

    @property
    def refutes_hallucination_rate(self) -> float:
        if self.n_refutes_total == 0:
            return float("nan")
        return self.n_refutes_hallucination / self.n_refutes_total

    # ------------------------------------------------------------------
    # Latency properties
    # ------------------------------------------------------------------

    @property
    def avg_latency_s(self) -> float:
        if not self._latency_list:
            return float("nan")
        return float(np.mean(self._latency_list))

    @property
    def median_latency_s(self) -> float:
        if not self._latency_list:
            return float("nan")
        return float(np.median(self._latency_list))

    @property
    def p95_latency_s(self) -> float:
        if not self._latency_list:
            return float("nan")
        return float(np.percentile(self._latency_list, 95))

    # ------------------------------------------------------------------
    # Token properties
    # ------------------------------------------------------------------

    @property
    def total_input_tokens(self) -> int:
        return sum(self._input_tokens_list)

    @property
    def total_output_tokens(self) -> int:
        return sum(self._output_tokens_list)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_tokens_per_sample(self) -> float:
        if not self._input_tokens_list:
            return float("nan")
        totals = [i + o for i, o in zip(self._input_tokens_list, self._output_tokens_list)]
        return float(np.mean(totals))

    # ------------------------------------------------------------------
    # Cost properties
    # ------------------------------------------------------------------

    @property
    def total_cost_usd(self) -> Optional[float]:
        rates = TOKEN_COST_PER_1K.get(self.model_name)
        if rates is None or rates["input"] is None or rates["output"] is None:
            return None
        return (
            self.total_input_tokens * rates["input"]
            + self.total_output_tokens * rates["output"]
        ) / 1000.0

    @property
    def cost_per_sample_usd(self) -> Optional[float]:
        cost = self.total_cost_usd
        if cost is None or self.n_total == 0:
            return None
        return cost / self.n_total

    # ------------------------------------------------------------------
    # Binary classification helpers
    # ------------------------------------------------------------------

    def _binary_arrays(self) -> tuple[list[int], list[int]]:
        """Build (y_true, y_pred) for binary hallucination classification.

        Skips samples with llm_error set or hallucination is None.
        y_true = 1 for (SUPPORTS+CONTRADICTION) or (REFUTES+ENTAILMENT), else 0.
        y_pred = 1 if hallucination == True, else 0.
        Returns ([], []) when no valid samples.
        """
        y_true: list[int] = []
        y_pred: list[int] = []
        for r in self.sample_results:
            if r.llm_error or r.hallucination is None:
                continue
            is_true_halluc = (
                (r.fever_label == "SUPPORTS" and r.nli_label == "CONTRADICTION")
                or (r.fever_label == "REFUTES" and r.nli_label == "ENTAILMENT")
            )
            y_true.append(1 if is_true_halluc else 0)
            y_pred.append(1 if r.hallucination else 0)
        return y_true, y_pred

    @property
    def cohen_kappa(self) -> float:
        y_true, y_pred = self._binary_arrays()
        if len(y_true) < 2:
            return float("nan")
        try:
            return float(cohen_kappa_score(y_true, y_pred))
        except Exception:
            return float("nan")

    @property
    def mcc(self) -> float:
        y_true, y_pred = self._binary_arrays()
        if len(y_true) < 2:
            return float("nan")
        try:
            return float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            return float("nan")

    @property
    def precision_hallucination(self) -> float:
        y_true, y_pred = self._binary_arrays()
        if not y_true:
            return float("nan")
        try:
            p, _, _, _ = precision_recall_fscore_support(
                y_true, y_pred, pos_label=1, average="binary", zero_division=0
            )
            return float(p)
        except Exception:
            return float("nan")

    @property
    def recall_hallucination(self) -> float:
        y_true, y_pred = self._binary_arrays()
        if not y_true:
            return float("nan")
        try:
            _, r, _, _ = precision_recall_fscore_support(
                y_true, y_pred, pos_label=1, average="binary", zero_division=0
            )
            return float(r)
        except Exception:
            return float("nan")

    @property
    def f1_hallucination(self) -> float:
        y_true, y_pred = self._binary_arrays()
        if not y_true:
            return float("nan")
        try:
            _, _, f, _ = precision_recall_fscore_support(
                y_true, y_pred, pos_label=1, average="binary", zero_division=0
            )
            return float(f)
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Per-category error analysis
    # ------------------------------------------------------------------

    def error_category_counts(self) -> dict[str, int]:
        """Count occurrences of each fever_label → nli_label combination.

        Skips samples where llm_error is set.
        """
        counts: dict[str, int] = {}
        for r in self.sample_results:
            if r.llm_error:
                continue
            key = f"{r.fever_label}→{r.nli_label}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Core accumulator
    # ------------------------------------------------------------------

    def add_result(self, result: SampleResult) -> None:
        self.sample_results.append(result)
        self.n_total += 1

        # Always track latency and tokens (latency is real even on failure)
        self._latency_list.append(result.latency_s)
        self._input_tokens_list.append(result.input_tokens)
        self._output_tokens_list.append(result.output_tokens)

        if result.llm_error:
            self.n_errors += 1
            return

        # NLI distribution
        if result.nli_label == "ENTAILMENT":
            self.nli_entailment_count += 1
        elif result.nli_label == "NEUTRAL":
            self.nli_neutral_count += 1
        else:
            self.nli_contradiction_count += 1

        # Per-fever-label counts
        if result.fever_label == "SUPPORTS":
            self.n_supports_total += 1
        elif result.fever_label == "REFUTES":
            self.n_refutes_total += 1

        if result.hallucination is None:
            self.n_neutral += 1
            return

        self.n_evaluated += 1
        if result.hallucination:
            self.n_hallucinations += 1
            if result.fever_label == "SUPPORTS":
                self.n_supports_hallucination += 1
            elif result.fever_label == "REFUTES":
                self.n_refutes_hallucination += 1
        else:
            self.n_correct += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary_dict(self) -> dict:
        def _f(v: Optional[float], ndigits: int = 4) -> Optional[float]:
            if v is None:
                return None
            if isinstance(v, float) and math.isnan(v):
                return None
            return round(v, ndigits)

        return {
            "model": self.model_name,
            "provider": self.provider,
            "n_total": self.n_total,
            "n_evaluated": self.n_evaluated,
            "n_errors": self.n_errors,
            "n_neutral_nli": self.n_neutral,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "n_hallucinations": self.n_hallucinations,
            "n_correct": self.n_correct,
            "supports_hallucination_rate": round(self.supports_hallucination_rate, 4),
            "refutes_hallucination_rate": round(self.refutes_hallucination_rate, 4),
            "nli_entailment_pct": round(self.nli_entailment_count / max(self.n_total, 1), 4),
            "nli_neutral_pct": round(self.nli_neutral_count / max(self.n_total, 1), 4),
            "nli_contradiction_pct": round(self.nli_contradiction_count / max(self.n_total, 1), 4),
            # New: latency
            "avg_latency_s": _f(self.avg_latency_s),
            "median_latency_s": _f(self.median_latency_s),
            "p95_latency_s": _f(self.p95_latency_s),
            # New: tokens
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_sample": _f(self.avg_tokens_per_sample, 1),
            # New: cost
            "total_cost_usd": _f(self.total_cost_usd, 6),
            "cost_per_sample_usd": _f(self.cost_per_sample_usd, 8),
            # New: classification
            "precision_hallucination": _f(self.precision_hallucination),
            "recall_hallucination": _f(self.recall_hallucination),
            "f1_hallucination": _f(self.f1_hallucination),
            "cohen_kappa": _f(self.cohen_kappa),
            "mcc": _f(self.mcc),
        }


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def print_metrics_table(metrics_list: list[LLMMetrics]) -> None:
    def _pct(v: float) -> str:
        return "N/A" if math.isnan(v) else f"{v:.1%}"

    def _flt(v: float, fmt: str = ".3f") -> str:
        return "N/A" if math.isnan(v) else format(v, fmt)

    def _cost(v: Optional[float]) -> str:
        if v is None:
            return "free"
        if math.isnan(v):
            return "N/A"
        return f"${v:.6f}"

    table = Table(title="Hallucination Evaluation Results", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Provider")
    table.add_column("Samples", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Hallucination Rate", justify="right", style="red")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("SUPPORTS Halluc.", justify="right")
    table.add_column("REFUTES Halluc.", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Kappa", justify="right")
    table.add_column("MCC", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Tokens/sample", justify="right")
    table.add_column("Cost/sample", justify="right")

    for m in metrics_list:
        avg_lat_s = m.avg_latency_s
        avg_lat_str = "N/A" if math.isnan(avg_lat_s) else f"{avg_lat_s:.2f}s"
        table.add_row(
            m.model_name,
            m.provider,
            str(m.n_total),
            str(m.n_errors),
            _pct(m.hallucination_rate),
            _pct(m.accuracy),
            _pct(m.supports_hallucination_rate),
            _pct(m.refutes_hallucination_rate),
            _pct(m.precision_hallucination),
            _pct(m.recall_hallucination),
            _pct(m.f1_hallucination),
            _flt(m.cohen_kappa),
            _flt(m.mcc),
            avg_lat_str,
            _flt(m.avg_tokens_per_sample, ".0f"),
            _cost(m.cost_per_sample_usd),
        )

    console.print(table)


def save_results(
    metrics_list: list[LLMMetrics],
    output_dir: str,
    save_responses: bool = True,
) -> None:
    """Save metrics and (optionally) raw sample results to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summaries = [m.summary_dict() for m in metrics_list]
    df = pd.DataFrame(summaries)
    summary_path = out / "summary.csv"
    df.to_csv(summary_path, index=False)
    console.print(f"Saved summary → [cyan]{summary_path}[/cyan]")

    # Per-model detail
    if save_responses:
        for m in metrics_list:
            safe_name = m.model_name.replace("/", "_").replace(":", "_")
            detail_path = out / f"{safe_name}_samples.jsonl"
            with open(detail_path, "w", encoding="utf-8") as f:
                for r in m.sample_results:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            console.print(f"Saved samples → [cyan]{detail_path}[/cyan]")

    # Plots
    _plot_hallucination_rates(metrics_list, out)
    _plot_nli_distribution(metrics_list, out)
    _plot_statistical_metrics(metrics_list, out)
    _plot_nli_score_distributions(metrics_list, out)
    _plot_latency_tokens(metrics_list, out)
    _plot_cost_efficiency(metrics_list, out)
    _plot_summary_heatmap(metrics_list, out)
    for m in metrics_list:
        _plot_confusion_matrix(m, out)
        _plot_error_categories(m, out)


# ------------------------------------------------------------------
# Existing plots (updated with style + value labels)
# ------------------------------------------------------------------

def _plot_hallucination_rates(metrics_list: list[LLMMetrics], out: Path) -> None:
    models = [m.model_name for m in metrics_list]
    overall  = [m.hallucination_rate          for m in metrics_list]
    supports = [m.supports_hallucination_rate  for m in metrics_list]
    refutes  = [m.refutes_hallucination_rate   for m in metrics_list]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    b1 = ax.bar(x - width, overall,  width, label="Overall",         color="steelblue")
    b2 = ax.bar(x,         supports, width, label="SUPPORTS claims",  color="seagreen")
    b3 = ax.bar(x + width, refutes,  width, label="REFUTES claims",   color="tomato")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if not math.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.1%}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out / "hallucination_rates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")


def _plot_nli_distribution(metrics_list: list[LLMMetrics], out: Path) -> None:
    models      = [m.model_name for m in metrics_list]
    entailment  = [m.nli_entailment_count  / max(m.n_total, 1) for m in metrics_list]
    neutral     = [m.nli_neutral_count     / max(m.n_total, 1) for m in metrics_list]
    contradiction = [m.nli_contradiction_count / max(m.n_total, 1) for m in metrics_list]

    df = pd.DataFrame(
        {"ENTAILMENT": entailment, "NEUTRAL": neutral, "CONTRADICTION": contradiction},
        index=models,
    )

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    df.plot(kind="bar", ax=ax, color=["seagreen", "gold", "tomato"])

    for container in ax.containers:
        labels = [
            f"{v:.1%}" if not math.isnan(float(v)) else ""
            for v in container.datavalues
        ]
        ax.bar_label(container, labels=labels, fontsize=7, padding=2)

    ax.set_ylabel("Fraction of Samples")
    ax.set_title("NLI Label Distribution per Model")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out / "nli_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")


# ------------------------------------------------------------------
# New plots
# ------------------------------------------------------------------

def _plot_statistical_metrics(metrics_list: list[LLMMetrics], out: Path) -> None:
    models = [m.model_name for m in metrics_list]
    prec  = [m.precision_hallucination for m in metrics_list]
    rec   = [m.recall_hallucination    for m in metrics_list]
    f1    = [m.f1_hallucination        for m in metrics_list]
    kappa = [m.cohen_kappa             for m in metrics_list]
    mcc_v = [m.mcc                     for m in metrics_list]

    x = np.arange(len(models))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(models) * 2.5), 5))

    # Left: Precision / Recall / F1
    b1 = ax1.bar(x - w, prec, w, label="Precision", color="#4C72B0")
    b2 = ax1.bar(x,     rec,  w, label="Recall",    color="#55A868")
    b3 = ax1.bar(x + w, f1,   w, label="F1",        color="#C44E52")
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if not math.isnan(h):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7,
                )
    ax1.set_ylabel("Score")
    ax1.set_title("Classification Metrics")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.set_ylim(0, 1.15)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: Cohen's Kappa / MCC
    b4 = ax2.bar(x - w / 2, kappa, w, label="Cohen's Kappa", color="#8172B2")
    b5 = ax2.bar(x + w / 2, mcc_v, w, label="MCC",           color="#CCB974")
    for bars in [b4, b5]:
        for bar in bars:
            h = bar.get_height()
            if not math.isnan(h):
                offset = 0.01 if h >= 0 else -0.03
                va = "bottom" if h >= 0 else "top"
                ax2.text(
                    bar.get_x() + bar.get_width() / 2, h + offset,
                    f"{h:.2f}", ha="center", va=va, fontsize=7,
                )
    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Score")
    ax2.set_title("Agreement Metrics")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out / "statistical_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")


def _plot_nli_score_distributions(metrics_list: list[LLMMetrics], out: Path) -> None:
    records = []
    for m in metrics_list:
        for r in m.sample_results:
            if r.llm_error or r.hallucination is None:
                continue
            records.append({
                "Model": m.model_name,
                "P(entailment)": r.nli_entailment,
                "P(neutral)": r.nli_neutral,
                "P(contradiction)": r.nli_contradiction,
                "Outcome": "Hallucination" if r.hallucination else "Correct",
            })

    if not records:
        return

    df = pd.DataFrame(records)
    palette = {"Correct": "#55A868", "Hallucination": "#C44E52"}
    score_cols = ["P(entailment)", "P(neutral)", "P(contradiction)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, score_cols):
        try:
            sns.boxplot(
                data=df, x="Model", y=col, hue="Outcome",
                ax=ax, palette=palette,
            )
        except Exception:
            ax.text(0.5, 0.5, "insufficient data",
                    transform=ax.transAxes, ha="center", va="center")
        ax.set_title(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("NLI Score Distributions by Model and Outcome")
    fig.tight_layout()
    path = out / "nli_score_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")


def _plot_confusion_matrix(m: LLMMetrics, out: Path) -> None:
    y_true, y_pred = m._binary_arrays()
    if not y_true:
        return

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm_norm[i, j] * 100:.1f}%\n(n={cm[i, j]})"

    kappa_s = "N/A" if math.isnan(m.cohen_kappa)        else f"{m.cohen_kappa:.3f}"
    mcc_s   = "N/A" if math.isnan(m.mcc)                else f"{m.mcc:.3f}"
    f1_s    = "N/A" if math.isnan(m.f1_hallucination)   else f"{m.f1_hallucination:.3f}"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues",
        xticklabels=["Correct", "Hallucination"],
        yticklabels=["Correct", "Hallucination"],
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{m.model_name}\nκ={kappa_s}  MCC={mcc_s}  F1={f1_s}")

    fig.tight_layout()
    safe = m.model_name.replace("/", "_").replace(":", "_")
    path = out / f"confusion_matrix_{safe}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")


def _plot_summary_heatmap(metrics_list: list[LLMMetrics], out: Path) -> None:
    if len(metrics_list) < 2:
        return

    cols = [
        "Halluc.Rate", "Accuracy", "Precision", "Recall", "F1",
        "Kappa", "MCC", "SUPPORTS Halluc.", "REFUTES Halluc.",
    ]
    rows = []
    for m in metrics_list:
        rows.append([
            m.hallucination_rate,
            m.accuracy,
            m.precision_hallucination,
            m.recall_hallucination,
            m.f1_hallucination,
            m.cohen_kappa,
            m.mcc,
            m.supports_hallucination_rate,
            m.refutes_hallucination_rate,
        ])

    models = [m.model_name for m in metrics_list]
    df = pd.DataFrame(rows, index=models, columns=cols)
    df = df.applymap(
        lambda v: np.nan if (isinstance(v, float) and math.isnan(v)) else v
    )
    annot = df.applymap(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")

    fig, ax = plt.subplots(
        figsize=(max(10, len(cols) * 1.2), max(4, len(models) * 0.8 + 2))
    )
    sns.heatmap(
        df, annot=annot, fmt="", cmap="RdYlGn_r",
        ax=ax, vmin=0, vmax=1, mask=df.isna(),
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Model Comparison Heatmap")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    path = out / "summary_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")


def _plot_latency_tokens(metrics_list: list[LLMMetrics], out: Path) -> None:
    models = [m.model_name for m in metrics_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(models) * 2.5), 5))

    # Left: per-sample latency box plot with mean diamond
    latency_data = [m._latency_list for m in metrics_list]
    bp = ax1.boxplot(
        latency_data, labels=models, patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.7)
    for i, lats in enumerate(latency_data):
        if lats:
            ax1.plot(i + 1, float(np.mean(lats)), marker="D", color="red",
                     markersize=6, zorder=5, label="mean" if i == 0 else "")
    ax1.set_ylabel("Latency (s)")
    ax1.set_title("Per-sample Latency Distribution")
    ax1.set_xticklabels(models, rotation=30, ha="right")
    if any(latency_data):
        ax1.legend(["mean"], loc="upper right", markerscale=0.8)
    ax1.grid(axis="y", alpha=0.3)

    # Right: stacked bar — avg input + output tokens per sample
    x = np.arange(len(models))
    avg_in  = [m.total_input_tokens  / max(m.n_total, 1) for m in metrics_list]
    avg_out = [m.total_output_tokens / max(m.n_total, 1) for m in metrics_list]
    ax2.bar(x, avg_in,  label="Avg input tokens",  color="#4C72B0")
    ax2.bar(x, avg_out, bottom=avg_in, label="Avg output tokens", color="#C44E52")
    all_totals = [vi + vo for vi, vo in zip(avg_in, avg_out)]
    max_total = max(all_totals) if all_totals else 1.0
    for i, (vi, vo) in enumerate(zip(avg_in, avg_out)):
        total = vi + vo
        if not math.isnan(total):
            ax2.text(i, total + max_total * 0.01,
                     f"{total:.0f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Tokens")
    ax2.set_title("Avg Tokens per Sample")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out / "latency_tokens.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")


def _plot_cost_efficiency(metrics_list: list[LLMMetrics], out: Path) -> None:
    models = [m.model_name for m in metrics_list]
    costs_per_1k: list[float] = []
    is_local: list[bool] = []
    for m in metrics_list:
        cps = m.cost_per_sample_usd
        if cps is None:
            costs_per_1k.append(0.0)
            is_local.append(True)
        else:
            costs_per_1k.append(cps * 1000)
            is_local.append(False)

    halluc_rates = [m.hallucination_rate for m in metrics_list]
    max_cost = max(costs_per_1k) if costs_per_1k else 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(models) * 2.5), 5))

    # Left: cost per 1 000 samples
    x = np.arange(len(models))
    colors = ["#AAAAAA" if local else "#4C72B0" for local in is_local]
    bars = ax1.bar(x, costs_per_1k, color=colors)
    for bar, val, local in zip(bars, costs_per_1k, is_local):
        label = "free" if local else f"${val:.3f}"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_cost * 0.02,
            label, ha="center", va="bottom", fontsize=8,
        )
    ax1.set_ylabel("Cost (USD)")
    ax1.set_title("Cost per 1 000 Samples")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    # Right: scatter hallucination_rate vs cost per 1 000
    for m, cost, local, hr in zip(metrics_list, costs_per_1k, is_local, halluc_rates):
        if math.isnan(hr):
            continue
        x_val = 0.0 if local else cost
        ax2.scatter(x_val, hr, marker="^" if local else "o", s=100, zorder=3)
        ax2.annotate(m.model_name, (x_val, hr),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax2.set_xlabel("Cost per 1 000 samples (USD)")
    ax2.set_ylabel("Hallucination Rate")
    ax2.set_title("Hallucination Rate vs Cost")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = out / "cost_efficiency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")


def _plot_error_categories(m: LLMMetrics, out: Path) -> None:
    categories = [
        "SUPPORTS→ENTAILMENT",
        "SUPPORTS→NEUTRAL",
        "SUPPORTS→CONTRADICTION",
        "REFUTES→ENTAILMENT",
        "REFUTES→NEUTRAL",
        "REFUTES→CONTRADICTION",
    ]
    color_map = {
        "SUPPORTS→ENTAILMENT":    "#55A868",  # correct (green)
        "SUPPORTS→NEUTRAL":       "#FFA500",  # ambiguous (amber)
        "SUPPORTS→CONTRADICTION": "#C44E52",  # hallucination (red)
        "REFUTES→ENTAILMENT":     "#C44E52",  # hallucination (red)
        "REFUTES→NEUTRAL":        "#FFA500",  # ambiguous (amber)
        "REFUTES→CONTRADICTION":  "#55A868",  # correct (green)
    }

    counts = m.error_category_counts()
    values = [counts.get(cat, 0) for cat in categories]
    colors = [color_map[cat] for cat in categories]
    max_val = max(values) if any(v > 0 for v in values) else 1

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    bars = ax.bar(x, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + max_val * 0.01,
            str(val), ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Count")
    ax.set_title(f"Error Category Breakdown — {m.model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.legend(handles=[
        Patch(facecolor="#55A868", label="Correct"),
        Patch(facecolor="#FFA500", label="Ambiguous"),
        Patch(facecolor="#C44E52", label="Hallucination"),
    ])
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    safe = m.model_name.replace("/", "_").replace(":", "_")
    path = out / f"error_categories_{safe}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")
