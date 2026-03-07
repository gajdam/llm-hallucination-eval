"""Evaluation metrics and report generation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table

console = Console()


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

    def add_result(self, result: SampleResult) -> None:
        self.sample_results.append(result)
        self.n_total += 1

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

    def summary_dict(self) -> dict:
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
        }


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def print_metrics_table(metrics_list: list[LLMMetrics]) -> None:
    table = Table(title="Hallucination Evaluation Results", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Provider")
    table.add_column("Samples", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Hallucination Rate", justify="right", style="red")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("SUPPORTS Halluc.", justify="right")
    table.add_column("REFUTES Halluc.", justify="right")

    for m in metrics_list:
        table.add_row(
            m.model_name,
            m.provider,
            str(m.n_total),
            str(m.n_errors),
            f"{m.hallucination_rate:.1%}",
            f"{m.accuracy:.1%}",
            f"{m.supports_hallucination_rate:.1%}",
            f"{m.refutes_hallucination_rate:.1%}",
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

    # Comparison plots
    _plot_hallucination_rates(metrics_list, out)
    _plot_nli_distribution(metrics_list, out)


def _plot_hallucination_rates(metrics_list: list[LLMMetrics], out: Path) -> None:
    models = [m.model_name for m in metrics_list]
    overall = [m.hallucination_rate for m in metrics_list]
    supports = [m.supports_hallucination_rate for m in metrics_list]
    refutes = [m.refutes_hallucination_rate for m in metrics_list]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    ax.bar(x - width, overall, width, label="Overall", color="steelblue")
    ax.bar(x, supports, width, label="SUPPORTS claims", color="seagreen")
    ax.bar(x + width, refutes, width, label="REFUTES claims", color="tomato")

    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out / "hallucination_rates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")


def _plot_nli_distribution(metrics_list: list[LLMMetrics], out: Path) -> None:
    models = [m.model_name for m in metrics_list]
    entailment = [
        m.nli_entailment_count / max(m.n_total, 1) for m in metrics_list
    ]
    neutral = [
        m.nli_neutral_count / max(m.n_total, 1) for m in metrics_list
    ]
    contradiction = [
        m.nli_contradiction_count / max(m.n_total, 1) for m in metrics_list
    ]

    df = pd.DataFrame(
        {"ENTAILMENT": entailment, "NEUTRAL": neutral, "CONTRADICTION": contradiction},
        index=models,
    )

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    df.plot(kind="bar", ax=ax, color=["seagreen", "gold", "tomato"])
    ax.set_ylabel("Fraction of Samples")
    ax.set_title("NLI Label Distribution per Model")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out / "nli_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    console.print(f"Saved plot   → [cyan]{path}[/cyan]")
