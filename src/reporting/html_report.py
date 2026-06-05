"""Self-contained HTML report using Plotly + Jinja2."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _nan(v) -> Optional[float]:
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _fmt_pct(v) -> str:
    v = _nan(v)
    if v is None:
        return "N/A"
    return f"{v:.1%}"


def _fmt_f(v, fmt: str = ".3f") -> str:
    v = _nan(v)
    if v is None:
        return "N/A"
    return format(v, fmt)


def _cell_color(rate: Optional[float], reverse: bool = False) -> str:
    """HSL background for metric cells: low=green, high=red (or reversed)."""
    if rate is None:
        return "background-color: #f5f5f5;"
    rate = max(0.0, min(1.0, rate))
    if reverse:
        rate = 1 - rate
    hue = 120 * (1 - rate)
    return f"background-color: hsla({hue:.0f}, 65%, 55%, 0.22);"


# ------------------------------------------------------------------
# Plotly chart builders — each returns an HTML fragment (div + script)
# ------------------------------------------------------------------

def _chart_html(fig, div_id: str, height: str = "420px") -> str:
    import plotly.io as pio
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        default_height=height,
        default_width="100%",
    )


def _base_layout(**extra) -> dict:
    return dict(
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="white",
        margin=dict(t=70, b=60, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **extra,
    )


def _make_hallucination_bar(metrics_list) -> str:
    import plotly.graph_objects as go

    models = [m.model_name for m in metrics_list]
    series = [
        ("Overall",        "hallucination_rate",         "#4C72B0"),
        ("SUPPORTS claims","supports_hallucination_rate", "#55A868"),
        ("REFUTES claims", "refutes_hallucination_rate",  "#C44E52"),
    ]
    fig = go.Figure()
    for name, attr, color in series:
        vals = [_nan(getattr(m, attr)) for m in metrics_list]
        fig.add_trace(go.Bar(
            name=name, x=models, y=vals,
            text=[_fmt_pct(v) for v in vals],
            textposition="outside",
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group",
        title="Hallucination Rate by Model & Claim Type",
        yaxis=dict(tickformat=".0%", range=[0, 1.3], title="Hallucination Rate"),
        **_base_layout(),
    )
    return _chart_html(fig, "chart-hallucination")


def _make_nli_distribution(metrics_list) -> str:
    import plotly.graph_objects as go

    models = [m.model_name for m in metrics_list]
    series = [
        ("ENTAILMENT",    "nli_entailment_count",    "#55A868"),
        ("NEUTRAL",       "nli_neutral_count",        "#FFC107"),
        ("CONTRADICTION", "nli_contradiction_count",  "#C44E52"),
    ]
    fig = go.Figure()
    for name, attr, color in series:
        vals = [getattr(m, attr) / max(m.n_total, 1) for m in metrics_list]
        fig.add_trace(go.Bar(
            name=name, x=models, y=vals,
            text=[f"{v:.1%}" for v in vals],
            textposition="outside",
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group",
        title="NLI Label Distribution by Model",
        yaxis=dict(tickformat=".0%", range=[0, 1.3], title="Fraction of Samples"),
        **_base_layout(),
    )
    return _chart_html(fig, "chart-nli")


def _make_classification_metrics(metrics_list) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    models = [m.model_name for m in metrics_list]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Precision / Recall / F1", "Agreement Metrics (Kappa & MCC)"],
    )

    prf_series = [
        ("Precision", "precision_hallucination", "#4C72B0"),
        ("Recall",    "recall_hallucination",    "#55A868"),
        ("F1",        "f1_hallucination",         "#C44E52"),
    ]
    for name, attr, color in prf_series:
        vals = [_nan(getattr(m, attr)) for m in metrics_list]
        fig.add_trace(go.Bar(
            name=name, x=models, y=vals,
            text=[_fmt_f(v, ".2f") for v in vals],
            textposition="outside",
            marker_color=color,
        ), row=1, col=1)

    agree_series = [
        ("Cohen κ", "cohen_kappa", "#8172B2"),
        ("MCC",     "mcc",         "#CCB974"),
    ]
    for name, attr, color in agree_series:
        vals = [_nan(getattr(m, attr)) for m in metrics_list]
        fig.add_trace(go.Bar(
            name=name, x=models, y=vals,
            text=[_fmt_f(v, ".2f") for v in vals],
            textposition="outside",
            marker_color=color,
        ), row=1, col=2)

    fig.update_layout(barmode="group", **_base_layout())
    fig.update_yaxes(range=[0, 1.3], row=1, col=1)
    fig.update_yaxes(range=[-0.6, 1.3], row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="grey", row=1, col=2)
    return _chart_html(fig, "chart-classification")


def _make_latency_box(metrics_list) -> str:
    import plotly.graph_objects as go

    fig = go.Figure()
    for m in metrics_list:
        if m._latency_list:
            fig.add_trace(go.Box(
                y=m._latency_list,
                name=m.model_name,
                boxmean="sd",
                marker_color="#4C72B0",
                line_color="#2d5a8e",
            ))
    fig.update_layout(
        title="Response Latency Distribution",
        yaxis=dict(title="Latency (s)"),
        showlegend=False,
        **_base_layout(),
    )
    return _chart_html(fig, "chart-latency")


def _make_tokens_bar(metrics_list) -> str:
    import plotly.graph_objects as go

    models = [m.model_name for m in metrics_list]
    avg_in  = [m.total_input_tokens  / max(m.n_total, 1) for m in metrics_list]
    avg_out = [m.total_output_tokens / max(m.n_total, 1) for m in metrics_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Input tokens",  x=models, y=avg_in,  marker_color="#4C72B0"))
    fig.add_trace(go.Bar(name="Output tokens", x=models, y=avg_out, marker_color="#C44E52"))
    fig.update_layout(
        barmode="stack",
        title="Avg Token Usage per Sample",
        yaxis=dict(title="Tokens"),
        **_base_layout(),
    )
    return _chart_html(fig, "chart-tokens")


def _make_cost_charts(metrics_list) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    models = [m.model_name for m in metrics_list]
    costs_per_1k, is_local = [], []
    for m in metrics_list:
        cps = m.cost_per_sample_usd
        if cps is None:
            costs_per_1k.append(0.0)
            is_local.append(True)
        else:
            costs_per_1k.append(cps * 1000)
            is_local.append(False)

    colors = ["#BBBBBB" if loc else "#4C72B0" for loc in is_local]
    labels = ["free" if loc else f"${v:.3f}" for v, loc in zip(costs_per_1k, is_local)]
    halluc = [_nan(m.hallucination_rate) for m in metrics_list]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Cost per 1 000 Samples (USD)", "Hallucination Rate vs Cost"],
    )
    fig.add_trace(go.Bar(
        x=models, y=costs_per_1k, text=labels,
        textposition="outside", marker_color=colors, showlegend=False,
    ), row=1, col=1)

    for m, cost, loc, hr in zip(metrics_list, costs_per_1k, is_local, halluc):
        if hr is None:
            continue
        fig.add_trace(go.Scatter(
            x=[0.0 if loc else cost], y=[hr],
            mode="markers+text",
            text=[m.model_name], textposition="top center",
            marker=dict(size=14, symbol="triangle-up" if loc else "circle",
                        color="#BBBBBB" if loc else "#4C72B0"),
            showlegend=False, name=m.model_name,
        ), row=1, col=2)

    fig.update_layout(**_base_layout())
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Hallucination Rate", tickformat=".0%", row=1, col=2)
    fig.update_xaxes(title_text="Cost per 1 000 samples (USD)", row=1, col=2)
    return _chart_html(fig, "chart-cost")


def _make_summary_heatmap(metrics_list) -> Optional[str]:
    if len(metrics_list) < 2:
        return None
    import plotly.graph_objects as go

    cols = [
        ("Halluc. Rate",     "hallucination_rate"),
        ("Accuracy",         "accuracy"),
        ("Precision",        "precision_hallucination"),
        ("Recall",           "recall_hallucination"),
        ("F1",               "f1_hallucination"),
        ("Kappa",            "cohen_kappa"),
        ("MCC",              "mcc"),
        ("SUPPORTS Halluc.", "supports_hallucination_rate"),
        ("REFUTES Halluc.",  "refutes_hallucination_rate"),
    ]
    col_labels = [c[0] for c in cols]
    models = [m.model_name for m in metrics_list]

    z, text = [], []
    for m in metrics_list:
        row, trow = [], []
        for _, attr in cols:
            v = _nan(getattr(m, attr))
            row.append(v)
            trow.append(_fmt_f(v, ".2f") if v is not None else "N/A")
        z.append(row)
        text.append(trow)

    height = max(320, len(models) * 55 + 180)
    fig = go.Figure(go.Heatmap(
        z=z, x=col_labels, y=models,
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn_r", zmin=0, zmax=1,
        colorbar=dict(title="Score", len=0.8),
    ))
    fig.update_layout(
        title="Model Comparison Heatmap",
        xaxis=dict(side="top"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=110, b=20, l=150, r=20),
        height=height,
    )
    return _chart_html(fig, "chart-heatmap", height=f"{height}px")


def _make_confusion_matrix(m, idx: int) -> Optional[str]:
    y_true, y_pred = m._binary_arrays()
    if len(y_true) < 2:
        return None
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    text = [
        [f"{cm_norm[i,j]*100:.1f}%<br>(n={cm[i,j]})" for j in range(2)]
        for i in range(2)
    ]
    kappa_s = _fmt_f(_nan(m.cohen_kappa))
    mcc_s   = _fmt_f(_nan(m.mcc))
    f1_s    = _fmt_f(_nan(m.f1_hallucination))

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=["Correct (pred)", "Hallucination (pred)"],
        y=["Correct (true)", "Hallucination (true)"],
        text=text, texttemplate="%{text}",
        colorscale="Blues", zmin=0, zmax=1, showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix<br><sup>κ={kappa_s} · MCC={mcc_s} · F1={f1_s}</sup>",
        xaxis=dict(side="bottom"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=80, b=60, l=130, r=20),
        height=350,
    )
    return _chart_html(fig, f"chart-cm-{idx}", height="350px")


def _make_error_categories(m, idx: int) -> Optional[str]:
    import plotly.graph_objects as go

    categories = [
        ("SUPPORTS→ENTAILMENT",    "#55A868"),
        ("SUPPORTS→NEUTRAL",       "#FFC107"),
        ("SUPPORTS→CONTRADICTION", "#C44E52"),
        ("REFUTES→ENTAILMENT",     "#C44E52"),
        ("REFUTES→NEUTRAL",        "#FFC107"),
        ("REFUTES→CONTRADICTION",  "#55A868"),
    ]
    counts = m.error_category_counts()
    labels = [c[0] for c in categories]
    values = [counts.get(c[0], 0) for c in categories]
    colors = [c[1] for c in categories]

    if not any(values):
        return None

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=values, textposition="outside",
    ))
    fig.update_layout(
        title="FEVER×NLI Category Breakdown",
        yaxis=dict(title="Count"),
        xaxis=dict(tickangle=-25),
        plot_bgcolor="#F8F9FA", paper_bgcolor="white",
        margin=dict(t=60, b=100, l=50, r=20),
        showlegend=False,
        height=350,
    )
    return _chart_html(fig, f"chart-ec-{idx}", height="350px")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def generate_html_report(
    metrics_list: list,
    output_dir: str,
    config: Optional[dict] = None,
) -> Optional[Path]:
    """Render a self-contained HTML report and save it to output_dir/report.html."""
    try:
        import plotly  # noqa: F401
        from jinja2 import Template
    except ImportError as exc:
        console.print(f"[yellow]HTML report skipped (missing dependency: {exc})[/yellow]")
        return None

    if not metrics_list:
        return None

    config = config or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Metadata ---
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    fever_cfg = config.get("fever", {})
    nli_cfg = config.get("nli", {})
    n_samples = metrics_list[0].n_total if metrics_list else 0

    valid_rates = [(m.hallucination_rate, m) for m in metrics_list
                   if not math.isnan(m.hallucination_rate)]
    best_m  = min(valid_rates, key=lambda x: x[0])[1] if valid_rates else None
    worst_m = max(valid_rates, key=lambda x: x[0])[1] if valid_rates else None

    # --- Summary table rows ---
    def _cost_1k(m) -> str:
        cps = m.cost_per_sample_usd
        return "free" if cps is None else f"${cps * 1000:.3f}"

    def _lat(v) -> str:
        v = _nan(v)
        return "N/A" if v is None else f"{v:.2f}s"

    table_rows = []
    for m in metrics_list:
        table_rows.append({
            "model":          m.model_name,
            "provider":       m.provider,
            "n_total":        m.n_total,
            "n_errors":       m.n_errors,
            "halluc_rate":    _fmt_pct(_nan(m.hallucination_rate)),
            "halluc_color":   _cell_color(_nan(m.hallucination_rate)),
            "accuracy":       _fmt_pct(_nan(m.accuracy)),
            "accuracy_color": _cell_color(_nan(m.accuracy), reverse=True),
            "supports_rate":  _fmt_pct(_nan(m.supports_hallucination_rate)),
            "supports_color": _cell_color(_nan(m.supports_hallucination_rate)),
            "refutes_rate":   _fmt_pct(_nan(m.refutes_hallucination_rate)),
            "refutes_color":  _cell_color(_nan(m.refutes_hallucination_rate)),
            "f1":             _fmt_f(_nan(m.f1_hallucination)),
            "kappa":          _fmt_f(_nan(m.cohen_kappa)),
            "avg_latency":    _lat(m.avg_latency_s),
            "cost_per_1k":    _cost_1k(m),
        })

    # --- Per-model detail blocks ---
    per_model_details = []
    for i, m in enumerate(metrics_list):
        total_cost = (
            f"${m.total_cost_usd:.4f}"
            if m.total_cost_usd is not None
            else "free (local)"
        )
        per_model_details.append({
            "name":            m.model_name,
            "provider":        m.provider,
            "halluc_rate":     _fmt_pct(_nan(m.hallucination_rate)),
            "f1":              _fmt_f(_nan(m.f1_hallucination)),
            "n_total":         m.n_total,
            "n_evaluated":     m.n_evaluated,
            "n_errors":        m.n_errors,
            "n_neutral":       m.n_neutral,
            "total_tokens":    f"{m.total_tokens:,}",
            "avg_latency":     _lat(m.avg_latency_s),
            "median_latency":  _lat(m.median_latency_s),
            "p95_latency":     _lat(m.p95_latency_s),
            "total_cost":      total_cost,
            "confusion_chart": _make_confusion_matrix(m, i),
            "error_chart":     _make_error_categories(m, i),
        })

    # --- Charts ---
    heatmap = _make_summary_heatmap(metrics_list)
    ctx = dict(
        generated_at=generated_at,
        dataset="FEVER",
        split=fever_cfg.get("split", "labelled_dev"),
        nli_model=nli_cfg.get("model", "—"),
        n_models=len(metrics_list),
        n_samples=n_samples,
        best_model=best_m.model_name if best_m else "N/A",
        best_rate=_fmt_pct(best_m.hallucination_rate) if best_m else "N/A",
        worst_model=worst_m.model_name if worst_m else "N/A",
        worst_rate=_fmt_pct(worst_m.hallucination_rate) if worst_m else "N/A",
        table_rows=table_rows,
        hallucination_chart=_make_hallucination_bar(metrics_list),
        nli_chart=_make_nli_distribution(metrics_list),
        classification_chart=_make_classification_metrics(metrics_list),
        latency_chart=_make_latency_box(metrics_list),
        tokens_chart=_make_tokens_bar(metrics_list),
        cost_chart=_make_cost_charts(metrics_list),
        heatmap_chart=heatmap,
        per_model_details=per_model_details,
    )

    template_path = Path(__file__).parent / "template.html"
    template_src = template_path.read_text(encoding="utf-8")
    html = Template(template_src).render(**ctx)

    path = out / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
