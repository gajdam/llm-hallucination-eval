"""Main evaluation pipeline.

Flow per LLM:
  1. Load FEVER samples.
  2. For each sample, ask the LLM to comment on the claim.
  3. Run NLI(claim, llm_response) to measure semantic alignment.
  4. Classify each response as hallucination / correct / ambiguous.
  5. Aggregate metrics and save results.
"""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from dotenv import load_dotenv

from .data.fever_loader import FeverSample, load_fever_samples
from .evaluation.metrics import LLMMetrics, SampleResult, print_metrics_table, save_results
from .llm.base import BaseLLM
from .llm.registry import build_llms
from .nli.nli_scorer import NLIScorer, is_hallucination

load_dotenv()
console = Console()


class EvaluationPipeline:
    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, llm_filter: Optional[str] = None) -> list[LLMMetrics]:
        cfg = self.config

        # 1. Load FEVER samples
        filter_cfg = cfg.get("filtering", {})
        filtering_on = filter_cfg.get("enabled", False)
        samples = load_fever_samples(
            split=cfg["fever"]["split"],
            max_samples=cfg["fever"].get("max_samples"),
            labels=cfg["fever"]["labels"],
            seed=cfg["fever"].get("seed", 42),
            min_words=filter_cfg.get("min_words", 0) if filtering_on else 0,
            filter_vague_predicates=filter_cfg.get("filter_vague_predicates", False) and filtering_on,
        )

        # 2. Build LLM instances
        console.print("\n[bold]Initialising LLMs...[/bold]")
        llms = build_llms(cfg["llms"], llm_filter=llm_filter)
        if not llms:
            console.print("[red]No LLMs available. Check your config and API keys.[/red]")
            return []

        # 3. Load NLI model (once, shared across all LLMs)
        console.print("\n[bold]Loading NLI model...[/bold]")
        nli_cfg = cfg.get("nli", {})
        nli_scorer = NLIScorer(
            model_name=nli_cfg.get("model", "cross-encoder/nli-deberta-v3-large"),
            device=nli_cfg.get("device", "auto"),
            max_length=nli_cfg.get("max_length", 512),
            batch_size=nli_cfg.get("batch_size", 16),
        )

        # 4. Evaluate each LLM
        all_metrics: list[LLMMetrics] = []
        for llm in llms:
            console.print(f"\n[bold cyan]Evaluating: {llm.provider}/{llm.name}[/bold cyan]")
            metrics = self._evaluate_llm(llm, samples, nli_scorer)
            all_metrics.append(metrics)

        # 5. Print comparison table and save
        console.print("\n")
        print_metrics_table(all_metrics)

        eval_cfg = cfg.get("evaluation", {})
        save_results(
            all_metrics,
            output_dir=eval_cfg.get("output_dir", "results"),
            save_responses=eval_cfg.get("save_responses", True),
        )

        return all_metrics

    # ------------------------------------------------------------------
    # Per-LLM evaluation
    # ------------------------------------------------------------------

    def _evaluate_llm(
        self,
        llm: BaseLLM,
        samples: list[FeverSample],
        nli_scorer: NLIScorer,
    ) -> LLMMetrics:
        prompts_cfg = self.config.get("prompts", {})
        system_prompt: Optional[str] = prompts_cfg.get("system")
        user_template: str = prompts_cfg.get(
            "user_template",
            "Please provide accurate factual information about the following statement. "
            "Be specific and concise (2-4 sentences).\n\nStatement: {claim}",
        )
        request_delay = self.config.get("evaluation", {}).get("request_delay", 0.5)

        metrics = LLMMetrics(model_name=llm.name, provider=llm.provider)

        # --- Step A: generate LLM responses ---
        responses: list[tuple[FeverSample, str, Optional[str], float, dict]] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating responses", total=len(samples))
            for sample in samples:
                prompt = user_template.format(claim=sample.claim)
                t0 = time.perf_counter()
                response = llm.generate(prompt, system=system_prompt)
                latency = time.perf_counter() - t0
                responses.append((sample, response.text, response.error, latency, response.usage))
                progress.advance(task)
                if not response.failed:
                    time.sleep(request_delay)

        # --- Step B: run NLI in batch ---
        # premise=llm_response, hypothesis=claim: asks "does the response imply the claim?"
        valid_pairs: list[tuple[int, str, str]] = []  # (index, premise, hypothesis)
        for i, (sample, text, error, latency, usage) in enumerate(responses):
            if not error and text.strip():
                valid_pairs.append((i, text, sample.claim))

        nli_results_map: dict[int, object] = {}
        if valid_pairs:
            indices, premises, hypotheses = zip(*valid_pairs)
            console.print(f"  Running NLI on {len(valid_pairs)} responses...")
            nli_batch = nli_scorer.predict_batch(list(zip(premises, hypotheses)))
            for idx, nli_result in zip(indices, nli_batch):
                nli_results_map[idx] = nli_result

        # --- Step C: compute per-sample hallucination ---
        for i, (sample, text, error, latency, usage) in enumerate(responses):
            nli_result = nli_results_map.get(i)
            input_tokens = usage.get("input_tokens", 0) if usage else 0
            output_tokens = usage.get("output_tokens", 0) if usage else 0
            if nli_result is None:
                # API error or empty response
                sr = SampleResult(
                    sample_id=sample.id,
                    claim=sample.claim,
                    fever_label=sample.label,
                    llm_response=text,
                    llm_error=error or "empty response",
                    nli_label="",
                    nli_entailment=0.0,
                    nli_neutral=0.0,
                    nli_contradiction=0.0,
                    hallucination=None,
                    latency_s=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            else:
                hallucination = is_hallucination(sample.label, nli_result)
                sr = SampleResult(
                    sample_id=sample.id,
                    claim=sample.claim,
                    fever_label=sample.label,
                    llm_response=text,
                    llm_error=None,
                    nli_label=nli_result.label,
                    nli_entailment=nli_result.entailment,
                    nli_neutral=nli_result.neutral,
                    nli_contradiction=nli_result.contradiction,
                    hallucination=hallucination,
                    latency_s=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            metrics.add_result(sr)

        console.print(
            f"  Hallucination rate: [red]{metrics.hallucination_rate:.1%}[/red]  "
            f"Accuracy: [green]{metrics.accuracy:.1%}[/green]  "
            f"Errors: {metrics.n_errors}"
        )
        return metrics
