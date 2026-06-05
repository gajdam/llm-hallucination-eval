"""Main evaluation pipeline.

Flow:
  1. Load FEVER samples.
  2. Build LLM instances.
  3. Load NLI model.
  4. Pre-fetch KG/IR contexts for all samples (if kg.enabled in config).
  5. For each LLM × track: ask the LLM, run NLI, aggregate metrics.
  6. Save results.

Tracks:
  blind   — LLM receives only the claim (baseline).
  kg      — LLM receives claim + Wikidata entity descriptions.
  hybrid  — LLM receives claim + Wikidata descriptions + Wikipedia summaries.
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

# Prompt used when the LLM is given background context (kg / hybrid tracks)
_GROUNDED_USER_TEMPLATE = (
    "Use the following background information to verify the statement.\n\n"
    "Background:\n{context}\n\n"
    "Is the following statement TRUE or FALSE? "
    'Start your response with "TRUE:" or "FALSE:", then give a one-sentence explanation.\n\n'
    "Statement: {claim}"
)


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

        # 3. Load NLI model (once, shared across all LLMs and tracks)
        console.print("\n[bold]Loading NLI model...[/bold]")
        nli_cfg = cfg.get("nli", {})
        nli_scorer = NLIScorer(
            model_name=nli_cfg.get("model", "cross-encoder/nli-deberta-v3-large"),
            device=nli_cfg.get("device", "auto"),
            max_length=nli_cfg.get("max_length", 512),
            batch_size=nli_cfg.get("batch_size", 16),
        )

        # 4. Pre-fetch KG/IR contexts (once for all LLMs)
        kg_cfg = cfg.get("kg", {})
        tracks: list[str] = ["blind"]
        contexts: dict[int, dict[str, str]] = {}

        if kg_cfg.get("enabled", False):
            tracks = kg_cfg.get("tracks", ["blind", "kg", "hybrid"])
            if any(t != "blind" for t in tracks):
                contexts = self._prefetch_contexts(samples, kg_cfg)

        # 5. Evaluate each LLM × each track
        all_metrics: list[LLMMetrics] = []
        for llm in llms:
            for track in tracks:
                console.print(
                    f"\n[bold cyan]Evaluating: {llm.provider}/{llm.name}  "
                    f"[yellow]track={track}[/yellow][/bold cyan]"
                )
                metrics = self._evaluate_llm(llm, samples, nli_scorer, track, contexts)
                all_metrics.append(metrics)

        # 6. Print comparison table and save
        console.print("\n")
        print_metrics_table(all_metrics)

        eval_cfg = cfg.get("evaluation", {})
        save_results(
            all_metrics,
            output_dir=eval_cfg.get("output_dir", "results"),
            save_responses=eval_cfg.get("save_responses", True),
            config=self.config,
        )

        return all_metrics

    # ------------------------------------------------------------------
    # Context pre-fetching
    # ------------------------------------------------------------------

    def _prefetch_contexts(
        self,
        samples: list[FeverSample],
        kg_cfg: dict,
    ) -> dict[int, dict[str, str]]:
        """Fetch Wikidata/Wikipedia contexts for all samples (with entity cache)."""
        from .kg.kg_retriever import KGRetriever

        retriever = KGRetriever(
            max_entities=kg_cfg.get("max_entities", 3),
            wikidata_timeout=kg_cfg.get("wikidata_timeout", 5),
            wikipedia_timeout=kg_cfg.get("wikipedia_timeout", 5),
        )

        console.print("\n[bold]Fetching KG/IR contexts...[/bold]")
        contexts: dict[int, dict[str, str]] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching contexts", total=len(samples))
            for sample in samples:
                kg = retriever.get_kg_context(sample.claim)
                ir = retriever.get_ir_context(sample.claim)
                hybrid = retriever.get_hybrid_context(kg, ir)
                contexts[sample.id] = {"blind": "", "kg": kg, "hybrid": hybrid}
                progress.advance(task)

        covered = sum(1 for c in contexts.values() if c["kg"].strip())
        console.print(
            f"  KG coverage: [green]{covered}/{len(samples)}[/green] claims have context"
        )
        return contexts

    # ------------------------------------------------------------------
    # Per-LLM × track evaluation
    # ------------------------------------------------------------------

    def _evaluate_llm(
        self,
        llm: BaseLLM,
        samples: list[FeverSample],
        nli_scorer: NLIScorer,
        track: str = "blind",
        contexts: Optional[dict[int, dict[str, str]]] = None,
    ) -> LLMMetrics:
        prompts_cfg = self.config.get("prompts", {})
        system_prompt: Optional[str] = prompts_cfg.get("system")
        blind_user_template: str = prompts_cfg.get(
            "user_template",
            "Please provide accurate factual information about the following statement. "
            "Be specific and concise (2-4 sentences).\n\nStatement: {claim}",
        )
        request_delay = self.config.get("evaluation", {}).get("request_delay", 0.5)

        metrics = LLMMetrics(model_name=llm.name, provider=llm.provider, track=track)

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
            task = progress.add_task(f"[{track}] Generating responses", total=len(samples))
            for sample in samples:
                context = (contexts or {}).get(sample.id, {}).get(track, "")
                prompt = self._build_prompt(sample.claim, track, context, blind_user_template)
                t0 = time.perf_counter()
                response = llm.generate(prompt, system=system_prompt)
                latency = time.perf_counter() - t0
                responses.append((sample, response.text, response.error, latency, response.usage))
                progress.advance(task)
                if not response.failed:
                    time.sleep(request_delay)

        # --- Step B: run NLI in batch ---
        valid_pairs: list[tuple[int, str, str]] = []
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
                    track=track,
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
                    track=track,
                )
            metrics.add_result(sr)

        console.print(
            f"  [{track}] Hallucination rate: [red]{metrics.hallucination_rate:.1%}[/red]  "
            f"Accuracy: [green]{metrics.accuracy:.1%}[/green]  "
            f"Errors: {metrics.n_errors}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(claim: str, track: str, context: str, blind_template: str) -> str:
        """Return the user prompt for the given track and context."""
        if track == "blind" or not context.strip():
            return blind_template.format(claim=claim)
        return _GROUNDED_USER_TEMPLATE.format(context=context.strip(), claim=claim)
