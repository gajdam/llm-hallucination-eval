"""Load and preprocess the FEVER dataset from HuggingFace."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from rich.console import Console

console = Console()

# Matches "[Subject] is/are a/an [1-3 words]." — too vague for NLI
_VAGUE_CLAIM = re.compile(r"^.+\s+(?:is|are)\s+an?\s+\w+(?:\s+\w+){0,2}\.?$", re.IGNORECASE)


@dataclass
class FeverSample:
    id: int
    claim: str
    label: str  # "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"


def load_fever_samples(
    split: str = "labelled_dev",
    max_samples: Optional[int] = None,
    labels: Optional[list[str]] = None,
    seed: int = 42,
    min_words: int = 0,
    filter_vague_predicates: bool = False,
) -> list[FeverSample]:
    """Load FEVER claims from HuggingFace datasets.

    The FEVER dataset is in wide format — each claim appears multiple times
    (once per evidence sentence). This function deduplicates by claim text
    and filters to the requested label types.

    Filtering (applied before capping max_samples):
        min_words:               Drop claims shorter than this word count.
        filter_vague_predicates: Drop claims matching "[X] is/are a [word]."

    Args:
        split:       Dataset split. Options: train, labelled_dev, paper_dev, paper_test.
        max_samples: Cap on number of samples (applied after filtering).
        labels:      Label types to include. Defaults to ["SUPPORTS", "REFUTES"].
        seed:        Random seed for shuffling before capping.

    Returns:
        List of FeverSample objects.
    """
    if labels is None:
        labels = ["SUPPORTS", "REFUTES"]

    console.print(f"[bold]Loading FEVER dataset[/bold] (split={split})...")
    try:
        dataset = load_dataset("fever", "v1.0", split=split, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load FEVER dataset: {e}\n"
            "Make sure you have 'datasets' installed: pip install datasets"
        ) from e

    # Deduplicate by claim — the dataset lists the same claim once per evidence sentence
    seen: dict[str, FeverSample] = {}
    for item in dataset:
        claim = item["claim"]
        label = item["label"]
        if label not in labels:
            continue
        if claim not in seen:
            seen[claim] = FeverSample(
                id=int(item["id"]),
                claim=claim,
                label=label,
            )

    samples = list(seen.values())

    # Apply quality filters before capping — so max_samples applies to filtered set
    if min_words > 0:
        before = len(samples)
        samples = [s for s in samples if len(s.claim.split()) >= min_words]
        console.print(f"  min_words={min_words}: kept {len(samples)}/{before} claims")

    if filter_vague_predicates:
        before = len(samples)
        samples = [s for s in samples if not _VAGUE_CLAIM.match(s.claim)]
        console.print(f"  filter_vague_predicates: kept {len(samples)}/{before} claims")

    # Shuffle with fixed seed so experiments are reproducible
    rng = random.Random(seed)
    rng.shuffle(samples)

    if max_samples is not None:
        # Balance SUPPORTS/REFUTES if possible
        samples = _balance_labels(samples, max_samples, labels)

    label_counts = {}
    for s in samples:
        label_counts[s.label] = label_counts.get(s.label, 0) + 1

    console.print(
        f"Loaded [green]{len(samples)}[/green] samples: "
        + ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items()))
    )
    return samples


def _balance_labels(
    samples: list[FeverSample], max_samples: int, labels: list[str]
) -> list[FeverSample]:
    """Return up to max_samples with balanced label distribution."""
    per_label = max_samples // len(labels)
    buckets: dict[str, list[FeverSample]] = {lbl: [] for lbl in labels}
    for s in samples:
        if s.label in buckets and len(buckets[s.label]) < per_label:
            buckets[s.label].append(s)

    result: list[FeverSample] = []
    for lbl in labels:
        result.extend(buckets[lbl])

    # If one label had fewer samples than per_label, fill remaining slots
    if len(result) < max_samples:
        used_ids = {s.id for s in result}
        for s in samples:
            if s.id not in used_ids:
                result.append(s)
                if len(result) >= max_samples:
                    break

    return result[:max_samples]
