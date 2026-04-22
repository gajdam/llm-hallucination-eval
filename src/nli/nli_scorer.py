"""NLI scorer — wraps a HuggingFace NLI model.

The NLI model decides the relationship between a premise and a hypothesis:
  ENTAILMENT    – hypothesis logically follows from the premise
  NEUTRAL       – hypothesis is unrelated / cannot be determined
  CONTRADICTION – hypothesis contradicts the premise

We use premise=llm_response, hypothesis=claim.
This asks: "Does what the LLM said imply the original claim?"

Hallucination detection:
  FEVER=SUPPORTS + NLI(llm_response, claim)=CONTRADICTION → hallucination
  FEVER=REFUTES  + NLI(llm_response, claim)=ENTAILMENT    → hallucination
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from rich.console import Console
from transformers import AutoModelForSequenceClassification, AutoTokenizer

console = Console()

# Standard NLI label names (normalised to uppercase)
ENTAILMENT = "ENTAILMENT"
NEUTRAL = "NEUTRAL"
CONTRADICTION = "CONTRADICTION"


@dataclass
class NLIResult:
    label: str  # ENTAILMENT | NEUTRAL | CONTRADICTION
    entailment: float
    neutral: float
    contradiction: float

    @property
    def scores(self) -> dict[str, float]:
        return {
            ENTAILMENT: self.entailment,
            NEUTRAL: self.neutral,
            CONTRADICTION: self.contradiction,
        }


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


class NLIScorer:
    """Wraps a HuggingFace sequence-classification model for NLI."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = _resolve_device(device)

        console.print(f"[bold]Loading NLI model[/bold]: {model_name} → device={self._device}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self._device
        )
        self._model.eval()

        # Build a normalised label → column-index mapping
        id2label: dict[int, str] = {
            int(k): v.upper() for k, v in self._model.config.id2label.items()
        }
        # Normalise aliases: "ENTAILS" → "ENTAILMENT", "CONTRADICT" → "CONTRADICTION"
        _aliases = {
            "ENTAILS": ENTAILMENT,
            "ENTAILMENT": ENTAILMENT,
            "NEUTRAL": NEUTRAL,
            "CONTRADICTION": CONTRADICTION,
            "CONTRADICTS": CONTRADICTION,
            "CONTRADICT": CONTRADICTION,
        }
        self._idx: dict[str, int] = {}
        for idx, raw_label in id2label.items():
            canonical = _aliases.get(raw_label, raw_label)
            self._idx[canonical] = idx

        # Verify we have all three labels
        missing = {ENTAILMENT, NEUTRAL, CONTRADICTION} - set(self._idx.keys())
        if missing:
            raise ValueError(
                f"NLI model '{model_name}' is missing labels: {missing}. Available: {id2label}"
            )
        console.print(f"  Label mapping: {self._idx}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """Score a single (premise, hypothesis) pair."""
        return self.predict_batch([(premise, hypothesis)])[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Score a list of (premise, hypothesis) pairs efficiently."""
        all_results: list[NLIResult] = []

        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            enc = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = F.softmax(logits, dim=-1).cpu()

            for row in probs:
                e = float(row[self._idx[ENTAILMENT]])
                n = float(row[self._idx[NEUTRAL]])
                c = float(row[self._idx[CONTRADICTION]])
                label = max(
                    {ENTAILMENT: e, NEUTRAL: n, CONTRADICTION: c},
                    key=lambda k: {ENTAILMENT: e, NEUTRAL: n, CONTRADICTION: c}[k],
                )
                all_results.append(NLIResult(label=label, entailment=e, neutral=n, contradiction=c))

        return all_results


# ------------------------------------------------------------------
# Hallucination detection logic
# ------------------------------------------------------------------


def is_hallucination(
    fever_label: str,
    nli_result: NLIResult,
) -> bool | None:
    """Map a FEVER label + NLI result to a hallucination judgment.

    Returns:
        True   — hallucination detected
        False  — no hallucination detected
        None   — NEUTRAL / ambiguous (cannot decide)

    Logic:
        SUPPORTS + CONTRADICTION  → hallucination (LLM contradicts the true claim)
        SUPPORTS + ENTAILMENT     → no hallucination
        REFUTES  + ENTAILMENT     → hallucination (LLM asserts the false claim)
        REFUTES  + CONTRADICTION  → no hallucination (LLM correctly rejects false claim)
        anything + NEUTRAL        → ambiguous
    """
    label = nli_result.label
    if fever_label == "SUPPORTS":
        if label == CONTRADICTION:
            return True
        if label == ENTAILMENT:
            return False
    elif fever_label == "REFUTES":
        if label == ENTAILMENT:
            return True
        if label == CONTRADICTION:
            return False
    return None  # NEUTRAL — we don't count this as hallucination
