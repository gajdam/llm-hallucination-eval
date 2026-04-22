"""Abstract base class for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)
    error: str | None = None
    latency_s: float = 0.0

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("input_tokens", 0) + self.usage.get("output_tokens", 0)


class BaseLLM(ABC):
    """Common interface for every LLM provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier (e.g. 'claude-opus-4-6')."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name (e.g. 'anthropic')."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate a response for the given prompt."""
