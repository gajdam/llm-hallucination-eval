"""Abstract base class for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None


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
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate a response for the given prompt."""
