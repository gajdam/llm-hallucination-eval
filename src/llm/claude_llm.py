"""Claude LLM via Anthropic API."""

from __future__ import annotations

import anthropic

from .base import BaseLLM, LLMResponse

# Models that support adaptive thinking
_ADAPTIVE_THINKING_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}


class ClaudeLLM(BaseLLM):
    def __init__(self, model: str = "claude-opus-4-6"):
        self._model = model
        self._client = anthropic.Anthropic()

    @property
    def name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            # Use streaming + get_final_message to avoid HTTP timeouts
            with self._client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()

            text = next((b.text for b in response.content if b.type == "text"), "")
            return LLMResponse(
                text=text,
                model=self._model,
                provider="anthropic",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )
        except anthropic.RateLimitError as e:
            return LLMResponse(
                text="", model=self._model, provider="anthropic", error=f"RateLimitError: {e}"
            )
        except anthropic.APIError as e:
            return LLMResponse(
                text="", model=self._model, provider="anthropic", error=f"APIError: {e}"
            )
