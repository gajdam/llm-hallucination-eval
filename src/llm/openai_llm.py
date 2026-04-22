"""OpenAI GPT LLM."""

from __future__ import annotations

from .base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o"):
        self._model = model
        # Import lazily so the package doesn't hard-fail if openai is missing
        try:
            from openai import OpenAI

            self._client = OpenAI()
        except ImportError as e:
            raise ImportError("OpenAI package not installed. Run: pip install openai") from e

    @property
    def name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        from openai import APIError, RateLimitError

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=512,
            )
            text = response.choices[0].message.content or ""
            usage = {}
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
            return LLMResponse(
                text=text,
                model=self._model,
                provider="openai",
                usage=usage,
            )
        except RateLimitError as e:
            return LLMResponse(
                text="", model=self._model, provider="openai", error=f"RateLimitError: {e}"
            )
        except APIError as e:
            return LLMResponse(
                text="", model=self._model, provider="openai", error=f"APIError: {e}"
            )
