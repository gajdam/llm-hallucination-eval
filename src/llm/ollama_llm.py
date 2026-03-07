"""Ollama LLM — runs open-source models locally.

Requires Ollama to be installed and running: https://ollama.com
Pull a model first:  ollama pull llama3.2
"""

from __future__ import annotations

import json
from typing import Optional

import requests

from .base import BaseLLM, LLMResponse


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "ollama"

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 512},
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            usage = {}
            if "prompt_eval_count" in data:
                usage = {
                    "input_tokens": data.get("prompt_eval_count", 0),
                    "output_tokens": data.get("eval_count", 0),
                }
            return LLMResponse(
                text=text,
                model=self._model,
                provider="ollama",
                usage=usage,
            )
        except requests.ConnectionError:
            return LLMResponse(
                text="", model=self._model, provider="ollama",
                error=(
                    f"Cannot connect to Ollama at {self._base_url}. "
                    "Make sure Ollama is running: https://ollama.com"
                ),
            )
        except requests.HTTPError as e:
            return LLMResponse(
                text="", model=self._model, provider="ollama",
                error=f"HTTPError: {e}"
            )
        except (json.JSONDecodeError, KeyError) as e:
            return LLMResponse(
                text="", model=self._model, provider="ollama",
                error=f"ParseError: {e}"
            )

    def is_available(self) -> bool:
        """Check whether Ollama is running and the model is pulled."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self._model in m for m in models)
        except requests.RequestException:
            return False
