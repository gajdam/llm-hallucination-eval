"""Testy połączenia z każdym LLM.

Uruchomienie:
    pytest src/tests/test.py -v

Opcjonalne flagi:
    pytest src/tests/test.py -v -m anthropic    # tylko Claude
    pytest src/tests/test.py -v -m openai       # tylko OpenAI
    pytest src/tests/test.py -v -m ollama       # tylko Ollama
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# Prosta wiadomość testowa — odpowiedź musi być niepusta
PING_PROMPT = "Reply with exactly one word: Hello"

# -----------------------------------------------------------------------
# Pomocnicze asercje
# -----------------------------------------------------------------------


def assert_valid_response(response, expected_provider: str) -> None:
    """Sprawdza, że odpowiedź LLM jest poprawna."""
    assert response is not None, "Brak odpowiedzi (None)"
    assert response.provider == expected_provider, (
        f"Oczekiwano provider='{expected_provider}', dostałem '{response.provider}'"
    )
    assert not response.failed, f"LLM zwrócił błąd: {response.error}"
    assert isinstance(response.text, str), "Odpowiedź nie jest stringiem"
    assert len(response.text.strip()) > 0, "Odpowiedź jest pusta"


# -----------------------------------------------------------------------
# Claude (Anthropic)
# -----------------------------------------------------------------------


@pytest.mark.anthropic
class TestClaude:
    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Brak ANTHROPIC_API_KEY — ustaw zmienną środowiskową lub plik .env")

    def _make_llm(self, model: str):
        from src.llm.claude_llm import ClaudeLLM

        return ClaudeLLM(model=model)

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ],
    )
    def test_connection(self, model: str):
        """Każdy model Claude powinien zwrócić niepustą odpowiedź."""
        llm = self._make_llm(model)
        response = llm.generate(PING_PROMPT)
        assert_valid_response(response, "anthropic")
        print(f"\n  [{model}] → '{response.text.strip()}'")

    def test_system_prompt(self):
        """Claude powinien respektować system prompt."""
        llm = self._make_llm("claude-haiku-4-5")
        response = llm.generate(
            prompt="What is 2+2?",
            system="Always respond in Polish.",
        )
        assert_valid_response(response, "anthropic")
        # Nie sprawdzamy dokładnej treści, tylko że odpowiedź przyszła
        print(f"\n  system_prompt test → '{response.text.strip()}'")

    def test_usage_tokens_reported(self):
        """Pole usage powinno zawierać liczby tokenów."""
        llm = self._make_llm("claude-haiku-4-5")
        response = llm.generate(PING_PROMPT)
        assert_valid_response(response, "anthropic")
        assert "input_tokens" in response.usage, "Brak 'input_tokens' w usage"
        assert "output_tokens" in response.usage, "Brak 'output_tokens' w usage"
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0


# -----------------------------------------------------------------------
# OpenAI GPT
# -----------------------------------------------------------------------


@pytest.mark.openai
class TestOpenAI:
    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Brak OPENAI_API_KEY — ustaw zmienną środowiskową lub plik .env")

    def _make_llm(self, model: str):
        from src.llm.openai_llm import OpenAILLM

        return OpenAILLM(model=model)

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
        ],
    )
    def test_connection(self, model: str):
        """Każdy model GPT powinien zwrócić niepustą odpowiedź."""
        llm = self._make_llm(model)
        response = llm.generate(PING_PROMPT)
        assert_valid_response(response, "openai")
        print(f"\n  [{model}] → '{response.text.strip()}'")

    def test_system_prompt(self):
        """GPT powinien respektować system prompt."""
        llm = self._make_llm("gpt-4o-mini")
        response = llm.generate(
            prompt="What is 2+2?",
            system="Always respond in Polish.",
        )
        assert_valid_response(response, "openai")
        print(f"\n  system_prompt test → '{response.text.strip()}'")

    def test_usage_tokens_reported(self):
        """Pole usage powinno zawierać liczby tokenów."""
        llm = self._make_llm("gpt-4o-mini")
        response = llm.generate(PING_PROMPT)
        assert_valid_response(response, "openai")
        assert "input_tokens" in response.usage
        assert "output_tokens" in response.usage
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0


# -----------------------------------------------------------------------
# Ollama (lokalne modele)
# -----------------------------------------------------------------------


@pytest.mark.ollama
class TestOllama:
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @pytest.fixture(autouse=True)
    def require_ollama(self):
        """Pomiń, jeśli Ollama nie działa lokalnie."""
        import requests

        try:
            r = requests.get(f"{self.BASE_URL}/api/tags", timeout=3)
            r.raise_for_status()
        except Exception:
            pytest.skip(f"Ollama niedostępna pod {self.BASE_URL} — uruchom: https://ollama.com")

    def _make_llm(self, model: str):
        from src.llm.ollama_llm import OllamaLLM

        return OllamaLLM(model=model, base_url=self.BASE_URL)

    def _available_models(self) -> list[str]:
        """Pobiera listę zaciągniętych modeli z Ollama."""
        import requests

        r = requests.get(f"{self.BASE_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]

    def test_ollama_is_running(self):
        """Ollama musi odpowiadać na /api/tags."""
        import requests

        r = requests.get(f"{self.BASE_URL}/api/tags", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        pulled = [m["name"] for m in data["models"]]
        print(f"\n  Dostępne modele Ollama: {pulled}")

    def test_connection_first_available(self):
        """Testuje połączenie z pierwszym dostępnym modelem."""
        models = self._available_models()
        if not models:
            pytest.skip("Brak zaciągniętych modeli — uruchom: ollama pull llama3.2")

        model_name = models[0].split(":")[0]  # np. "llama3.2:latest" → "llama3.2"
        llm = self._make_llm(model_name)
        response = llm.generate(PING_PROMPT)
        assert_valid_response(response, "ollama")
        print(f"\n  [{model_name}] → '{response.text.strip()}'")

    @pytest.mark.parametrize("model", ["llama3.2", "mistral", "gemma"])
    def test_connection_specific_model(self, model: str):
        """Testuje konkretny model — pomija jeśli nie jest zaciągnięty."""
        available = self._available_models()
        if not any(model in m for m in available):
            pytest.skip(f"Model '{model}' nie jest zaciągnięty — uruchom: ollama pull {model}")

        llm = self._make_llm(model)
        response = llm.generate(PING_PROMPT)
        assert_valid_response(response, "ollama")
        print(f"\n  [{model}] → '{response.text.strip()}'")
