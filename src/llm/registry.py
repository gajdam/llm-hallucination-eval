"""Instantiate LLM objects from config entries."""

from __future__ import annotations

from rich.console import Console

from .base import BaseLLM
from .claude_llm import ClaudeLLM
from .openai_llm import OpenAILLM
from .ollama_llm import OllamaLLM

console = Console()


def build_llms(llm_configs: list[dict], llm_filter: str | None = None) -> list[BaseLLM]:
    """Build LLM instances from the 'llms' section of config.yaml.

    Args:
        llm_configs: List of dicts with 'name' and 'provider' keys.
        llm_filter:  Optional model name to select a single LLM.

    Returns:
        List of ready-to-use BaseLLM instances.
    """
    llms: list[BaseLLM] = []
    for cfg in llm_configs:
        name = cfg["name"]
        provider = cfg.get("provider", "").lower()

        if llm_filter and name != llm_filter:
            continue

        try:
            if provider == "anthropic":
                llms.append(ClaudeLLM(model=name))

            elif provider == "openai":
                llms.append(OpenAILLM(model=name))

            elif provider == "ollama":
                llm = OllamaLLM(
                    model=name,
                    base_url=cfg.get("base_url", "http://localhost:11434"),
                )
                if not llm.is_available():
                    console.print(
                        f"[yellow]Skipping Ollama model '{name}': "
                        "Ollama not running or model not pulled. "
                        f"Run: ollama pull {name}[/yellow]"
                    )
                    continue
                llms.append(llm)

            else:
                console.print(f"[yellow]Unknown provider '{provider}' for '{name}' — skipping[/yellow]")
                continue

            console.print(f"  [green]✓[/green] {provider}/{name}")

        except Exception as e:
            console.print(f"  [red]✗[/red] {provider}/{name}: {e}")

    return llms
