"""Wikipedia REST API client for article summary retrieval."""

from __future__ import annotations

import requests

_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"
_HEADERS = {"User-Agent": "llm-hallucination-eval/1.0 (academic research)"}
_MAX_CHARS = 600


class WikipediaRetriever:
    def __init__(self, timeout: int = 5):
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def get_summary(self, title: str) -> str | None:
        """Return the plain-text summary extract for a Wikipedia article title."""
        safe_title = title.replace(" ", "_")
        try:
            r = self._session.get(
                f"{_BASE}/{safe_title}",
                timeout=self._timeout,
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            extract = r.json().get("extract", "")
            return extract[:_MAX_CHARS] if extract else None
        except Exception:
            return None
