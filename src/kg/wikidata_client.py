"""Wikidata REST API client for entity lookup."""

from __future__ import annotations

import time

import requests

_API = "https://www.wikidata.org/w/api.php"
_HEADERS = {"User-Agent": "llm-hallucination-eval/1.0 (academic research)"}


class WikidataClient:
    def __init__(self, timeout: int = 5):
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def search_entity(self, name: str) -> str | None:
        """Return the first Wikidata entity ID matching name, or None."""
        try:
            r = self._session.get(
                _API,
                params={
                    "action": "wbsearchentities",
                    "search": name,
                    "language": "en",
                    "limit": 1,
                    "format": "json",
                },
                timeout=self._timeout,
            )
            r.raise_for_status()
            results = r.json().get("search", [])
            return results[0]["id"] if results else None
        except Exception:
            return None

    def get_description(self, entity_id: str) -> str | None:
        """Return 'label: description' for an entity ID, or None."""
        try:
            r = self._session.get(
                _API,
                params={
                    "action": "wbgetentities",
                    "ids": entity_id,
                    "props": "labels|descriptions",
                    "languages": "en",
                    "format": "json",
                },
                timeout=self._timeout,
            )
            r.raise_for_status()
            entity = r.json().get("entities", {}).get(entity_id, {})
            label = entity.get("labels", {}).get("en", {}).get("value", "")
            desc = entity.get("descriptions", {}).get("en", {}).get("value", "")
            if not label:
                return None
            return f"{label}: {desc}" if desc else label
        except Exception:
            return None

    def lookup(self, name: str) -> str | None:
        """Full pipeline: name → entity ID → 'label: description'."""
        entity_id = self.search_entity(name)
        if not entity_id:
            return None
        time.sleep(0.1)  # gentle rate limiting between two API calls
        return self.get_description(entity_id)
