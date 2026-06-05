"""Retrieves factual context for a claim from Wikidata (KG) and Wikipedia (IR)."""

from __future__ import annotations

from .entity_extractor import EntityExtractor
from .wikidata_client import WikidataClient
from ..retrieval.wikipedia_retriever import WikipediaRetriever


class KGRetriever:
    def __init__(
        self,
        max_entities: int = 3,
        wikidata_timeout: int = 5,
        wikipedia_timeout: int = 5,
    ):
        self._extractor = EntityExtractor()
        self._wikidata = WikidataClient(timeout=wikidata_timeout)
        self._wikipedia = WikipediaRetriever(timeout=wikipedia_timeout)
        self._max_entities = max_entities
        # In-memory caches keyed by entity name — shared across all claims in a run
        self._kg_cache: dict[str, str | None] = {}
        self._ir_cache: dict[str, str | None] = {}

    def get_kg_context(self, claim: str) -> str:
        """Return Wikidata descriptions for entities found in the claim."""
        entities = self._extractor.extract(claim, max_entities=self._max_entities)
        snippets: list[str] = []
        for ent in entities:
            if ent not in self._kg_cache:
                self._kg_cache[ent] = self._wikidata.lookup(ent)
            text = self._kg_cache[ent]
            if text:
                snippets.append(text)
        return "\n".join(snippets)

    def get_ir_context(self, claim: str) -> str:
        """Return Wikipedia article summaries for entities found in the claim."""
        entities = self._extractor.extract(claim, max_entities=self._max_entities)
        snippets: list[str] = []
        for ent in entities:
            if ent not in self._ir_cache:
                self._ir_cache[ent] = self._wikipedia.get_summary(ent)
            text = self._ir_cache[ent]
            if text:
                snippets.append(text)
        return "\n".join(snippets)

    def get_hybrid_context(self, kg_context: str, ir_context: str) -> str:
        """Merge KG and IR contexts with section headers."""
        parts: list[str] = []
        if kg_context.strip():
            parts.append(f"[Knowledge Graph]\n{kg_context.strip()}")
        if ir_context.strip():
            parts.append(f"[Retrieved Text]\n{ir_context.strip()}")
        return "\n\n".join(parts)
