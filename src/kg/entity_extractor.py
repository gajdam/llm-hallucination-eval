"""Named entity extraction using spaCy.

Primary strategy: spaCy NER (PERSON, ORG, GPE, WORK_OF_ART, etc.)
Fallback: extract noun chunks that start with a capital letter when NER finds nothing.
This significantly improves coverage on short FEVER-style factoid claims.
"""

from __future__ import annotations

import re
from functools import lru_cache

_LABELS = {"PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT", "PRODUCT", "LOC", "FAC", "NORP"}
# Matches a sequence of title-cased or all-caps words (excludes sentence-start noise)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


@lru_cache(maxsize=1)
def _load_nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )
    except ImportError:
        raise RuntimeError(
            "spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm"
        )


class EntityExtractor:
    def __init__(self):
        self._nlp = _load_nlp()

    def extract(self, text: str, max_entities: int = 3) -> list[str]:
        """Return up to max_entities named entities from text.

        Falls back to regex-matched proper-noun phrases when spaCy NER finds nothing.
        """
        doc = self._nlp(text)
        seen: set[str] = set()
        result: list[str] = []

        for ent in doc.ents:
            if ent.label_ in _LABELS and ent.text not in seen:
                seen.add(ent.text)
                result.append(ent.text)
                if len(result) >= max_entities:
                    return result

        if not result:
            # Fallback: multi-word proper-noun phrases (title case)
            # Skip the very first word to avoid treating sentence starts as entities
            rest = text[text.index(" ") + 1:] if " " in text else ""
            for match in _PROPER_NOUN_RE.finditer(rest):
                phrase = match.group()
                if phrase not in seen:
                    seen.add(phrase)
                    result.append(phrase)
                    if len(result) >= max_entities:
                        break

        return result
