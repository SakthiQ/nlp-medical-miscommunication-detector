"""Slide stage 3 — terminology identification.

Two sources of terms are merged here:
- the dictionary matcher, which finds knowledge-base terms with exact precision, and
- NER entities, which catch everything else.

Dictionary matches win on overlapping spans. Commonly understood terms ("fever") are
matched but never flagged. An NER term with no KB entry becomes a coverage gap.
"""

from __future__ import annotations

import re

from app.models import Entity, FlaggedTerm, Sentence
from app.retrieval.knowledge_base import KnowledgeBase

IMPLEMENTATION = "partial"


def _surface_pattern(surface: str) -> str:
    """Match a KB surface form, letting spaces and hyphens vary ('low-density' / 'low density')."""
    tokens = [token for token in re.split(r"[\s\-]+", surface) if token]
    return r"[\s\-]+".join(re.escape(token) for token in tokens)


class DictionaryMatcher:
    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb
        surfaces = {surface for entry in kb.entries for surface in (entry.term, *entry.aliases)}
        # Longest first, so "fatty liver disease" wins over "fatty liver" at the same position.
        alternatives = "|".join(
            _surface_pattern(surface) for surface in sorted(surfaces, key=len, reverse=True)
        )
        self._regex = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)

    def match(self, sentences: list[Sentence]) -> list[Entity]:
        entities: list[Entity] = []
        for sentence in sentences:
            for found in self._regex.finditer(sentence.text):
                entry = self._kb.lookup(found.group())
                if entry is None:
                    continue
                entities.append(
                    Entity(
                        text=found.group(),
                        label=entry.label,
                        start=sentence.start + found.start(),
                        end=sentence.start + found.end(),
                        sentence_index=sentence.index,
                        confidence=1.0,
                        source="dictionary",
                        kb_id=entry.id,
                    )
                )
        return entities


def identify_terms(
    dictionary_entities: list[Entity],
    ner_entities: list[Entity],
    kb: KnowledgeBase,
) -> list[FlaggedTerm]:
    terms: list[FlaggedTerm] = []
    taken: list[tuple[int, int]] = []

    def overlaps(entity: Entity) -> bool:
        return any(entity.start < end and start < entity.end for start, end in taken)

    for entity in sorted(dictionary_entities, key=lambda e: e.start):
        taken.append((entity.start, entity.end))
        entry = kb.get(entity.kb_id) if entity.kb_id else None
        if entry is None or entry.is_common:
            continue
        terms.append(_flag(entity, entry.id, "explained"))

    for entity in sorted(ner_entities, key=lambda e: e.start):
        if overlaps(entity):
            continue
        taken.append((entity.start, entity.end))
        entry = kb.lookup(entity.text)
        if entry is not None and entry.is_common:
            continue
        if entry is not None:
            terms.append(_flag(entity, entry.id, "explained"))
        else:
            terms.append(_flag(entity, None, "no_vetted_definition"))

    return sorted(terms, key=lambda term: term.start)


def _flag(entity: Entity, kb_id: str | None, status: str) -> FlaggedTerm:
    return FlaggedTerm(
        text=entity.text,
        label=entity.label,
        start=entity.start,
        end=entity.end,
        sentence_index=entity.sentence_index,
        kb_id=kb_id,
        status=status,
        negated=entity.negated,
    )
