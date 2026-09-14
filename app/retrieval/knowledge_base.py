"""Knowledge base loading, validation, and alias indexing.

The KB is the precision path of the pipeline: a term is explained only if a human
curated a definition for it. Terms found in a report that are *not* in the KB are a
first-class outcome (a coverage gap), not a failure — see `KnowledgeBase.lookup`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_KB_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base.json"

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(term: str) -> str:
    """Casefold and strip punctuation/whitespace so 'HbA1c' and 'hba1c' match."""
    return _NORMALIZE_RE.sub(" ", term.casefold()).strip()


class KBEntry(BaseModel):
    id: str
    term: str
    aliases: list[str] = Field(default_factory=list)
    label: str
    plain_definition: str
    clinical_context: str
    source_citation: str
    source_url: str | None = None
    is_common: bool = False

    @property
    def all_surfaces(self) -> list[str]:
        """Every string that should match this entry, including the canonical term."""
        surfaces = [self.term, *self.aliases]
        seen: dict[str, None] = {}
        for surface in surfaces:
            seen.setdefault(normalize(surface), None)
        return list(seen)


class KnowledgeBaseFile(BaseModel):
    schema_version: str
    notes: str | None = None
    entries: list[KBEntry]


class KnowledgeBaseError(ValueError):
    """Raised when the curated KB violates an invariant the pipeline depends on."""


@dataclass
class KnowledgeBase:
    entries: list[KBEntry]
    _by_id: dict[str, KBEntry] = field(default_factory=dict, repr=False)
    _by_alias: dict[str, KBEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for entry in self.entries:
            if entry.id in self._by_id:
                raise KnowledgeBaseError(f"duplicate KB entry id: {entry.id}")
            self._by_id[entry.id] = entry

        for entry in self.entries:
            for surface in entry.all_surfaces:
                existing = self._by_alias.get(surface)
                if existing is not None and existing.id != entry.id:
                    raise KnowledgeBaseError(
                        f"alias {surface!r} maps to both {existing.id} and {entry.id}; "
                        "aliases must be unambiguous for dictionary matching"
                    )
                self._by_alias[surface] = entry

    def get(self, entry_id: str) -> KBEntry | None:
        return self._by_id.get(entry_id)

    def lookup(self, surface: str) -> KBEntry | None:
        """Exact (normalized) surface-form lookup. Returns None on a coverage gap."""
        return self._by_alias.get(normalize(surface))

    @property
    def explainable(self) -> list[KBEntry]:
        """Entries worth explaining to a patient (excludes commonly understood terms)."""
        return [entry for entry in self.entries if not entry.is_common]

    @property
    def common(self) -> list[KBEntry]:
        """Commonly understood terms — matched, but never flagged for explanation."""
        return [entry for entry in self.entries if entry.is_common]

    def __len__(self) -> int:
        return len(self.entries)


def load_knowledge_base(path: Path | str = DEFAULT_KB_PATH) -> KnowledgeBase:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    parsed = KnowledgeBaseFile.model_validate(raw)
    return KnowledgeBase(entries=parsed.entries)
