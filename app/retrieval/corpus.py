"""Evaluation corpus loading.

The corpus is hand-annotated. A gold term with `kb_id: null` is a deliberate coverage
gap — the pipeline is expected to detect the term and report that no vetted definition
exists, rather than inventing one.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_corpus.json"


class GoldTerm(BaseModel):
    surface: str
    kb_id: str | None = None


class Snippet(BaseModel):
    id: str
    source_type: str
    text: str
    gold_terms: list[GoldTerm]

    @property
    def covered_terms(self) -> list[GoldTerm]:
        return [term for term in self.gold_terms if term.kb_id is not None]

    @property
    def gap_terms(self) -> list[GoldTerm]:
        return [term for term in self.gold_terms if term.kb_id is None]


class EvalCorpus(BaseModel):
    schema_version: str
    notes: str | None = None
    snippets: list[Snippet]

    def __len__(self) -> int:
        return len(self.snippets)


def load_corpus(path: Path | str = DEFAULT_CORPUS_PATH) -> EvalCorpus:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalCorpus.model_validate(raw)
