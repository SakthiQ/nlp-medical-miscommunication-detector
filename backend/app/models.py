"""Data passed between pipeline stages.

Offsets (`start`, `end`) are always absolute character positions in the original report
text, so any stage's output can be traced back to — and highlighted in — the source.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Sentence(BaseModel):
    index: int
    text: str
    start: int
    end: int


class Entity(BaseModel):
    """A medical term found in the report, before deciding whether to explain it."""

    text: str
    label: str
    start: int
    end: int
    sentence_index: int
    confidence: float
    source: Literal["dictionary", "ner"]
    kb_id: str | None = None
    negated: bool = False


TermStatus = Literal["explained", "no_vetted_definition"]


class FlaggedTerm(BaseModel):
    """A term the patient will see highlighted, with or without a vetted definition."""

    text: str
    label: str
    start: int
    end: int
    sentence_index: int
    kb_id: str | None
    status: TermStatus
    negated: bool = False


class RetrievedChunk(BaseModel):
    """A KB entry returned by retrieval.

    `exact` chunks come from a term's own KB link and may be used to explain it.
    `semantic` chunks are nearest neighbours by embedding similarity; they are only
    ever curation suggestions, never patient-facing evidence.
    """

    kb_id: str
    term: str
    plain_definition: str
    clinical_context: str
    source_citation: str
    source_url: str | None = None
    score: float
    match: Literal["exact", "semantic"] = "exact"


class ExplanationSentence(BaseModel):
    """One sentence of the explanation, tagged with the KB entries it draws from.

    `definition` sentences must cite at least one retrieved KB id. `gap` and `notice`
    sentences make no medical claim, so they must cite none.
    """

    text: str
    kb_ids: list[str]
    kind: Literal["definition", "gap", "notice"]


class Explanation(BaseModel):
    sentences: list[ExplanationSentence]
    generator: str

    @property
    def plain_text(self) -> str:
        return "\n".join(sentence.text for sentence in self.sentences)


class ValidationResult(BaseModel):
    passed: bool
    flags: list[str]
    final_text: str
