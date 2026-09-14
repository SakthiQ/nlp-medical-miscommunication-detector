from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.models import TermStatus


class ExplainRequest(BaseModel):
    report_text: str = Field(min_length=1, max_length=20_000)
    target_language: str = "en"
    include_audio: bool = False

    @field_validator("report_text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("report_text must contain some text")
        return value


class TermOut(BaseModel):
    text: str
    label: str
    start: int
    end: int
    status: TermStatus
    kb_id: str | None
    definition: str | None
    source_citation: str | None
    negated: bool


class CitationOut(BaseModel):
    kb_id: str
    term: str
    source_citation: str
    source_url: str | None


class SuggestionOut(BaseModel):
    kb_id: str
    term: str
    score: float


class CurationSuggestion(BaseModel):
    """For knowledge-base curators; never shown to patients.

    The nearest KB entries to a term that has no vetted definition. Similarity means
    related, not equivalent ("renal" is nearest to hydronephrosis), so a curator must
    confirm a suggestion by adding an alias, or reject it.
    """

    text: str
    suggestions: list[SuggestionOut]


class ValidationOut(BaseModel):
    passed: bool
    flags: list[str]
    fallback_used: bool


class ExplainResponse(BaseModel):
    original_text: str
    target_language: str
    terms: list[TermOut]
    explanation: str
    citations: list[CitationOut]
    curation_suggestions: list[CurationSuggestion] = []
    validation: ValidationOut
    audio_base64: str | None = None
    pipeline: dict[str, str]
    timings_ms: dict[str, float]
    provenance_id: str | None = None
    """Set only when audit logging (AUDIT_LOG_PATH) is enabled; the id of this request's
    logged provenance record, for a clinician to look up. Off by default."""
