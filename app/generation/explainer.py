"""Slide stage 5 — LLM explanation (Llama 3).

Only the terms with a clean, positive KB match go to the LLM: `status == "explained"`,
not negated, one occurrence per KB entry. Negated findings and coverage gaps keep the
deterministic template wording instead of being handed to the model, for two reasons
found while building this:

1. The template's wording for those cases is already correct and tested (a negated
   finding must never read as present; a gap must never be explained). An LLM asked to
   reproduce that nuance mixed it up in testing — it explained the underlying condition
   before mentioning it was "not found", and it invented a non-existent sentence `kind`.
2. It shrinks the LLM's job to what it is actually useful for: turning a definition into
   natural prose. Everything else stays exactly as safe as it was in step 0.

If the LLM is unreachable, times out, or returns something that doesn't parse into the
expected shape, `generate_explanation` raises `LLMUnavailableError` rather than silently
returning bad text. `app/pipeline.py` catches that and falls back to the template,
logging why.

The prompt is not hardcoded here — see app/generation/prompts/explain_system_prompt.txt,
kept separate so it can be iterated on without touching this file.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.generation.llm_client import LLMUnavailableError, generate_json
from app.generation.template import template_explanation
from app.models import Explanation, ExplanationSentence, FlaggedTerm, RetrievedChunk

IMPLEMENTATION = "real"

PROMPT_PATH = Path(__file__).parent / "prompts" / "explain_system_prompt.txt"


class _LLMSentence(BaseModel):
    text: str
    kb_ids: list[str]


class _LLMResponse(BaseModel):
    sentences: list[_LLMSentence]


def _selectable(terms: list[FlaggedTerm]) -> dict[str, FlaggedTerm]:
    """The terms the LLM may explain: one per KB entry, explained, not negated."""
    selected: dict[str, FlaggedTerm] = {}
    for term in terms:
        if term.status == "explained" and term.kb_id and not term.negated:
            selected.setdefault(term.kb_id, term)
    return selected


def _build_prompt(original_text: str, evidence: list[RetrievedChunk]) -> str:
    system = PROMPT_PATH.read_text(encoding="utf-8")
    listing = "\n".join(f"- id: {c.kb_id}, term: {c.term}, definition: {c.plain_definition}" for c in evidence)
    return f"{system}\n\nOriginal clinical text: {original_text}\n\nRetrieved Evidence:\n{listing}\n"


def _call_llm(original_text: str, evidence: list[RetrievedChunk]) -> list[ExplanationSentence]:
    raw = generate_json(_build_prompt(original_text, evidence))
    try:
        parsed = _LLMResponse.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMUnavailableError(f"unparseable response: {exc}") from exc

    allowed_ids = {c.kb_id for c in evidence}
    sentences: list[ExplanationSentence] = []
    for sentence in parsed.sentences:
        # The model sometimes cites an id we didn't send, or none at all. Keep only the
        # ids it was actually given evidence for; drop a sentence grounded in nothing.
        kb_ids = [kb_id for kb_id in dict.fromkeys(sentence.kb_ids) if kb_id in allowed_ids]
        if kb_ids and sentence.text.strip():
            sentences.append(ExplanationSentence(text=sentence.text.strip(), kb_ids=kb_ids, kind="definition"))

    if not sentences:
        raise LLMUnavailableError("response contained no groundable sentences")
    return sentences


def generate_explanation(
    original_text: str,
    terms: list[FlaggedTerm],
    retrieved: dict[str, list[RetrievedChunk]],
) -> Explanation:
    selectable = _selectable(terms)
    evidence = [retrieved[kb_id][0] for kb_id in selectable if retrieved.get(kb_id)]

    # Every instance of a term the LLM is covering is excluded here, not just the first
    # occurrence, so the same KB entry is never explained twice (once by each generator).
    def llm_is_covering(term: FlaggedTerm) -> bool:
        return term.status == "explained" and not term.negated and term.kb_id in selectable

    other_terms = [t for t in terms if not llm_is_covering(t)]
    fixed_sentences = [
        s for s in template_explanation(other_terms, retrieved).sentences if s.kind != "notice"
    ] if other_terms else []

    if not evidence:
        return template_explanation(terms, retrieved)  # nothing suitable for the LLM

    llm_sentences = _call_llm(original_text, evidence)  # LLMUnavailableError propagates
    return Explanation(sentences=[*llm_sentences, *fixed_sentences], generator=f"llm:{_model_name()}")


def _model_name() -> str:
    from app.generation.llm_client import MODEL_NAME

    return MODEL_NAME
