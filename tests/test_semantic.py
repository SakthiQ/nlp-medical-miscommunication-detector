"""Step 6: semantic grounding advisory check. Uses the real embedding model from the
local cache (fast), never the LLM (no `llm` mark needed)."""

from __future__ import annotations

import pytest

from app.models import Explanation, ExplanationSentence, RetrievedChunk
from app.retrieval.knowledge_base import load_knowledge_base
from app.retrieval.retriever import Retriever
from app.safety.semantic import ADVISORY_THRESHOLD, semantic_advisory_flags


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


@pytest.fixture(scope="module")
def retriever(kb):
    return Retriever(kb)


def _chunk(kb, kb_id):
    entry = kb.get(kb_id)
    return RetrievedChunk(
        kb_id=entry.id, term=entry.term, plain_definition=entry.plain_definition,
        clinical_context=entry.clinical_context, source_citation=entry.source_citation,
        score=1.0,
    )


def _explanation(*sentences):
    return Explanation(sentences=list(sentences), generator="llm:test")


def test_sentence_matching_its_definition_is_not_flagged(kb, retriever):
    entry = kb.get("kb-001")
    explanation = _explanation(
        ExplanationSentence(text=entry.plain_definition, kb_ids=["kb-001"], kind="definition")
    )
    flags = semantic_advisory_flags(explanation, {"kb-001": [_chunk(kb, "kb-001")]}, retriever.embed)
    assert flags == []


def test_wildly_unrelated_sentence_is_flagged(kb, retriever):
    explanation = _explanation(
        ExplanationSentence(
            text="The weather was sunny with a light breeze from the northwest.",
            kb_ids=["kb-001"], kind="definition",
        )
    )
    [flag] = semantic_advisory_flags(explanation, {"kb-001": [_chunk(kb, "kb-001")]}, retriever.embed)
    assert flag.startswith("semantic_advisory: sentence 0")
    assert "kb-001" in flag
    assert "informational only, not enforced" in flag


def test_gap_and_notice_sentences_are_never_checked(kb, retriever):
    explanation = _explanation(
        ExplanationSentence(text="Completely unrelated filler text about the weather.", kb_ids=[], kind="gap"),
        ExplanationSentence(text="Also unrelated filler text about traffic.", kb_ids=[], kind="notice"),
    )
    assert semantic_advisory_flags(explanation, {}, retriever.embed) == []


def test_sentence_citing_an_unretrieved_id_is_skipped_not_crashed(kb, retriever):
    explanation = _explanation(
        ExplanationSentence(text="Some text.", kb_ids=["kb-999"], kind="definition")
    )
    assert semantic_advisory_flags(explanation, {}, retriever.embed) == []


def test_no_definition_sentences_returns_empty_without_calling_embed(kb):
    def fail(texts):
        raise AssertionError("embed must not be called with nothing to check")

    explanation = _explanation(ExplanationSentence(text="x", kb_ids=[], kind="notice"))
    assert semantic_advisory_flags(explanation, {}, fail) == []


def test_threshold_is_calibrated_below_the_measured_faithful_floor():
    """See eval/semantic_grounding_results.md: the lowest faithful score measured was
    0.265. The advisory threshold must stay below that, or it starts flagging
    already-known-good sentences by construction."""
    assert ADVISORY_THRESHOLD < 0.265


# --- pipeline wiring: advisory flags never change pass/fail -------------------------


def test_advisory_flag_never_changes_validation_passed(monkeypatch):
    """A low semantic score alone must not fail the response — see app/safety/semantic.py."""
    from app.retrieval.retriever import Retriever as RetrieverClass

    from app.pipeline import ExplainPipeline

    kb = load_knowledge_base()
    pipeline = ExplainPipeline(kb, ner=_StubNER(), retriever=RetrieverClass(kb))

    def unrelated_llm(original_text, terms, retrieved):
        return Explanation(
            sentences=[
                ExplanationSentence(
                    text="The weather was sunny with a light breeze.",
                    kb_ids=["kb-001"], kind="definition",
                )
            ],
            generator="llm:test",
        )

    monkeypatch.setattr("app.pipeline.generate_explanation", unrelated_llm)
    response = pipeline.run("Mild hepatic steatosis with elevated ALT.")

    assert response.validation.passed is True
    assert response.validation.fallback_used is False
    assert any(f.startswith("semantic_advisory:") for f in response.validation.flags)
    assert "sunny" in response.explanation  # the advisory-flagged text was still shown


class _StubNER:
    def extract(self, text, sentences):
        return []
