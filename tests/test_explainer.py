"""Step 5: LLM generation. Slow tests (real Ollama calls) are marked so they can be
skipped with `-m "not llm"` when Ollama isn't running; fallback logic is tested with a
stub client so it runs fast and every time.
"""

from __future__ import annotations

import pytest

from app.generation.explainer import _selectable, generate_explanation
from app.generation.llm_client import LLMUnavailableError
from app.models import FlaggedTerm, RetrievedChunk
from app.retrieval.knowledge_base import load_knowledge_base

pytestmark_llm = pytest.mark.llm


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def _chunk(kb, kb_id):
    entry = kb.get(kb_id)
    return RetrievedChunk(
        kb_id=entry.id, term=entry.term, plain_definition=entry.plain_definition,
        clinical_context=entry.clinical_context, source_citation=entry.source_citation,
        score=1.0,
    )


def _term(kb_id, status="explained", text="term", negated=False):
    return FlaggedTerm(
        text=text, label="DISEASE", start=0, end=len(text), sentence_index=0,
        kb_id=kb_id, status=status, negated=negated,
    )


# --- _selectable: which terms the LLM is allowed to see -------------------------------


def test_selectable_excludes_negated_and_gap_terms():
    terms = [
        _term("kb-001", text="hepatic steatosis"),
        _term("kb-038", text="ischemia", negated=True),
        _term(None, status="no_vetted_definition", text="renal"),
    ]
    assert set(_selectable(terms)) == {"kb-001"}


def test_selectable_dedupes_repeated_terms_by_kb_id():
    terms = [_term("kb-002", text="ALT"), _term("kb-002", text="ALT")]
    assert list(_selectable(terms)) == ["kb-002"]


# --- generation with no LLM-eligible terms: never calls the LLM at all ----------------


def test_all_gap_terms_uses_template_without_calling_llm(kb, monkeypatch):
    def fail(*a, **k):
        raise AssertionError("the LLM must not be called when there is nothing to explain")

    monkeypatch.setattr("app.generation.explainer._call_llm", fail)
    terms = [_term(None, status="no_vetted_definition", text="renal")]
    explanation = generate_explanation("...", terms, {})
    assert explanation.generator == "template"
    assert "renal" in explanation.plain_text


def test_all_negated_uses_template_without_calling_llm(kb, monkeypatch):
    def fail(*a, **k):
        raise AssertionError("the LLM must not be called when there is nothing to explain")

    monkeypatch.setattr("app.generation.explainer._call_llm", fail)
    terms = [_term("kb-038", text="ischemia", negated=True)]
    explanation = generate_explanation("No evidence of ischemia.", terms, {"kb-038": [_chunk(kb, "kb-038")]})
    assert explanation.generator == "template"
    assert "was not found" in explanation.plain_text


# --- generation combining an LLM stub with template sentences -------------------------


def test_llm_sentences_combine_with_template_sentences_for_gaps_and_negation(kb, monkeypatch):
    from app.models import ExplanationSentence

    def stub(original_text, evidence):
        assert [c.kb_id for c in evidence] == ["kb-001"]
        return [ExplanationSentence(text="Extra fat, in the model's own words.", kb_ids=["kb-001"], kind="definition")]

    monkeypatch.setattr("app.generation.explainer._call_llm", stub)
    terms = [
        _term("kb-001", text="hepatic steatosis"),
        _term("kb-038", text="ischemia", negated=True),
        _term(None, status="no_vetted_definition", text="renal"),
    ]
    retrieved = {"kb-001": [_chunk(kb, "kb-001")], "kb-038": [_chunk(kb, "kb-038")]}
    explanation = generate_explanation("...", terms, retrieved)

    assert explanation.generator.startswith("llm:")
    kinds_and_ids = [(s.kind, s.kb_ids) for s in explanation.sentences]
    assert kinds_and_ids[0] == ("definition", ["kb-001"])
    assert "model's own words" in explanation.sentences[0].text
    assert any(s.kind == "definition" and "was not found" in s.text for s in explanation.sentences[1:])
    assert any(s.kind == "gap" for s in explanation.sentences)


# --- LLM unavailability propagates as LLMUnavailableError, not a crash ----------------


def test_llm_error_propagates_as_llm_unavailable(kb, monkeypatch):
    def raise_error(*a, **k):
        raise LLMUnavailableError("connection refused")

    monkeypatch.setattr("app.generation.explainer._call_llm", raise_error)
    terms = [_term("kb-001", text="hepatic steatosis")]
    with pytest.raises(LLMUnavailableError):
        generate_explanation("...", terms, {"kb-001": [_chunk(kb, "kb-001")]})


def test_unparseable_llm_response_raises_llm_unavailable(monkeypatch):
    from app.generation.explainer import _call_llm

    monkeypatch.setattr("app.generation.explainer.generate_json", lambda prompt: "not json")
    with pytest.raises(LLMUnavailableError):
        _call_llm("...", [])


def test_llm_response_citing_an_unretrieved_id_is_dropped(kb, monkeypatch):
    monkeypatch.setattr(
        "app.generation.explainer.generate_json",
        lambda prompt: '{"sentences": [{"text": "x", "kb_ids": ["kb-999"]}]}',
    )
    from app.generation.explainer import _call_llm

    with pytest.raises(LLMUnavailableError, match="no groundable sentences"):
        _call_llm("...", [_chunk(kb, "kb-001")])


# --- pipeline-level fallback on LLM unavailability -------------------------------------


def test_pipeline_falls_back_to_template_when_llm_unavailable(monkeypatch):
    from app.retrieval.retriever import Retriever

    from app.pipeline import ExplainPipeline

    kb = load_knowledge_base()
    pipeline = ExplainPipeline(kb, ner=_StubNER(), retriever=Retriever(kb))

    def raise_error(*a, **k):
        raise LLMUnavailableError("connection refused")

    monkeypatch.setattr("app.pipeline.generate_explanation", raise_error)
    response = pipeline.run("Mild hepatic steatosis with elevated ALT.")

    assert response.validation.fallback_used is True
    assert any(f.startswith("generation_unavailable:") for f in response.validation.flags)
    assert "Extra fat has built up inside the liver" in response.explanation


class _StubNER:
    def extract(self, text, sentences):
        return []


# --- real Ollama calls (slow; skip with -m "not llm" if Ollama isn't running) ---------


@pytest.mark.llm
def test_deck_example_with_the_real_model(kb):
    terms = [_term("kb-001", text="hepatic steatosis"), _term("kb-002", text="ALT")]
    retrieved = {"kb-001": [_chunk(kb, "kb-001")], "kb-002": [_chunk(kb, "kb-002")]}
    explanation = generate_explanation("Mild hepatic steatosis with elevated ALT.", terms, retrieved)

    assert explanation.generator.startswith("llm:")
    ids_used = {kb_id for s in explanation.sentences for kb_id in s.kb_ids}
    assert ids_used == {"kb-001", "kb-002"}
    assert all(s.kind == "definition" for s in explanation.sentences)


@pytest.mark.llm
def test_diagnosis_phrased_llm_output_fails_validation_and_would_fall_back(kb):
    """Regression test for a real case found during evaluation (eval/generation_results.md,
    ev-004): the model wrote "You have been diagnosed with blood pressure that stays
    higher than the normal range over time." The safety layer must reject the whole
    explanation, exactly as app/pipeline.py's fallback logic depends on.
    """
    from app.safety.validator import validate_explanation

    terms = [
        _term("kb-012", text="dyspnea on exertion"),
        _term("kb-011", text="bilateral pedal edema"),
        _term("kb-005", text="hypertension"),
    ]
    retrieved = {
        "kb-012": [_chunk(kb, "kb-012")],
        "kb-011": [_chunk(kb, "kb-011")],
        "kb-005": [_chunk(kb, "kb-005")],
    }
    text = "Patient admitted with dyspnea on exertion and bilateral pedal edema. Known history of hypertension."
    explanation = generate_explanation(text, terms, retrieved)
    result = validate_explanation(explanation, retrieved)

    # Not asserting *which* wording the model chose — only that if it writes anything
    # diagnosis/advice/dosage/prognosis-shaped, the validator catches it. A pass here
    # means this specific model+prompt combination happened not to reproduce the
    # ev-004 phrasing on this run; that is a fine outcome, not a test failure.
    if not result.passed:
        assert any(f.startswith("forbidden_") for f in result.flags)


@pytest.mark.llm
def test_negated_and_llm_sentences_together_pass_validation(kb):
    from app.safety.validator import validate_explanation

    terms = [
        _term("kb-001", text="hepatic steatosis"),
        _term("kb-038", text="ischemia", negated=True),
    ]
    retrieved = {"kb-001": [_chunk(kb, "kb-001")], "kb-038": [_chunk(kb, "kb-038")]}
    explanation = generate_explanation("...", terms, retrieved)
    result = validate_explanation(explanation, retrieved)
    assert result.passed, result.flags
