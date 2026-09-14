"""Step 6: provenance / audit logging. Off by default; opt-in via AUDIT_LOG_PATH."""

from __future__ import annotations

import json

import pytest

from app.generation.template import template_explanation
from app.models import FlaggedTerm
from app.retrieval.knowledge_base import load_knowledge_base
from app.retrieval.retriever import Retriever
from app.safety.provenance import record_provenance
from app.safety.validator import validate_explanation


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def _term(kb_id, status="explained", text="term"):
    return FlaggedTerm(
        text=text, label="DISEASE", start=0, end=len(text), sentence_index=0,
        kb_id=kb_id, status=status,
    )


def _sample(kb):
    terms = [_term("kb-001", text="hepatic steatosis")]
    from app.models import RetrievedChunk

    entry = kb.get("kb-001")
    retrieved = {
        "kb-001": [
            RetrievedChunk(
                kb_id=entry.id, term=entry.term, plain_definition=entry.plain_definition,
                clinical_context=entry.clinical_context, source_citation=entry.source_citation,
                score=1.0,
            )
        ]
    }
    explanation = template_explanation(terms, retrieved)
    validation = validate_explanation(explanation, retrieved)
    return terms, retrieved, explanation, validation


def test_disabled_by_default_returns_none(kb):
    terms, retrieved, explanation, validation = _sample(kb)
    record_id = record_provenance(
        original_text="Mild hepatic steatosis.", terms=terms, retrieved=retrieved,
        explanation=explanation, validation=validation, fallback_used=False,
        generation_error=None, timings_ms={}, path=None,
    )
    assert record_id is None


def test_enabled_writes_one_json_line(kb, tmp_path):
    terms, retrieved, explanation, validation = _sample(kb)
    log_path = tmp_path / "audit.jsonl"

    record_id = record_provenance(
        original_text="Mild hepatic steatosis.", terms=terms, retrieved=retrieved,
        explanation=explanation, validation=validation, fallback_used=False,
        generation_error=None, timings_ms={"preprocessing": 1.2}, path=str(log_path),
    )

    assert record_id is not None
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["id"] == record_id
    assert record["original_text"] == "Mild hepatic steatosis."
    assert record["entities"][0]["text"] == "hepatic steatosis"
    assert record["retrieved_chunk_ids"] == ["kb-001"]
    assert record["validation_passed"] is True
    assert record["fallback_used"] is False
    assert record["timings_ms"] == {"preprocessing": 1.2}
    assert "timestamp" in record


def test_appends_rather_than_overwrites(kb, tmp_path):
    terms, retrieved, explanation, validation = _sample(kb)
    log_path = tmp_path / "audit.jsonl"

    for _ in range(3):
        record_provenance(
            original_text="x", terms=terms, retrieved=retrieved, explanation=explanation,
            validation=validation, fallback_used=False, generation_error=None,
            timings_ms={}, path=str(log_path),
        )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    ids = {json.loads(line)["id"] for line in lines}
    assert len(ids) == 3  # each record gets a distinct id


def test_creates_parent_directories(kb, tmp_path):
    terms, retrieved, explanation, validation = _sample(kb)
    log_path = tmp_path / "nested" / "dir" / "audit.jsonl"

    record_provenance(
        original_text="x", terms=terms, retrieved=retrieved, explanation=explanation,
        validation=validation, fallback_used=False, generation_error=None,
        timings_ms={}, path=str(log_path),
    )
    assert log_path.exists()


def test_write_failure_is_swallowed_not_raised(kb, tmp_path, monkeypatch, caplog):
    """A broken audit log must never take the product down."""
    terms, retrieved, explanation, validation = _sample(kb)

    def broken_mkdir(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.mkdir", broken_mkdir)
    record_id = record_provenance(
        original_text="x", terms=terms, retrieved=retrieved, explanation=explanation,
        validation=validation, fallback_used=False, generation_error=None,
        timings_ms={}, path=str(tmp_path / "nested" / "audit.jsonl"),
    )
    assert record_id is None


def test_generation_error_is_recorded_when_present(kb, tmp_path):
    terms, retrieved, explanation, validation = _sample(kb)
    log_path = tmp_path / "audit.jsonl"

    record_provenance(
        original_text="x", terms=terms, retrieved=retrieved, explanation=explanation,
        validation=validation, fallback_used=True, generation_error="llama3 via Ollama: timeout",
        timings_ms={}, path=str(log_path),
    )
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["generation_error"] == "llama3 via Ollama: timeout"
    assert record["fallback_used"] is True


# --- pipeline wiring ---------------------------------------------------------------


@pytest.mark.llm  # deck example has explained, un-negated terms -> a real LLM call
def test_pipeline_logs_a_provenance_record_when_enabled(monkeypatch, tmp_path):
    from app.pipeline import ExplainPipeline

    kb = load_knowledge_base()
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log_path))
    monkeypatch.setattr("app.pipeline.AUDIT_LOG_PATH", str(log_path))

    pipeline = ExplainPipeline(kb, ner=_StubNER(), retriever=Retriever(kb))
    response = pipeline.run("Mild hepatic steatosis with elevated ALT.")

    assert response.provenance_id is not None
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == response.provenance_id


@pytest.mark.llm  # deck example has explained, un-negated terms -> a real LLM call
def test_pipeline_does_not_log_when_disabled(monkeypatch):
    from app.pipeline import ExplainPipeline

    kb = load_knowledge_base()
    monkeypatch.setattr("app.pipeline.AUDIT_LOG_PATH", None)
    pipeline = ExplainPipeline(kb, ner=_StubNER(), retriever=Retriever(kb))
    response = pipeline.run("Mild hepatic steatosis with elevated ALT.")
    assert response.provenance_id is None


class _StubNER:
    def extract(self, text, sentences):
        return []
