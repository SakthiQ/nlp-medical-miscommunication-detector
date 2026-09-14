"""Step 4: embedding retrieval. Uses the real model from the local Hugging Face cache."""

import json
from pathlib import Path

import pytest

from app.models import FlaggedTerm
from app.retrieval.knowledge_base import load_knowledge_base
from app.retrieval.retriever import Retriever

QUERIES = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "retrieval_queries.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


@pytest.fixture(scope="module")
def retriever(kb):
    return Retriever(kb)


def _term(kb_id, status="explained", text="term"):
    return FlaggedTerm(
        text=text, label="DISEASE", start=0, end=len(text), sentence_index=0,
        kb_id=kb_id, status=status,
    )


def test_linked_term_gets_only_its_exact_entry(retriever):
    [chunk] = retriever.for_term(_term("kb-001"))
    assert (chunk.kb_id, chunk.score, chunk.match) == ("kb-001", 1.0, "exact")


def test_unvetted_term_gets_no_explanation_evidence(retriever):
    # However close its nearest entry is, an unlinked term is never given evidence.
    assert retriever.for_term(_term(None, "no_vetted_definition", "renal")) == []


def test_semantic_search_finds_the_deck_term(retriever):
    top = retriever.search("hepatic steatosis")[0]
    assert (top.kb_id, top.match) == ("kb-001", "semantic")


def test_paraphrase_recall_at_3(retriever):
    results = retriever.suggest([q["query"] for q in QUERIES["paraphrases"]], k=3)
    missed = [
        q["query"] for q, chunks in zip(QUERIES["paraphrases"], results)
        if q["kb_id"] not in [c.kb_id for c in chunks]
    ]
    assert missed == []


def test_unrelated_terms_can_outscore_real_paraphrases(retriever):
    """The measured reason semantic matches never explain terms: no threshold is safe."""
    paraphrase_results = retriever.suggest([q["query"] for q in QUERIES["paraphrases"]], k=3)
    correct = [
        next(c.score for c in chunks if c.kb_id == q["kb_id"])
        for q, chunks in zip(QUERIES["paraphrases"], paraphrase_results)
    ]
    unrelated_best = [chunks[0].score for chunks in retriever.suggest(QUERIES["unrelated"], k=1)]
    assert max(unrelated_best) > min(correct)


def test_common_terms_are_not_indexed(retriever, kb):
    assert all(not c.kb_id.startswith("kb-common") for c in retriever.search("fever headache nausea", k=10))
    assert len(retriever.search("anything", k=1000)) == len(kb.explainable)


def test_suggestions_are_ordered_best_first(retriever):
    for chunks in retriever.suggest(["renal", "ECG"], k=3):
        scores = [c.score for c in chunks]
        assert len(chunks) == 3 and scores == sorted(scores, reverse=True)


def test_blank_queries_return_nothing(retriever):
    assert retriever.search("   ") == []
    assert retriever.suggest([]) == []
