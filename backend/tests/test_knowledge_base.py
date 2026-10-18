import pytest

from app.retrieval.corpus import load_corpus
from app.retrieval.knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseError,
    load_knowledge_base,
    normalize,
)


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return load_knowledge_base()


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_kb_loads_and_is_non_trivial(kb):
    assert len(kb) >= 40


def test_every_entry_has_a_citation(kb):
    for entry in kb.entries:
        assert entry.source_citation.strip(), f"{entry.id} is missing a citation"


def test_deck_example_terms_resolve(kb):
    steatosis = kb.lookup("hepatic steatosis")
    assert steatosis is not None
    assert steatosis.label == "DISEASE"

    alt = kb.lookup("ALT")
    assert alt is not None
    assert alt.label == "BIOMARKER"


def test_alias_lookup_is_case_and_punctuation_insensitive(kb):
    assert kb.lookup("HbA1c") is kb.lookup("hba1c")
    assert kb.lookup("hba1c") is kb.lookup("Hemoglobin A1c")


def test_common_terms_are_separated_from_explainable_ones(kb):
    assert kb.lookup("fever").is_common is True
    assert kb.lookup("headache").is_common is True
    assert kb.lookup("hepatic steatosis").is_common is False
    assert kb.explainable and kb.common
    assert len(kb.explainable) + len(kb.common) == len(kb)


def test_unknown_term_is_a_coverage_gap_not_an_error(kb):
    assert kb.lookup("pneumomediastinum") is None


def test_duplicate_alias_across_entries_is_rejected():
    from app.retrieval.knowledge_base import KBEntry

    def entry(entry_id: str, term: str) -> KBEntry:
        return KBEntry(
            id=entry_id,
            term=term,
            aliases=["shared alias"],
            label="DISEASE",
            plain_definition="d",
            clinical_context="c",
            source_citation="s",
        )

    with pytest.raises(KnowledgeBaseError, match="unambiguous"):
        KnowledgeBase(entries=[entry("a", "alpha"), entry("b", "beta")])


def test_normalize_strips_case_and_punctuation():
    assert normalize("HbA1c") == "hba1c"
    assert normalize("  Low-Density  Lipoprotein ") == "low density lipoprotein"


# --- corpus / KB cross-validation -------------------------------------------------


def test_corpus_loads(corpus):
    assert len(corpus) >= 15


def test_every_gold_kb_id_resolves(kb, corpus):
    for snippet in corpus.snippets:
        for term in snippet.covered_terms:
            assert kb.get(term.kb_id) is not None, (
                f"{snippet.id} references unknown KB id {term.kb_id}"
            )


def test_every_gold_surface_appears_in_its_snippet(corpus):
    for snippet in corpus.snippets:
        for term in snippet.gold_terms:
            assert term.surface in snippet.text, (
                f"{snippet.id}: gold surface {term.surface!r} not found in text"
            )


def test_covered_gold_surfaces_are_matchable_by_dictionary_lookup(kb, corpus):
    """The dictionary matcher must find every term we claim the KB covers."""
    for snippet in corpus.snippets:
        for term in snippet.covered_terms:
            found = kb.lookup(term.surface)
            assert found is not None, (
                f"{snippet.id}: {term.surface!r} is annotated as covered by "
                f"{term.kb_id} but no alias matches it"
            )
            assert found.id == term.kb_id, (
                f"{snippet.id}: {term.surface!r} matched {found.id}, expected {term.kb_id}"
            )


def test_gap_terms_are_genuinely_absent_from_kb(kb, corpus):
    """A term annotated as a coverage gap must not secretly be in the KB."""
    for snippet in corpus.snippets:
        for term in snippet.gap_terms:
            assert kb.lookup(term.surface) is None, (
                f"{snippet.id}: {term.surface!r} is annotated as a gap but the KB has it"
            )


def test_corpus_exercises_both_paths(corpus):
    """The corpus must contain real coverage gaps, or the gap path is never tested."""
    total_gaps = sum(len(s.gap_terms) for s in corpus.snippets)
    total_covered = sum(len(s.covered_terms) for s in corpus.snippets)
    assert total_gaps >= 5
    assert total_covered >= 40
