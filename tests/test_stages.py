"""Unit tests for individual pipeline stages."""

import pytest

from app.generation.template import template_explanation
from app.models import Entity, Explanation, ExplanationSentence, RetrievedChunk
from app.ner.preprocessing import preprocess_report
from app.ner.terminology import DictionaryMatcher, identify_terms
from app.retrieval.corpus import load_corpus
from app.retrieval.knowledge_base import load_knowledge_base
from app.safety.validator import validate_explanation


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


@pytest.fixture(scope="module")
def matcher(kb):
    return DictionaryMatcher(kb)


def _match(matcher, text):
    return matcher.match(preprocess_report(text))


# --- preprocessing ---------------------------------------------------------------


def test_sentences_keep_original_offsets():
    text = "  Mild hepatic steatosis.   Elevated ALT!  "
    sentences = preprocess_report(text)
    assert [s.text for s in sentences] == ["Mild hepatic steatosis.", "Elevated ALT!"]
    for sentence in sentences:
        assert text[sentence.start:sentence.end] == sentence.text


def test_decimal_numbers_do_not_split_sentences():
    sentences = preprocess_report("Serum creatinine 1.1 mg/dL with eGFR 78.")
    assert len(sentences) == 1


# --- dictionary matching ---------------------------------------------------------


def test_longest_alias_wins(matcher):
    [entity] = _match(matcher, "Imaging shows fatty liver disease.")
    assert entity.text == "fatty liver disease"
    assert entity.kb_id == "kb-001"


def test_hyphen_and_space_variants_both_match(matcher):
    for text in ("low-density lipoprotein", "low density lipoprotein"):
        [entity] = _match(matcher, f"Elevated {text}.")
        assert entity.kb_id == "kb-017"


def test_aliases_do_not_match_inside_words(matcher):
    # "last" must not match AST; "mIU" must not match MI (myocardial infarction).
    assert {e.kb_id for e in _match(matcher, "At last the nausea eased.")} == {"kb-common-005"}
    assert {e.kb_id for e in _match(matcher, "TSH 8.2 mIU/L.")} == {"kb-033"}


def test_dictionary_matcher_is_exact_on_the_eval_corpus(matcher):
    """Every covered gold term is found (recall), and nothing else is (precision)."""
    for snippet in load_corpus().snippets:
        found = {(e.start, e.end, e.kb_id) for e in _match(matcher, snippet.text)}
        expected = set()
        for term in snippet.covered_terms:
            start = snippet.text.index(term.surface)
            expected.add((start, start + len(term.surface), term.kb_id))
        assert found == expected, snippet.id


# --- terminology identification --------------------------------------------------


def _ner_entity(text, start):
    return Entity(
        text=text, label="DISEASE", start=start, end=start + len(text),
        sentence_index=0, confidence=0.9, source="ner",
    )


def test_common_terms_are_matched_but_not_flagged(kb, matcher):
    entities = _match(matcher, "Fever and hepatic steatosis.")
    terms = identify_terms(entities, [], kb)
    assert [t.kb_id for t in terms] == ["kb-001"]


def test_ner_term_outside_kb_becomes_a_gap(kb):
    [term] = identify_terms([], [_ner_entity("consolidation", 10)], kb)
    assert term.status == "no_vetted_definition"
    assert term.kb_id is None


def test_dictionary_beats_overlapping_ner_span(kb, matcher):
    text = "Mild hepatic steatosis."
    ner = [_ner_entity("hepatic steatosis noted", 5)]
    terms = identify_terms(_match(matcher, text), ner, kb)
    assert [(t.text, t.status) for t in terms] == [("hepatic steatosis", "explained")]


def test_ner_hit_on_a_common_term_is_not_flagged(kb):
    assert identify_terms([], [_ner_entity("fever", 0)], kb) == []


# --- generation template ---------------------------------------------------------


def _chunk(kb, kb_id):
    entry = kb.get(kb_id)
    return RetrievedChunk(
        kb_id=entry.id, term=entry.term, plain_definition=entry.plain_definition,
        clinical_context=entry.clinical_context, source_citation=entry.source_citation,
        score=1.0,
    )


def test_template_explains_each_entry_once(kb, matcher):
    entities = _match(matcher, "ALT raised. Repeat ALT also raised.")
    terms = identify_terms(entities, [], kb)
    explanation = template_explanation(terms, {"kb-002": [_chunk(kb, "kb-002")]})
    assert len(explanation.sentences) == 1
    assert explanation.sentences[0].kb_ids == ["kb-002"]


# --- validation ------------------------------------------------------------------


def _explanation(text, kb_ids=("kb-001",), kind="definition"):
    return Explanation(
        sentences=[ExplanationSentence(text=text, kb_ids=list(kb_ids), kind=kind)],
        generator="test",
    )


@pytest.mark.parametrize(
    ("text", "flag"),
    [
        ("You should start metformin.", "forbidden_advice"),
        ("You have fatty liver disease.", "forbidden_diagnosis"),
        ("Take 500 mg twice a day.", "forbidden_dosage"),
        ("We recommend a low-fat diet.", "forbidden_advice"),
        ("This will go away on its own.", "forbidden_prognosis"),
    ],
)
def test_forbidden_phrasing_is_flagged(kb, text, flag):
    result = validate_explanation(_explanation(text), {"kb-001": [_chunk(kb, "kb-001")]})
    assert not result.passed
    assert any(f.startswith(flag) for f in result.flags)


def test_lab_units_are_not_mistaken_for_doses(kb):
    result = validate_explanation(
        _explanation("Your glucose was 142 mg/dL on the report."),
        {"kb-001": [_chunk(kb, "kb-001")]},
    )
    assert result.passed, result.flags


def test_definition_citing_unretrieved_entry_is_ungrounded(kb):
    result = validate_explanation(_explanation("Some claim.", kb_ids=["kb-999"]), {})
    assert not result.passed
    assert result.flags[0].startswith("ungrounded")


def test_gap_sentence_must_not_cite_sources(kb):
    result = validate_explanation(
        _explanation("Ask your doctor.", kb_ids=["kb-001"], kind="gap"),
        {"kb-001": [_chunk(kb, "kb-001")]},
    )
    assert not result.passed
