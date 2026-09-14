"""Step 2: medical NER. Uses the real model from the local Hugging Face cache."""

import pytest

from app.ner.biobert_ner import MedicalNER
from app.ner.preprocessing import preprocess_report
from app.ner.terminology import DictionaryMatcher, identify_terms
from app.retrieval.corpus import load_corpus
from app.retrieval.knowledge_base import load_knowledge_base

DECK_EXAMPLE = "Mild hepatic steatosis with elevated ALT."


@pytest.fixture(scope="module")
def ner():
    return MedicalNER()


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


@pytest.fixture(scope="module")
def matcher(kb):
    return DictionaryMatcher(kb)


@pytest.fixture(scope="module")
def corpus():
    return {snippet.id: snippet for snippet in load_corpus().snippets}


def _entities(ner, text):
    return ner.extract(text, preprocess_report(text))


def _terms(ner, matcher, kb, text):
    sentences = preprocess_report(text)
    return identify_terms(matcher.match(sentences), ner.extract(text, sentences), kb)


def test_deck_example_tags_hepatic_steatosis_as_disease(ner):
    found = [(e.text, e.label) for e in _entities(ner, DECK_EXAMPLE)]
    assert ("hepatic steatosis", "DISEASE") in found


def test_deck_example_alt_is_a_biomarker_after_terminology(ner, matcher, kb):
    # The model tags ALT as a TEST; the dictionary match wins and carries the KB label.
    labels = {t.text: t.label for t in _terms(ner, matcher, kb, DECK_EXAMPLE)}
    assert labels["ALT"] == "BIOMARKER"


def test_offsets_point_at_the_original_text(ner, corpus):
    for snippet in corpus.values():
        for entity in _entities(ner, snippet.text):
            assert snippet.text[entity.start:entity.end] == entity.text, snippet.id


def test_entities_fall_inside_their_assigned_sentence(ner, corpus):
    for snippet in corpus.values():
        sentences = preprocess_report(snippet.text)
        for entity in ner.extract(snippet.text, sentences):
            sentence = sentences[entity.sentence_index]
            assert sentence.start <= entity.start and entity.end <= sentence.end, snippet.id


def test_long_reports_are_covered_beyond_the_model_window(ner, corpus):
    filler = " ".join(snippet.text for snippet in corpus.values()) * 3
    text = f"{filler} Mild hepatic steatosis with elevated ALT."
    assert len(ner._pipe.tokenizer(text)["input_ids"]) > 512

    entities = _entities(ner, text)
    # Anything found after the filler proves the overlapping windows reached the end.
    # Which terms the model finds there varies with context, so no specific term is asserted.
    assert any(e.start > len(filler) for e in entities)
    for entity in entities:
        assert text[entity.start:entity.end] == entity.text


def test_low_confidence_and_descriptive_labels_are_dropped(ner, corpus):
    entities = _entities(ner, corpus["ev-007"].text)
    texts = {e.text for e in entities}
    assert "right" not in texts          # anatomy at 0.58 confidence
    assert "obstructing" not in texts    # symptom at 0.66 confidence
    assert "4 mm" not in texts           # Distance label
    assert all(e.confidence >= 0.80 for e in entities)


def test_single_character_fragments_are_dropped(ner, corpus):
    assert "T" not in {e.text for e in _entities(ner, corpus["ev-014"].text)}


def test_everyday_words_are_not_flagged(ner, corpus):
    texts = {e.text for e in _entities(ner, corpus["ev-017"].text)}
    assert "Liver" not in texts
    assert "Spleen" not in texts


@pytest.mark.parametrize(
    ("snippet_id", "gap"),
    [("ev-007", "renal"), ("ev-013", "Mediastinal"), ("ev-014", "femoral neck"), ("ev-015", "antral")],
)
def test_known_coverage_gaps_surface_as_unvetted_terms(ner, matcher, kb, corpus, snippet_id, gap):
    status = {t.text: t.status for t in _terms(ner, matcher, kb, corpus[snippet_id].text)}
    assert status.get(gap) == "no_vetted_definition"


def test_partial_ner_span_never_overrides_the_dictionary(ner, matcher, kb, corpus):
    terms = _terms(ner, matcher, kb, corpus["ev-016"].text)
    assert ("myocardial infarction", "kb-037") in [(t.text, t.kb_id) for t in terms]
    assert "infarction" not in [t.text for t in terms]


def test_empty_input_returns_nothing(ner):
    assert ner.extract("", []) == []
