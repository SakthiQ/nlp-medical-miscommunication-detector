"""Step 1: sentence segmentation and negation detection."""

import pytest

from app.models import FlaggedTerm
from app.ner.preprocessing import NegationDetector, preprocess_report
from app.ner.terminology import DictionaryMatcher, identify_terms
from app.retrieval.corpus import load_corpus
from app.retrieval.knowledge_base import load_knowledge_base


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


@pytest.fixture(scope="module")
def matcher(kb):
    return DictionaryMatcher(kb)


@pytest.fixture(scope="module")
def detector():
    return NegationDetector()


def _texts(text):
    return [s.text for s in preprocess_report(text)]


# --- segmentation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Seen by Dr. Smith today. Hepatic steatosis noted.",
         ["Seen by Dr. Smith today.", "Hepatic steatosis noted."]),
        ("Take 5 mg. daily was documented. ALT elevated.",
         ["Take 5 mg. daily was documented.", "ALT elevated."]),
        ("Bone density scan shows osteopenia. T-score -1.8. No osteoporosis.",
         ["Bone density scan shows osteopenia.", "T-score -1.8.", "No osteoporosis."]),
        ("Lumbar spondylosis at L4-L5. No canal stenosis.",
         ["Lumbar spondylosis at L4-L5.", "No canal stenosis."]),
        ("eGFR 78 mL/min/1.73m2. Creatinine 1.1 mg/dL.",
         ["eGFR 78 mL/min/1.73m2.", "Creatinine 1.1 mg/dL."]),
        ("Pt. c/o dyspnea. Hx. of HTN, i.e. hypertension.",
         ["Pt. c/o dyspnea.", "Hx. of HTN, i.e. hypertension."]),
        ("IMPRESSION:\n1. Mild hepatic steatosis.\n2. No hydronephrosis.",
         ["IMPRESSION:", "1. Mild hepatic steatosis.", "2. No hydronephrosis."]),
    ],
)
def test_clinical_sentence_segmentation(text, expected):
    assert _texts(text) == expected


def test_repeated_sentences_get_distinct_offsets():
    text = "ALT elevated. ALT elevated."
    sentences = preprocess_report(text)
    assert [(s.start, s.end) for s in sentences] == [(0, 13), (14, 27)]


def test_corpus_offsets_always_point_at_the_original_text():
    for snippet in load_corpus().snippets:
        for sentence in preprocess_report(snippet.text):
            assert snippet.text[sentence.start:sentence.end] == sentence.text, snippet.id


# --- negation --------------------------------------------------------------------


def _negation(kb, matcher, detector, text):
    sentences = preprocess_report(text)
    terms = identify_terms(matcher.match(sentences), [], kb)
    return {t.text: t.negated for t in detector.annotate(sentences, terms)}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No hydronephrosis.", {"hydronephrosis": True}),
        ("No evidence of acute ischemia.", {"ischemia": True}),
        ("Denies dyspnea.", {"dyspnea": True}),
        ("Negative for proteinuria.", {"proteinuria": True}),
        ("Afebrile, no hepatomegaly.", {"hepatomegaly": True}),
        # Added following-negation phrasings that NegEx's termset misses.
        ("Mediastinal lymphadenopathy is absent.", {"lymphadenopathy": True}),
        ("Pleural effusion not seen.", {"Pleural effusion": True}),
        # Genuine negation after the fact.
        ("Cholelithiasis was ruled out.", {"Cholelithiasis": True}),
        # "Rule out" means the doctor is checking for it: uncertain, never negated.
        ("Rule out cholelithiasis.", {"cholelithiasis": False}),
        # Negation scope ends at "but".
        ("No hydronephrosis but cholelithiasis is present.",
         {"hydronephrosis": True, "cholelithiasis": False}),
        ("Mild hepatic steatosis with elevated ALT.",
         {"hepatic steatosis": False, "ALT": False}),
    ],
)
def test_negation(kb, matcher, detector, text, expected):
    assert _negation(kb, matcher, detector, text) == expected


def test_negation_on_the_eval_corpus(kb, matcher, detector):
    """Exactly these four flagged terms in the corpus are negated, and no others."""
    negated = set()
    for snippet in load_corpus().snippets:
        for term, is_negated in _negation(kb, matcher, detector, snippet.text).items():
            if is_negated:
                negated.add((snippet.id, term))
    assert negated == {
        ("ev-007", "hydronephrosis"),
        ("ev-010", "ischemia"),
        ("ev-013", "lymphadenopathy"),
        ("ev-014", "osteoporosis"),
    }


def test_negation_is_judged_within_the_term_sentence(kb, matcher, detector):
    result = _negation(kb, matcher, detector, "No hydronephrosis. Cholelithiasis noted.")
    assert result == {"hydronephrosis": True, "Cholelithiasis": False}


def test_gap_terms_are_annotated_too(detector):
    sentences = preprocess_report("No focal consolidation.")
    gap = FlaggedTerm(
        text="consolidation", label="DISEASE", start=9, end=22, sentence_index=0,
        kb_id=None, status="no_vetted_definition",
    )
    [annotated] = detector.annotate(sentences, [gap])
    assert annotated.negated is True
