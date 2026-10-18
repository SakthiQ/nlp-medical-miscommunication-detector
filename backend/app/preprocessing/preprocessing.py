"""Slide stage 1 — text preprocessing.

Sentence segmentation uses pySBD, a rule-based segmenter that already handles titles
("Dr."), decimals ("1.8") and units ("mg/dL"). A post-pass re-joins the pieces it splits
after clinical abbreviations ("Pt. c/o dyspnea"). scispaCy is not used: it requires
spaCy <3.8, which has no Python 3.14 build.

Negation detection uses negspacy's NegEx implementation with the clinical termset.
Stopwords are never removed: "no" and "not" change the meaning of a finding.
"""

from __future__ import annotations

from collections import defaultdict

import negspacy.negation  # noqa: F401  (registers the "negex" spaCy factory)
import pysbd
import spacy
from negspacy.termsets import termset
from spacy.util import filter_spans

from app.models import FlaggedTerm, Sentence

IMPLEMENTATION = "real"

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)

# A piece ending in one of these, followed by a piece starting lowercase, is one sentence.
CLINICAL_ABBREVIATIONS = frozenset({
    "pt.", "pts.", "hx.", "dx.", "rx.", "tx.", "sx.", "fx.",
    "mg.", "mcg.", "ml.", "gm.", "approx.", "vs.", "sig.",
    "yr.", "yrs.", "wk.", "wks.", "mo.", "mos.",
})


def preprocess_report(text: str) -> list[Sentence]:
    spans: list[list[int]] = []
    cursor = 0
    for piece in _SEGMENTER.segment(text):
        piece = piece.strip()
        if not piece:
            continue
        # Locate each piece ourselves, moving forward, so repeated sentences
        # ("ALT elevated. ALT elevated.") get their own offsets.
        start = text.find(piece, cursor)
        if start == -1:
            remainder = text[cursor:].strip()
            if remainder:
                start = text.index(remainder, cursor)
                spans.append([start, start + len(remainder)])
            break
        cursor = start + len(piece)
        if spans and _continues_previous(text, spans[-1], start):
            spans[-1][1] = cursor
        else:
            spans.append([start, cursor])

    return [
        Sentence(index=index, text=text[start:end], start=start, end=end)
        for index, (start, end) in enumerate(spans)
    ]


def _continues_previous(text: str, previous: list[int], next_start: int) -> bool:
    last_token = text[previous[0]:previous[1]].split()[-1].lower()
    return last_token in CLINICAL_ABBREVIATIONS and text[next_start].islower()


# --- negation ---------------------------------------------------------------------

NEGATION_IMPLEMENTATION = "real"

# NegEx's clinical termset misses these common radiology phrasings.
_EXTRA_FOLLOWING_NEGATIONS = {
    "absent", "is absent", "are absent", "not seen", "not identified",
    "not visualized", "not demonstrated", "none seen",
}


def _is_uncertainty_cue(phrase: str) -> bool:
    """'Rule out X' means the doctor is checking for X — uncertain, not negated.

    Treating it as negated would tell a patient their report says X is absent.
    'X was ruled out' is a genuine negation and is kept.
    """
    phrase = phrase.lower()
    return ("rule" in phrase and "ruled" not in phrase) or "r/o" in phrase


class NegationDetector:
    def __init__(self) -> None:
        patterns = termset("en_clinical").get_patterns()
        patterns["preceding_negations"] = [
            p for p in patterns["preceding_negations"] if not _is_uncertainty_cue(p)
        ]
        patterns["following_negations"] = sorted(
            {p for p in patterns["following_negations"] if not _is_uncertainty_cue(p)}
            | _EXTRA_FOLLOWING_NEGATIONS
        )
        self._nlp = spacy.blank("en")
        self._negex = self._nlp.add_pipe(
            "negex", config={"neg_termset": patterns, "ent_types": []}
        )

    def annotate(self, sentences: list[Sentence], terms: list[FlaggedTerm]) -> list[FlaggedTerm]:
        """Return the terms with `negated` set, judged within each term's own sentence."""
        result = list(terms)
        by_sentence: dict[int, list[int]] = defaultdict(list)
        for position, term in enumerate(terms):
            by_sentence[term.sentence_index].append(position)

        for sentence in sentences:
            positions = by_sentence.get(sentence.index)
            if not positions:
                continue

            doc = self._nlp.make_doc(sentence.text)
            # The sentence is already segmented; stop spaCy from re-splitting it.
            for token in doc:
                token.is_sent_start = token.i == 0

            owners: dict[tuple[int, int], int] = {}
            spans = []
            for position in positions:
                term = terms[position]
                span = doc.char_span(
                    term.start - sentence.start,
                    term.end - sentence.start,
                    label=term.label,
                    alignment_mode="expand",
                )
                if span is not None:
                    spans.append(span)
                    owners.setdefault((span.start, span.end), position)

            doc.ents = filter_spans(spans)
            doc = self._negex(doc)
            for ent in doc.ents:
                position = owners.get((ent.start, ent.end))
                if position is not None:
                    result[position] = result[position].model_copy(
                        update={"negated": bool(ent._.negex)}
                    )

        return result
