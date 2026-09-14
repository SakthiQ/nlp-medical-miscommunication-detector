"""Measure term detection against the hand-annotated evaluation corpus.

    python -m eval.ner_eval

Prints the results and writes eval/ner_results.md. A predicted span counts as a match
when it overlaps a gold span ("overlap"), or covers exactly the same characters
("exact", case-insensitive). Three views are reported:

1. NER model alone, against every gold term.
2. Coverage-gap detection: gold terms with no KB entry that surface as
   "no vetted definition". This is the NER stage's actual job in the system.
3. The full terminology stage (dictionary + NER), against the gold terms a patient
   should see flagged (commonly understood terms excluded).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.ner.biobert_ner import MIN_CONFIDENCE, MedicalNER
from app.ner.preprocessing import preprocess_report
from app.ner.terminology import DictionaryMatcher, identify_terms
from app.retrieval.corpus import load_corpus
from app.retrieval.knowledge_base import load_knowledge_base

RESULTS_PATH = Path(__file__).resolve().parent / "ner_results.md"


@dataclass
class Score:
    gold: int = 0
    predicted: int = 0
    gold_found_overlap: int = 0
    gold_found_exact: int = 0
    predicted_correct: int = 0
    misses: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)

    def recall(self, exact: bool = False) -> float:
        found = self.gold_found_exact if exact else self.gold_found_overlap
        return found / self.gold if self.gold else 0.0

    def precision(self) -> float:
        return self.predicted_correct / self.predicted if self.predicted else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if p + r else 0.0


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _score(score: Score, snippet_id: str, text: str, gold, predicted, count_precision: bool = True) -> None:
    """gold / predicted: lists of (start, end) spans."""
    score.gold += len(gold)
    for span in gold:
        if any(_overlaps(span, p) for p in predicted):
            score.gold_found_overlap += 1
            if any(text[p[0]:p[1]].lower() == text[span[0]:span[1]].lower() for p in predicted):
                score.gold_found_exact += 1
        else:
            score.misses.append(f"{snippet_id}: {text[span[0]:span[1]]}")
    if count_precision:
        score.predicted += len(predicted)
        for span in predicted:
            if any(_overlaps(span, g) for g in gold):
                score.predicted_correct += 1
            else:
                score.extras.append(f"{snippet_id}: {text[span[0]:span[1]]}")


def evaluate() -> dict[str, Score]:
    kb = load_knowledge_base()
    matcher = DictionaryMatcher(kb)
    ner = MedicalNER()
    scores = {"ner_alone": Score(), "gap_detection": Score(), "terminology_stage": Score()}

    for snippet in load_corpus().snippets:
        text = snippet.text

        def span(surface: str) -> tuple[int, int]:
            start = text.index(surface)
            return start, start + len(surface)

        sentences = preprocess_report(text)
        ner_entities = ner.extract(text, sentences)
        terms = identify_terms(matcher.match(sentences), ner_entities, kb)

        all_gold = [span(g.surface) for g in snippet.gold_terms]
        gap_gold = [span(g.surface) for g in snippet.gap_terms]
        flaggable_gold = [
            span(g.surface) for g in snippet.gold_terms
            if g.kb_id is None or not kb.get(g.kb_id).is_common
        ]

        _score(scores["ner_alone"], snippet.id, text, all_gold, [(e.start, e.end) for e in ner_entities])
        _score(
            scores["gap_detection"], snippet.id, text, gap_gold,
            [(t.start, t.end) for t in terms if t.status == "no_vetted_definition"],
            count_precision=False,
        )
        _score(scores["terminology_stage"], snippet.id, text, flaggable_gold, [(t.start, t.end) for t in terms])

    return scores


def render(scores: dict[str, Score]) -> str:
    ner, gaps, stage = scores["ner_alone"], scores["gap_detection"], scores["terminology_stage"]
    pct = lambda x: f"{x:.1%}"
    lines = [
        "# Term detection results",
        "",
        "Measured on `data/eval_corpus.json` (18 synthetic reports, hand-annotated) with "
        f"`d4data/biomedical-ner-all`, aggregation `first`, confidence threshold {MIN_CONFIDENCE}.",
        "Regenerate with `python -m eval.ner_eval`.",
        "",
        "| View | Gold terms | Recall (overlap) | Recall (exact span) | Precision | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| NER model alone | {ner.gold} | {pct(ner.recall())} | {pct(ner.recall(exact=True))} | {pct(ner.precision())} | {pct(ner.f1())} |",
        f"| Coverage-gap detection | {gaps.gold} | {pct(gaps.recall())} | {pct(gaps.recall(exact=True))} | n/a | n/a |",
        f"| Terminology stage (dictionary + NER) | {stage.gold} | {pct(stage.recall())} | {pct(stage.recall(exact=True))} | {pct(stage.precision())} | {pct(stage.f1())} |",
        "",
        "Precision here is strict: the gold labels mark only the terms a patient most needs "
        "explained, so a correct medical term the annotator left unlabelled (for example "
        "\"Urinalysis\") counts against it.",
        "",
        "## Known failure modes",
        "",
        "- **Context sensitivity.** The model's output depends on the surrounding report. "
        "\"femoral neck\" is found in the full ev-014 report but missed when its sentence is "
        "sent alone; \"hepatic steatosis\" is tagged as a disease in the deck example but only "
        "\"hepatic\" (anatomy) at the end of a long report. This is why NER runs on the whole "
        "report, and why it is not the primary term finder.",
        "- **Common abbreviations and plain clinical words are missed** (GERD, ischemia, "
        "hypertension, anemia, benign, unremarkable). The dictionary matcher covers these "
        "when they are in the knowledge base, which is why the combined stage scores far higher.",
        "- **Partial spans.** \"canal\" for \"canal stenosis\", \"infarction\" for "
        "\"myocardial infarction\". Dictionary matches win on overlap, so this only affects "
        "terms outside the knowledge base.",
        "- **Test and procedure names are flagged as unvetted terms** (Urinalysis, ECG, "
        "Endoscopy). They are real medical terms, so this is arguably correct, but the gold "
        "labels do not include them and they lower the measured precision.",
        "",
        "## NER model misses",
        "",
        *[f"- {m}" for m in ner.misses],
        "",
        "## Coverage gaps not detected",
        "",
        *([f"- {m}" for m in gaps.misses] or ["- none"]),
        "",
        "## Terms flagged that are not in the gold labels",
        "",
        *([f"- {m}" for m in stage.extras] or ["- none"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    report = render(evaluate())
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
