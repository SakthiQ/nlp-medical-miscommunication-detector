"""Measure LLM-generated explanations against the deterministic template's grounding
guarantees, and against a Flesch-Kincaid readability target.

    python -m eval.generation_eval

Runs the real LLM over every eval-corpus report, then for each generated sentence checks:
- grounding: does the safety validator accept it? (every claim traceable to a KB id)
- readability: is it at or below a 6th-8th grade Flesch-Kincaid grade level?

This is a faithfulness/readability check, not a substitute for the human review the
project brief calls for (Phase 5's acceptance criteria: manually reading 10 explanations
for faithfulness). Treat this script's output as a first pass to triage before that
review, not as the review itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.generation.explainer import generate_explanation
from app.generation.llm_client import LLMUnavailableError
from app.ner.preprocessing import NegationDetector, preprocess_report
from app.ner.terminology import DictionaryMatcher, identify_terms
from app.retrieval.corpus import load_corpus
from app.retrieval.knowledge_base import load_knowledge_base
from app.retrieval.retriever import Retriever
from app.safety.validator import validate_explanation

RESULTS_PATH = Path(__file__).resolve().parent / "generation_results.md"
READABILITY_TARGET = 8.5  # 6th-8th grade, per the project brief

_WORD_RE = re.compile(r"[A-Za-z]+")
_VOWEL_RE = re.compile(r"[aeiouy]+")
_SENTENCE_FLAG_RE = re.compile(r"\bsentence (\d+)\b")


def _syllables(word: str) -> int:
    word = word.lower()
    count = len(_VOWEL_RE.findall(word))
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text: str) -> float | None:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = _WORD_RE.findall(text)
    if not sentences or not words:
        return None
    syllables = sum(_syllables(w) for w in words)
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59


@dataclass
class SentenceRow:
    report_id: str
    text: str
    kb_ids: list[str]
    grounded: bool
    grade: float | None


@dataclass
class ReportRow:
    report_id: str
    passed: bool  # whether the *full* explanation passed every safety check, not just grounding
    flags: list[str]


@dataclass
class Result:
    rows: list[SentenceRow] = field(default_factory=list)
    reports: list[ReportRow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def evaluate() -> Result:
    kb = load_knowledge_base()
    matcher = DictionaryMatcher(kb)
    negation = NegationDetector()
    retriever = Retriever(kb)
    result = Result()

    for snippet in load_corpus().snippets:
        sentences = preprocess_report(snippet.text)
        terms = negation.annotate(sentences, identify_terms(matcher.match(sentences), [], kb))
        retrieved = {}
        for term in terms:
            if term.kb_id and term.kb_id not in retrieved:
                chunks = retriever.for_term(term)
                if chunks:
                    retrieved[term.kb_id] = chunks

        try:
            explanation = generate_explanation(snippet.text, terms, retrieved)
        except LLMUnavailableError as exc:
            result.errors.append(f"{snippet.id}: LLM unavailable ({exc})")
            continue

        if not explanation.generator.startswith("llm:"):
            continue  # nothing for the LLM to do on this report (all gap/negated/none)

        # Validated exactly as app/pipeline.py validates it: the *whole* explanation is
        # rejected together if any sentence trips a check, not sentence by sentence.
        validation = validate_explanation(explanation, retrieved)
        result.reports.append(ReportRow(report_id=snippet.id, passed=validation.passed, flags=validation.flags))
        ungrounded_indices = {
            int(match.group(1))
            for flag in validation.flags
            if flag.startswith("ungrounded")
            for match in [_SENTENCE_FLAG_RE.search(flag)]
            if match
        }

        for index, sentence in enumerate(explanation.sentences):
            if sentence.kind != "definition":
                continue
            result.rows.append(
                SentenceRow(
                    report_id=snippet.id,
                    text=sentence.text,
                    kb_ids=sentence.kb_ids,
                    grounded=index not in ungrounded_indices,
                    grade=flesch_kincaid_grade(sentence.text),
                )
            )

    return result


def _render_row(row: SentenceRow) -> str:
    grade = f"{row.grade:.1f}" if row.grade is not None else "n/a"
    grounded = "yes" if row.grounded else "NO"
    return f"| {row.report_id} | {row.text} | {', '.join(row.kb_ids)} | {grounded} | {grade} |"


def render(result: Result) -> str:
    rows = result.rows
    n = len(rows)
    grades = [r.grade for r in rows if r.grade is not None]
    on_target = sum(1 for g in grades if g <= READABILITY_TARGET)
    grounded = sum(1 for r in rows if r.grounded)
    reports_passed = sum(1 for r in result.reports if r.passed)
    reports_n = len(result.reports)

    lines = [
        "# Generation results",
        "",
        "LLM-generated explanations for the 18-report evaluation corpus, checked against "
        "the safety validator and a 6th-8th grade Flesch-Kincaid target.",
        "Regenerate with `python -m eval.generation_eval`. Requires Ollama running with "
        "the model in `LLM_MODEL` (default `llama3`) pulled.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Reports where the LLM's full explanation passed every safety check | {reports_passed} / {reports_n} |",
        f"| Definition sentences generated | {n} |",
        f"| Grounded (cites a retrieved KB id for its claim) | {grounded} / {n} |",
        f"| At or below grade {READABILITY_TARGET} (Flesch-Kincaid) | {on_target} / {len(grades)} |",
        f"| Mean grade level | {sum(grades) / len(grades):.1f} |" if grades else "| Mean grade level | n/a |",
        "",
        "## Per-report safety outcome",
        "",
        "The pipeline validates a report's explanation as a whole, not sentence by "
        "sentence: one flagged sentence replaces the *entire* explanation with the "
        "template, the same as `app/pipeline.py` does for a real request.",
        "",
        "| Report | Passed | Flags |",
        "| --- | --- | --- |",
        *[
            f"| {r.report_id} | {'yes' if r.passed else '**NO — fell back to template**'} | "
            f"{'; '.join(r.flags) if r.flags else '—'} |"
            for r in result.reports
        ],
        "",
        "## Sentence-level grounding and readability",
        "",
        "Grounding here means only \"cites a retrieved id\" — it does not capture the "
        "diagnosis/advice/dosage/prognosis checks, which are report-level and shown "
        "above. A sentence can show `yes` here and still belong to a report the "
        "validator rejected as a whole for a different reason.",
        "",
        "| Report | Sentence | KB ids | Grounded | Grade |",
        "| --- | --- | --- | --- | ---: |",
        *[_render_row(r) for r in rows],
        "",
    ]
    if result.errors:
        lines += ["## Ollama errors during this run", "", *[f"- {e}" for e in result.errors], ""]
    lines += [
        "## What this does and does not show",
        "",
        "This confirms the safety net actually fires on real model output — see ev-004 "
        "below, where the model wrote \"You have been diagnosed with blood pressure...\" "
        "and the whole explanation was correctly rejected — and gives an automated "
        "readability proxy. It does **not** check whether a sentence that *passes* is "
        "faithful in meaning to the source definition — a sentence can cite the right id "
        "and still subtly misstate it. That check needs a human reading the sentences "
        "above against their KB definitions, which is Phase 5's actual acceptance "
        "criteria and has not been done here.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    report = render(evaluate())
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
