"""Measure real translation latency, and the back-translation signal's own reliability.

    python -m eval.translation_eval

This does NOT run on every request (see app/i18n/translator.py for why: round-tripping
through the same model roughly doubles an already-severe CPU latency, for a signal the
patient never sees). It exists here, offline, so the real cost and the real behaviour of
the QA signal are both documented with numbers instead of assumed.

Two things are measured, for Hindi (the only language this build enables) and — for
comparison only, NOT enabled in the app — Telugu and Malayalam, on real generated
explanations from the 18-report evaluation corpus:

1. Latency: forward translation, and round-trip (forward + back).
2. Back-translation similarity to the original English, using the same sentence
   embedding model retrieval already uses. A LOW score reliably means something went
   wrong. A HIGH score does not reliably mean the forward translation was correct — see
   the specific counter-example this script's own run turns up and documents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from app.generation.explainer import generate_explanation
from app.generation.llm_client import LLMUnavailableError
from app.i18n.translator import LANGUAGES, Translator
from app.ner.preprocessing import NegationDetector, preprocess_report
from app.ner.terminology import DictionaryMatcher, identify_terms
from app.retrieval.corpus import load_corpus
from app.retrieval.knowledge_base import load_knowledge_base
from app.retrieval.retriever import Retriever

RESULTS_PATH = Path(__file__).resolve().parent / "translation_results.md"

# "hi" is enabled in the app. "te" and "ml" are measured here only, to show the same
# latency and quality picture holds beyond Hindi — see app/i18n/translator.py.
LANGS_TO_MEASURE = [
    ("hi", "hin_Deva", "Hindi"),
    ("te", "tel_Telu", "Telugu"),
    ("ml", "mal_Mlym", "Malayalam"),
]
SAMPLE_SIZE = 3  # sentences measured per language; each one costs ~15-50s round-trip


@dataclass
class Row:
    language: str
    english: str
    forward: str
    back: str
    forward_s: float
    back_s: float
    similarity: float


@dataclass
class Result:
    rows: list[Row] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _sample_sentences() -> list[str]:
    """Sentences measured, reused across languages for a fair comparison.

    Two sources, deliberately: the app's own template wording literally quotes the
    clinical term ('Your report mentions "hepatic steatosis"...' — see
    app/generation/template.py), which is what earlier ad-hoc testing found is where a
    general-purpose translation model is most likely to corrupt a specialised term. LLM
    paraphrases, by contrast, often drop the literal term name entirely and are easier to
    translate. Both are real `final_text` content the translator actually receives, so
    both are measured — the template ones are not skipped just because they're fixed.
    """
    kb = load_knowledge_base()
    from app.generation.template import template_explanation
    from app.models import FlaggedTerm

    template_term = FlaggedTerm(
        text="hepatic steatosis", label="DISEASE", start=0, end=18, sentence_index=0,
        kb_id="kb-001", status="explained",
    )
    entry = kb.get("kb-001")
    from app.models import RetrievedChunk

    template_retrieved = {
        "kb-001": [
            RetrievedChunk(
                kb_id=entry.id, term=entry.term, plain_definition=entry.plain_definition,
                clinical_context=entry.clinical_context, source_citation=entry.source_citation,
                score=1.0,
            )
        ]
    }
    sentences = [
        s.text for s in template_explanation([template_term], template_retrieved).sentences
    ]

    matcher = DictionaryMatcher(kb)
    negation = NegationDetector()
    retriever = Retriever(kb)

    for snippet in load_corpus().snippets:
        if len(sentences) >= SAMPLE_SIZE:
            break
        segments = preprocess_report(snippet.text)
        terms = negation.annotate(segments, identify_terms(matcher.match(segments), [], kb))
        retrieved = {}
        for term in terms:
            if term.kb_id and term.kb_id not in retrieved:
                chunks = retriever.for_term(term)
                if chunks:
                    retrieved[term.kb_id] = chunks
        try:
            explanation = generate_explanation(snippet.text, terms, retrieved)
        except LLMUnavailableError:
            continue
        for sentence in explanation.sentences:
            if sentence.kind == "definition" and len(sentences) < SAMPLE_SIZE:
                sentences.append(sentence.text)

    return sentences


def evaluate() -> Result:
    kb = load_knowledge_base()
    embedder = Retriever(kb)  # reused only for its embedding model, not retrieval
    translator = Translator()
    result = Result()

    sentences = _sample_sentences()
    if not sentences:
        result.errors.append("Could not generate any sample sentences (is Ollama running?)")
        return result

    for lang_code, nllb_code, name in LANGS_TO_MEASURE:
        for text in sentences:
            t = time.perf_counter()
            forward = translator._pipe(text, src_lang="eng_Latn", tgt_lang=nllb_code, max_length=800)[0][
                "translation_text"
            ]
            forward_s = time.perf_counter() - t

            t = time.perf_counter()
            back = translator._pipe(forward, src_lang=nllb_code, tgt_lang="eng_Latn", max_length=800)[0][
                "translation_text"
            ]
            back_s = time.perf_counter() - t

            original_vec = embedder.embed([text])[0]
            back_vec = embedder.embed([back])[0]
            similarity = float((original_vec * back_vec).sum())

            result.rows.append(Row(name, text, forward, back, forward_s, back_s, similarity))

    return result


def render(result: Result) -> str:
    if not result.rows:
        return "# Translation results\n\n" + "\n".join(f"- {e}" for e in result.errors) + "\n"

    forward_times = [r.forward_s for r in result.rows]
    round_trip_times = [r.forward_s + r.back_s for r in result.rows]
    similarities = [r.similarity for r in result.rows]

    lines = [
        "# Translation results",
        "",
        "Model `facebook/nllb-200-distilled-600M`, CPU. Real generated sentences from the "
        "18-report evaluation corpus. Regenerate with `python -m eval.translation_eval`.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Forward translation (min / median / max) | {min(forward_times):.1f}s / "
        f"{sorted(forward_times)[len(forward_times) // 2]:.1f}s / {max(forward_times):.1f}s |",
        f"| Round trip, forward + back (min / median / max) | {min(round_trip_times):.1f}s / "
        f"{sorted(round_trip_times)[len(round_trip_times) // 2]:.1f}s / {max(round_trip_times):.1f}s |",
        f"| Back-translation similarity to original (min / median) | {min(similarities):.3f} / "
        f"{sorted(similarities)[len(similarities) // 2]:.3f} |",
        "",
        "**Only Hindi is enabled in the app.** Telugu and Malayalam are measured here for "
        "comparison, not built — see app/i18n/translator.py for the latency reasoning.",
        "",
        "## Every measured pair",
        "",
        "| Language | Forward (s) | Back (s) | Similarity | English | Forward translation | Back-translation |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
        *[
            f"| {r.language} | {r.forward_s:.1f} | {r.back_s:.1f} | {r.similarity:.3f} | "
            f"{r.english} | {r.forward} | {r.back} |"
            for r in result.rows
        ],
        "",
        "## Why a high similarity score is not proof of a correct translation",
        "",
        "Round-tripping a specialised clinical term through a general-purpose translation "
        "model can corrupt it in a way that still round-trips to something plausible. "
        "Look for any row above where the English mentions \"steatosis\" (fat in the "
        "liver) and the back-translation says \"stenosis\" (a narrowing — a different "
        "condition) instead: if present, it shows the back-translation score alone would "
        "have missed a clinically meaningful error, because the sentence *shape* survived "
        "the round trip even though a specific medical fact did not. This is exactly why "
        "this check is offline and advisory, not a live gate — the same reasoning as the "
        "semantic grounding check in step 6 (eval/semantic_grounding_results.md).",
        "",
    ]
    if result.errors:
        lines += ["## Errors during this run", "", *[f"- {e}" for e in result.errors], ""]
    return "\n".join(lines)


def main() -> None:
    report = render(evaluate())
    RESULTS_PATH.write_text(report, encoding="utf-8")
    # Non-Latin script (Hindi/Telugu/Malayalam) can't print on a Windows console using
    # the default cp1252 codepage; the file (UTF-8, written above) is what matters.
    print(report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nFull output with native script written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
