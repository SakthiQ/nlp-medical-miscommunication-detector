"""Measure whether embedding similarity can tell a faithful generated sentence from a
sentence paired with the wrong-but-plausible definition — i.e., whether a semantic
grounding check could safely reject explanations on its own.

    python -m eval.semantic_grounding_eval

Runs the real LLM over the 18-report evaluation corpus (as eval/generation_eval.py does),
collects (sentence, its own cited definition) as "faithful" pairs, pairs each sentence
with a random *other* entry's definition as "mismatched" pairs, and compares the two
similarity distributions. Regenerate with `python -m eval.semantic_grounding_eval`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from app.generation.explainer import generate_explanation
from app.generation.llm_client import LLMUnavailableError
from app.ner.preprocessing import NegationDetector, preprocess_report
from app.ner.terminology import DictionaryMatcher, identify_terms
from app.retrieval.corpus import load_corpus
from app.retrieval.knowledge_base import load_knowledge_base
from app.retrieval.retriever import MODEL_NAME, Retriever
from app.safety.semantic import ADVISORY_THRESHOLD

RESULTS_PATH = Path(__file__).resolve().parent / "semantic_grounding_results.md"
RANDOM_SEED = 0


@dataclass
class Pair:
    report_id: str
    sentence: str
    kb_id: str
    score: float


@dataclass
class Result:
    faithful: list[Pair] = field(default_factory=list)
    mismatched: list[Pair] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def evaluate() -> Result:
    kb = load_knowledge_base()
    matcher = DictionaryMatcher(kb)
    negation = NegationDetector()
    retriever = Retriever(kb)
    result = Result()

    triples: list[tuple[str, str, str, str]] = []  # (report_id, sentence, correct_def, correct_id)
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
            continue

        for sentence in explanation.sentences:
            if sentence.kind == "definition" and len(sentence.kb_ids) == 1:
                entry = kb.get(sentence.kb_ids[0])
                if entry:
                    triples.append((snippet.id, sentence.text, entry.plain_definition, entry.id))

    if not triples:
        return result

    rng = random.Random(RANDOM_SEED)
    all_defs = [(entry.id, entry.plain_definition) for entry in kb.explainable]

    texts = [t[1] for t in triples]
    correct_defs = [t[2] for t in triples]
    wrong_choices = [rng.choice([d for d in all_defs if d[0] != t[3]]) for t in triples]
    wrong_defs = [d[1] for d in wrong_choices]

    text_vecs = retriever.embed(texts)
    correct_vecs = retriever.embed(correct_defs)
    wrong_vecs = retriever.embed(wrong_defs)

    for (report_id, sentence, _, correct_id), tv, cv in zip(triples, text_vecs, correct_vecs):
        result.faithful.append(Pair(report_id, sentence, correct_id, float((tv * cv).sum())))
    for (report_id, sentence, _, _), tv, wv, (wrong_id, _) in zip(triples, text_vecs, wrong_vecs, wrong_choices):
        result.mismatched.append(Pair(report_id, sentence, wrong_id, float((tv * wv).sum())))

    return result


def _stats(pairs: list[Pair]) -> str:
    if not pairs:
        return "n=0"
    scores = sorted(p.score for p in pairs)
    n = len(scores)
    return f"n={n}  min={scores[0]:.3f}  median={scores[n // 2]:.3f}  max={scores[-1]:.3f}"


def render(result: Result) -> str:
    faithful_scores = [p.score for p in result.faithful]
    mismatched_scores = [p.score for p in result.mismatched]
    overlap = (
        bool(faithful_scores) and bool(mismatched_scores)
        and min(faithful_scores) < max(mismatched_scores)
    )
    below_threshold = sum(1 for s in faithful_scores if s < ADVISORY_THRESHOLD)

    lines = [
        "# Semantic grounding results",
        "",
        f"Model `{MODEL_NAME}`. Measures whether embedding similarity between a generated "
        "sentence and its cited KB definition can distinguish a faithful paraphrase from a "
        "sentence paired with a plausible but wrong definition.",
        "Regenerate with `python -m eval.semantic_grounding_eval`. Requires Ollama running.",
        "",
        "| Set | Stats |",
        "| --- | --- |",
        f"| Faithful (sentence vs. its own cited definition) | {_stats(result.faithful)} |",
        f"| Mismatched (sentence vs. a random other definition) | {_stats(result.mismatched)} |",
        "",
        (
            f"**The ranges overlap: worst faithful score ({min(faithful_scores):.3f}) is "
            f"below the worst mismatched score ({max(mismatched_scores):.3f}).** No fixed "
            "threshold separates the two sets without both false positives (rejecting real "
            "paraphrases) and false negatives (missing wrong-definition sentences)."
            if overlap else
            "The ranges did not overlap in this run — re-check before relying on this."
        ),
        "",
        f"At the advisory threshold used in code ({ADVISORY_THRESHOLD}), "
        f"{below_threshold}/{len(faithful_scores)} faithful sentences would be flagged "
        "(false positives if this were an enforced cutoff instead of an advisory one).",
        "",
        "## Lowest-scoring faithful pairs (would be flagged first if enforced)",
        "",
        "| Report | Score | KB id | Sentence |",
        "| --- | ---: | --- | --- |",
        *[
            f"| {p.report_id} | {p.score:.3f} | {p.kb_id} | {p.sentence} |"
            for p in sorted(result.faithful, key=lambda p: p.score)[:8]
        ],
        "",
        "## Highest-scoring mismatched pairs (would slip past an enforced cutoff)",
        "",
        "| Report | Score | Wrong KB id | Sentence |",
        "| --- | ---: | --- | --- |",
        *[
            f"| {p.report_id} | {p.score:.3f} | {p.kb_id} | {p.sentence} |"
            for p in sorted(result.mismatched, key=lambda p: -p.score)[:8]
        ],
        "",
        "## Conclusion",
        "",
        "Same finding as retrieval (step 4), for the same reason: embedding similarity "
        "measures relatedness, not equivalence. This is why "
        "`app/safety/semantic.py`'s check is advisory-only — logged for a human reviewer, "
        "never used to reject an explanation on its own. The structural citation check in "
        "`app/safety/validator.py` remains the sole gate.",
        "",
    ]
    if result.errors:
        lines += ["## Ollama errors during this run", "", *[f"- {e}" for e in result.errors], ""]
    return "\n".join(lines)


def main() -> None:
    report = render(evaluate())
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
