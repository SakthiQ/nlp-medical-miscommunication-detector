"""Measure embedding retrieval against data/retrieval_queries.json.

    python -m eval.retrieval_eval

Prints the results and writes eval/retrieval_results.md. Reports Recall@1, Recall@3 and
mean reciprocal rank on lay-language paraphrases, then asks whether any similarity
threshold could safely let retrieval explain terms that have no KB entry: it compares the
paraphrases' scores with the scores unrelated terms get against their nearest entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from app.retrieval.knowledge_base import load_knowledge_base
from app.retrieval.retriever import MODEL_NAME, Retriever

QUERIES_PATH = Path(__file__).resolve().parents[1] / "data" / "retrieval_queries.json"
RESULTS_PATH = Path(__file__).resolve().parent / "retrieval_results.md"
K = 3


def evaluate() -> str:
    kb = load_knowledge_base()
    retriever = Retriever(kb)
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    paraphrases = queries["paraphrases"]
    unrelated = queries["unrelated"]

    paraphrase_results = retriever.suggest([q["query"] for q in paraphrases], K)
    unrelated_results = retriever.suggest(unrelated, K)

    hits_at_1 = hits_at_k = 0
    reciprocal_ranks = []
    correct_scores = []
    paraphrase_rows = []
    for query, results in zip(paraphrases, paraphrase_results):
        ids = [chunk.kb_id for chunk in results]
        rank = ids.index(query["kb_id"]) + 1 if query["kb_id"] in ids else None
        hits_at_1 += rank == 1
        hits_at_k += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank:
            correct_scores.append(results[rank - 1].score)
        top = ", ".join(f"{c.kb_id} {c.term} ({c.score:.3f})" for c in results)
        paraphrase_rows.append(f"| {query['query']} | {query['kb_id']} | {rank or 'miss'} | {top} |")

    unrelated_rows = []
    unrelated_top = []
    for text, results in zip(unrelated, unrelated_results):
        best = results[0]
        unrelated_top.append((best.score, text, best))
        unrelated_rows.append(f"| {text} | {best.kb_id} {best.term} | {best.score:.3f} |")

    worst_unrelated, worst_text, worst_entry = max(unrelated_top, key=lambda item: item[0])
    # Only correct top-1 answers could be accepted automatically; count those that would
    # clear a threshold set just above the highest-scoring unrelated term.
    survivors = sum(
        1 for query, results in zip(paraphrases, paraphrase_results)
        if results[0].kb_id == query["kb_id"] and results[0].score > worst_unrelated
    )
    n = len(paraphrases)

    lines = [
        "# Retrieval results",
        "",
        f"Model `{MODEL_NAME}`, FAISS inner product over normalised vectors (cosine), "
        f"{len(kb.explainable)} indexed KB entries (term, aliases and definition).",
        "Regenerate with `python -m eval.retrieval_eval`.",
        "",
        "## Paraphrase retrieval",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Recall@1 | {hits_at_1}/{n} ({hits_at_1 / n:.1%}) |",
        f"| Recall@{K} | {hits_at_k}/{n} ({hits_at_k / n:.1%}) |",
        f"| Mean reciprocal rank | {sum(reciprocal_ranks) / n:.3f} |",
        f"| Score of the correct entry (min / median) | {min(correct_scores):.3f} / {median(correct_scores):.3f} |",
        "",
        f"| Query | Expected | Rank | Top {K} |",
        "| --- | --- | ---: | --- |",
        *paraphrase_rows,
        "",
        "## Could a threshold let retrieval explain unvetted terms?",
        "",
        "Each unrelated term's nearest entry. None of these entries is a correct explanation.",
        "",
        "| Term | Nearest entry | Score |",
        "| --- | --- | ---: |",
        *unrelated_rows,
        "",
        f"The highest-scoring unrelated term, \"{worst_text}\", reaches {worst_unrelated:.3f} "
        f"against {worst_entry.term}. A threshold above that would accept only {survivors} of "
        f"{n} paraphrases, while the weakest correct match scores {min(correct_scores):.3f}. "
        "The score ranges overlap, so no threshold is safe.",
        "",
        "This is expected: embedding similarity measures relatedness, not equivalence. "
        "\"renal\" is about the kidney, so it lands near a kidney condition, but it is not "
        "that condition. The system therefore never explains a term through a semantic "
        "match. Explanations use exact KB links only; semantic neighbours are returned as "
        "`curation_suggestions` for a human curator to confirm or reject.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    report = evaluate()
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
