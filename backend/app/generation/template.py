"""Deterministic template explanation.

Uses KB definitions word for word, so it cannot hallucinate. It serves as the
placeholder generator in step 0 and remains the safety fallback once the LLM
arrives: if an LLM explanation fails validation, this is what the patient sees.

Negated terms ("No hydronephrosis") are worded as not found, never as present.
"""

from __future__ import annotations

from app.models import Explanation, ExplanationSentence, FlaggedTerm, RetrievedChunk
from app.retrieval.knowledge_base import normalize

NO_TERMS_NOTICE = "We did not find any medical terms in this report that need explaining."

_GAP_ADVICE = (
    "We do not have a checked plain-language explanation for this term, "
    "so please ask your doctor about it."
)


def template_explanation(
    terms: list[FlaggedTerm],
    retrieved: dict[str, list[RetrievedChunk]],
) -> Explanation:
    sentences: list[ExplanationSentence] = []
    seen: set[tuple[str, bool]] = set()

    for term in terms:
        if term.status == "explained" and term.kb_id:
            key = (term.kb_id, term.negated)
            chunks = retrieved.get(term.kb_id, [])
            if key in seen or not chunks:
                continue
            seen.add(key)
            best = chunks[0]
            if term.negated:
                text = (
                    f'Your report says "{term.text}" was not found. '
                    f"For reference, it means: {best.plain_definition}"
                )
            else:
                text = f'Your report mentions "{term.text}". What it means: {best.plain_definition}'
            sentences.append(ExplanationSentence(text=text, kb_ids=[best.kb_id], kind="definition"))

        elif term.status == "no_vetted_definition":
            key = (normalize(term.text), term.negated)
            if key in seen:
                continue
            seen.add(key)
            lead = (
                f'Your report says "{term.text}" was not found.'
                if term.negated
                else f'Your report mentions "{term.text}".'
            )
            sentences.append(ExplanationSentence(text=f"{lead} {_GAP_ADVICE}", kb_ids=[], kind="gap"))

    if not sentences:
        sentences.append(ExplanationSentence(text=NO_TERMS_NOTICE, kb_ids=[], kind="notice"))

    return Explanation(sentences=sentences, generator="template")
