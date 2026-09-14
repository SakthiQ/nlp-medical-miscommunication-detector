"""Slide stage 6 — safety & validation.

PARTIAL (step 0): checks that every definition sentence cites a retrieved KB entry,
blocks diagnostic and prescriptive phrasing, and appends the disclaimer. Step 6 adds
semantic grounding checks and provenance logging.
"""

from __future__ import annotations

import re

from app.models import Explanation, RetrievedChunk, ValidationResult

IMPLEMENTATION = "partial"

DISCLAIMER = (
    "This explains findings already in your report. "
    "Please discuss any questions with your doctor."
)

FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "diagnosis": re.compile(r"\byou (have|are suffering from|are diagnosed)\b|\bdiagnos(is|ed|e)\b", re.I),
    "advice": re.compile(r"\byou (should|must|need to|ought to)\b|\b(i|we) recommend\b", re.I),
    "treatment": re.compile(r"\b(start|stop|begin) taking\b|\bprescri(be|bed|ption)\b", re.I),
    # A dose such as "500 mg" — but not a lab unit such as "142 mg/dL".
    "dosage": re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|units?)\b(?!\s*/)", re.I),
    "prognosis": re.compile(r"\b(will|won't|is going to) (get worse|improve|recover|go away)\b", re.I),
}


def validate_explanation(
    explanation: Explanation,
    retrieved: dict[str, list[RetrievedChunk]],
) -> ValidationResult:
    flags: list[str] = []
    retrieved_ids = {chunk.kb_id for chunks in retrieved.values() for chunk in chunks}

    for index, sentence in enumerate(explanation.sentences):
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(sentence.text):
                flags.append(f"forbidden_{name}: sentence {index}")

        if sentence.kind == "definition":
            if not sentence.kb_ids:
                flags.append(f"ungrounded: sentence {index} cites no source")
            unknown = [kb_id for kb_id in sentence.kb_ids if kb_id not in retrieved_ids]
            if unknown:
                flags.append(f"ungrounded: sentence {index} cites unretrieved {unknown}")
        elif sentence.kb_ids:
            flags.append(f"unexpected_citation: {sentence.kind} sentence {index} cites {sentence.kb_ids}")

    return ValidationResult(
        passed=not flags,
        flags=flags,
        final_text=f"{explanation.plain_text}\n\n{DISCLAIMER}",
    )
