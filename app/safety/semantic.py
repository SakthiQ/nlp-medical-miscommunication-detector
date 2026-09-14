"""Semantic grounding check — informational, not enforced.

Measured (eval/semantic_grounding_eval.py, 43 sentences from the real LLM over the
evaluation corpus):

- A sentence's similarity to its own cited definition ranges 0.265-0.970 (median 0.719).
- A sentence paired with a wrong-but-plausible definition ranges -0.000-0.495.

These ranges overlap: the worst faithful sentence (0.265) scores lower than the worst
mismatched one (0.495). No fixed threshold can reliably separate a genuine paraphrase
from a plausible-sounding wrong one — the same conclusion as retrieval step 4's "no
threshold is safe" finding, for the same underlying reason: embedding similarity measures
relatedness, not equivalence.

Given that, this check is advisory only. It never changes `ValidationResult.passed` and
never triggers the template fallback by itself — the structural check in
`app/safety/validator.py` (does the sentence cite a KB id that was actually retrieved?) is
what gates the response, and stays the sole gate, per the original design brief's own
recommendation that structured citation-checking is "far stronger than post-hoc
similarity." This check adds a low-confidence signal to the audit log for a human
reviewer: a threshold set just below the measured faithful floor, so it flags only the
more obviously adrift sentences, while accepting — openly, not silently — that it will
miss some drifted sentences and occasionally flag a genuinely fine one.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.models import Explanation, RetrievedChunk

# Just below the lowest score any known-faithful sentence reached in measurement (0.265).
ADVISORY_THRESHOLD = 0.25

EmbedFn = Callable[[list[str]], "np.ndarray"]


def semantic_advisory_flags(
    explanation: Explanation,
    retrieved: dict[str, list[RetrievedChunk]],
    embed: EmbedFn,
) -> list[str]:
    """"semantic_advisory: ..." flags for definition sentences that read as adrift from
    their cited definition. Never call this to decide whether an explanation is safe —
    see the module docstring.
    """
    candidates = [
        (index, sentence.text, sentence.kb_ids[0])
        for index, sentence in enumerate(explanation.sentences)
        if sentence.kind == "definition" and sentence.kb_ids and retrieved.get(sentence.kb_ids[0])
    ]
    if not candidates:
        return []

    text_vectors = embed([text for _, text, _ in candidates])
    definitions = [retrieved[kb_id][0].plain_definition for _, _, kb_id in candidates]
    definition_vectors = embed(definitions)
    # Both sides are pre-normalised (Retriever.embed), so the dot product is cosine similarity.
    similarities = np.sum(np.asarray(text_vectors) * np.asarray(definition_vectors), axis=1)

    return [
        f"semantic_advisory: sentence {index} scored {score:.2f} against {kb_id} "
        "(informational only, not enforced — see app/safety/semantic.py)"
        for (index, _, kb_id), score in zip(candidates, similarities)
        if score < ADVISORY_THRESHOLD
    ]
