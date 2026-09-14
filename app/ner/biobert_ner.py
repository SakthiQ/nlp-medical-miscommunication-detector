"""Slide stage 2 — medical NER.

Model: `d4data/biomedical-ner-all`, a DistilBERT token classifier trained on clinical
case reports (MACCROBAT). The slide calls this stage "BioBERT"; this is the checkpoint
the project brief names, and it is DistilBERT-based rather than BioBERT itself.

Its job here is coverage detection. The dictionary matcher already finds every KB term
exactly, so NER hits that overlap a dictionary match are discarded downstream, and the
rest surface as "no vetted definition" terms instead of being silently skipped.

Two findings shape how the model is called:
- It runs on the whole report, not sentence by sentence. It was trained on full case
  reports and loses terms without surrounding context ("femoral neck" and "antral" are
  found in the full report but missed in their sentence alone). Reports longer than the
  model's 512-token window are covered with overlapping windows (`stride`).
- Aggregation uses the "first" strategy, which labels whole words. The default "simple"
  strategy fragments medical words into subword pieces ("he" + "patic steatosis") and
  truncates others ("hydronephrosis" becomes "hydro").
"""

from __future__ import annotations

import bisect
import os
from pathlib import Path

from app.models import Entity, Sentence
from app.retrieval.knowledge_base import normalize

IMPLEMENTATION = "real"

MODEL_NAME = os.environ.get("NER_MODEL", "d4data/biomedical-ner-all")

# Below this, hits are mostly noise ("right" 0.58, "obstructing" 0.66, "Liver" 0.48).
MIN_CONFIDENCE = 0.80

# Single characters are fragments, e.g. "T" from "T-score".
MIN_LENGTH = 2

# Token overlap between windows when a report exceeds the model's input length.
WINDOW_STRIDE = 128

# Model labels kept, mapped to ours. Everything else (Severity, Lab_value, Distance,
# Detailed_description, ...) describes a finding rather than naming a medical term.
LABEL_MAP = {
    "Disease_disorder": "DISEASE",
    "Sign_symptom": "SYMPTOM",
    "Diagnostic_procedure": "TEST",
    "Biological_structure": "ANATOMY",
    "Medication": "DRUG",
}

COMMON_WORDS_PATH = Path(__file__).resolve().parents[2] / "data" / "common_words.txt"


def load_common_words(path: Path = COMMON_WORDS_PATH) -> frozenset[str]:
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            words.add(normalize(line))
    return frozenset(words)


class MedicalNER:
    def __init__(self, model_name: str = MODEL_NAME, min_confidence: float = MIN_CONFIDENCE) -> None:
        from transformers import pipeline  # heavy import; only paid when NER is built

        self.model_name = model_name
        self.min_confidence = min_confidence
        self.common_words = load_common_words()
        tokenizer, model = _load_local_first(model_name)
        self._pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            stride=WINDOW_STRIDE,
            device=-1,
        )

    def extract(self, text: str, sentences: list[Sentence]) -> list[Entity]:
        """Find medical terms in the full report, assigned to sentences by offset."""
        if not sentences or not text.strip():
            return []

        sentence_starts = [sentence.start for sentence in sentences]
        entities: list[Entity] = []
        for raw in self._pipe(text):
            label = LABEL_MAP.get(raw["entity_group"])
            score = float(raw["score"])
            if label is None or score < self.min_confidence:
                continue

            start, end = _trim(text, int(raw["start"]), int(raw["end"]))
            if end - start < MIN_LENGTH:
                continue
            surface = text[start:end]
            if normalize(surface) in self.common_words:
                continue

            sentence = sentences[max(bisect.bisect_right(sentence_starts, start) - 1, 0)]
            if not (sentence.start <= start and end <= sentence.end):
                continue  # spans a sentence boundary: not a single term

            entities.append(
                Entity(
                    text=surface,
                    label=label,
                    start=start,
                    end=end,
                    sentence_index=sentence.index,
                    confidence=round(score, 4),
                    source="ner",
                )
            )
        return entities


def _load_local_first(model_name: str):
    """Load from the local cache, going online only if the model isn't cached yet.

    Online loading asks Hugging Face for newer files on every start, even when the model
    is cached, which measured about 1 s here. Local-first also means a server can't
    silently pick up a changed model.
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    try:
        return (
            AutoTokenizer.from_pretrained(model_name, local_files_only=True),
            AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True),
        )
    except OSError:  # not cached yet: download once
        return (
            AutoTokenizer.from_pretrained(model_name),
            AutoModelForTokenClassification.from_pretrained(model_name),
        )


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Drop surrounding punctuation and whitespace the aggregation sometimes includes."""
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    return start, end
