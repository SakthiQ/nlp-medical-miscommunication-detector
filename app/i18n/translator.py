"""Output — translation.

Model: `facebook/nllb-200-distilled-600M` via HuggingFace transformers, not IndicTrans2.
IndicTrans2 needs its own custom tokenizer library (IndicTransToolkit) with a separate
preprocessing pipeline; NLLB covers the same target languages through the standard
transformers `pipeline("translation", ...)` API, which is what the rest of this project
already uses everywhere else, at the cost of being a general-purpose model rather than
one tuned specifically for Indian languages.

Only one language is enabled: Hindi. Not a placeholder gap — a measured one. On this
CPU-only machine, a single forward translation of a three-sentence explanation took
40-50 seconds (`eval/translation_eval.py`), on top of whatever the generation stage
already cost. That number is the same for every language NLLB supports, so "add the
other three languages" is not more engineering work, it is roughly 4x more waiting per
non-English request — a product decision, not a code change, and one this build does not
make silently: see README section 7 (i18n) and the roadmap notes.

Translation only ever receives the already-validated final text, never raw LLM output —
translating cannot be a way to bypass the safety checks in app/safety/validator.py.

Back-translation is NOT run on every request: round-tripping through the same model roughly
doubles an already-severe latency for a signal the patient never sees. It exists as an
offline check in `eval/translation_eval.py` instead, with its own documented caveat: a
stable round trip shows the model did not throw the content away, not that the forward
translation was correct. (Measured: "hepatic steatosis" translated into Telugu and
Malayalam round-trips back as "hepatic stenosis" — a different condition — in both, even
though the forward translation itself, read as a script, is a reasonable transliteration
of the clinical term. The round trip is not the ground truth.)
"""

from __future__ import annotations

import os

IMPLEMENTATION = "real"

MODEL_NAME = os.environ.get("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")

# language code (as used by the API / demo page) -> (display name, NLLB FLORES-200 code)
LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("English", "eng_Latn"),
    "hi": ("Hindi", "hin_Deva"),
    # Tamil (tam_Taml), Telugu (tel_Telu), Malayalam (mal_Mlym) are supported by the same
    # model and code path — enabling them is a one-line addition to this dict — but are
    # deliberately not turned on: see the module docstring for the measured latency cost.
}

SUPPORTED_LANGUAGES = frozenset(LANGUAGES)

MAX_LENGTH = 800


class UnsupportedLanguageError(ValueError):
    pass


class Translator:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # Heavy imports; only paid when translation is built. Local cache first, same
        # lesson as app/ner/biobert_ner.py and app/retrieval/retriever.py: loading
        # "online" re-checks Hugging Face for newer files on every start even when the
        # model is already downloaded, which cost seconds there — only go online here if
        # the model genuinely isn't cached yet.
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        self.model_name = model_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        except OSError:  # not cached yet: download once
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=-1)

    def translate(self, text: str, target_lang: str) -> str:
        if target_lang not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Translation to '{target_lang}' is not available yet. "
                f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
            )
        if target_lang == "en":
            return text  # no-op, and skips paying the model for the common case

        _, code = LANGUAGES[target_lang]
        # Translate line by line, not the whole block as one string: NLLB is trained on
        # sentence/short-passage pairs, and `final_text` is one sentence per line plus a
        # blank-line-separated disclaimer. Sending that as a single string with embedded
        # newlines risks the model running lines together; translating each separately
        # is both what was measured (eval/translation_eval.py) and safer for structure.
        lines = [line for line in text.splitlines() if line.strip()]
        translated = [
            self._pipe(line, src_lang="eng_Latn", tgt_lang=code, max_length=MAX_LENGTH)[0][
                "translation_text"
            ]
            for line in lines
        ]
        return "\n\n".join(translated)
