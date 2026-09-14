"""Output — text-to-speech.

Uses gTTS (Google Translate's TTS endpoint) — a free, no-API-key hosted service, unlike
every other model in this project. Measured: well under a second per call, since it's a
network request, not local inference — no latency trade-off to make here the way there
was for generation and translation.

Supports the same languages as translation (English, Hindi) for the same reason: the
audio is synthesized from `final_text`, which is only ever produced in a supported
language. `final_text` already carries the safety disclaimer (appended in
`app/safety/validator.py` before translation ever sees it), so the disclaimer is spoken
too, not just the plain-language explanation — nothing extra needed here to guarantee that.

Speech is optional and best-effort: if the request is genuinely for an unsupported
language, or the network call fails, `synthesize_speech` raises `SpeechUnavailableError`.
`app/pipeline.py` catches it and returns the rest of the response with no audio rather
than failing the whole request over an add-on feature — the same "don't let an optional
piece take down the primary one" pattern as provenance logging (app/safety/provenance.py).
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

IMPLEMENTATION = "real"

# Matches app/i18n/translator.py's SUPPORTED_LANGUAGES: audio is synthesized from
# final_text, which is only ever produced in a language translation actually supports.
SUPPORTED_LANGUAGES = frozenset({"en", "hi"})


class SpeechUnavailableError(Exception):
    """Audio could not be produced — unsupported language, or the network call failed."""


def synthesize_speech(text: str, language: str) -> bytes:
    if language not in SUPPORTED_LANGUAGES:
        raise SpeechUnavailableError(
            f"Speech is not available in '{language}'. "
            f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
        )
    if not text.strip():
        raise SpeechUnavailableError("Nothing to synthesize: the explanation was empty.")

    from gtts import gTTS  # heavy-ish import (requests-based); only paid when speech is built

    try:
        buffer = io.BytesIO()
        gTTS(text=text, lang=language).write_to_fp(buffer)
    except Exception as exc:  # gTTS raises a mix of its own errors and requests errors
        raise SpeechUnavailableError(f"gTTS request failed: {exc}") from exc

    audio = buffer.getvalue()
    if not audio:
        raise SpeechUnavailableError("gTTS returned no audio data.")
    return audio
