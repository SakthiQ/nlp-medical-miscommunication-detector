"""Sarvam AI client: translation (mayura:v1) and text-to-speech (bulbul:v3).

Optional. Enabled only when `SARVAM_API_KEY` is set; `app/i18n/translator.py` and
`app/tts/speech.py` fall back to NLLB and gTTS when it is absent or a call fails.
Unlike NLLB on CPU (5-15 s per sentence) these are hosted calls, so the text leaves
this machine: that is the trade-off, and why it is opt-in.

Endpoints and fields: https://docs.sarvam.ai/api-reference-docs/text/translate-text.md
and https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert
"""

from __future__ import annotations

import base64
import os

import requests

BASE_URL = os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai")
TRANSLATE_MODEL = os.environ.get("SARVAM_TRANSLATE_MODEL", "mayura:v1")
TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.environ.get("SARVAM_TTS_SPEAKER", "shubh")
TIMEOUT_S = 30

# app language code -> Sarvam BCP-47 code
LANGUAGE_CODES: dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ml": "ml-IN",
}

TRANSLATE_MAX_CHARS = 1000  # mayura:v1 limit
TTS_MAX_CHARS = 2500  # bulbul:v3 limit


class SarvamError(Exception):
    """A Sarvam call failed or Sarvam is not configured."""


def is_configured() -> bool:
    return bool(os.environ.get("SARVAM_API_KEY"))


def _post(path: str, payload: dict) -> dict:
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise SarvamError("SARVAM_API_KEY is not set.")
    try:
        response = requests.post(
            f"{BASE_URL}{path}",
            json=payload,
            headers={"api-subscription-key": key},
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SarvamError(f"Sarvam request to {path} failed: {exc}") from exc


def translate(text: str, target_lang: str) -> str:
    """Translate one English line to `target_lang` (an app language code)."""
    if len(text) > TRANSLATE_MAX_CHARS:
        raise SarvamError(f"Line is {len(text)} chars; Sarvam translate allows {TRANSLATE_MAX_CHARS}.")
    body = _post(
        "/translate",
        {
            "input": text,
            "source_language_code": LANGUAGE_CODES["en"],
            "target_language_code": LANGUAGE_CODES[target_lang],
            "model": TRANSLATE_MODEL,
            "mode": "formal",
        },
    )
    translated = body.get("translated_text")
    if not translated:
        raise SarvamError("Sarvam translate returned no text.")
    return translated


def synthesize(text: str, language: str) -> bytes:
    """Speak `text` in `language` (an app language code); returns MP3 bytes."""
    if len(text) > TTS_MAX_CHARS:
        raise SarvamError(f"Text is {len(text)} chars; Sarvam TTS allows {TTS_MAX_CHARS}.")
    body = _post(
        "/text-to-speech",
        {
            "text": text,
            "language_code": LANGUAGE_CODES[language],
            "model": TTS_MODEL,
            "speaker": TTS_SPEAKER,
            "output_audio_codec": "mp3",
        },
    )
    audios = body.get("audios") or []
    if not audios:
        raise SarvamError("Sarvam TTS returned no audio.")
    try:
        return base64.b64decode(audios[0])
    except ValueError as exc:
        raise SarvamError(f"Sarvam TTS returned invalid audio: {exc}") from exc
