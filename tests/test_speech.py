"""Step 7 (finished): text-to-speech. gTTS is a hosted call (needs internet, no API key)
and measured sub-second, so unlike the LLM/NLLB tests these run in the regular fast lane.
"""

from __future__ import annotations

import pytest

from app.tts.speech import SUPPORTED_LANGUAGES, SpeechUnavailableError, synthesize_speech


def test_supported_languages_matches_translation():
    from app.i18n.translator import SUPPORTED_LANGUAGES as TRANSLATION_LANGUAGES

    assert SUPPORTED_LANGUAGES == TRANSLATION_LANGUAGES


def test_english_synthesis_returns_real_audio():
    audio = synthesize_speech("Your report mentions hepatic steatosis.", "en")
    assert len(audio) > 1000
    assert audio[:3] == b"ID3" or (audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0)  # ID3 tag or MPEG frame sync


def test_unsupported_language_raises_without_a_network_call(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("gTTS must not be called for an unsupported language")

    monkeypatch.setattr("gtts.gTTS", fail)
    with pytest.raises(SpeechUnavailableError, match="fr"):
        synthesize_speech("text", "fr")


def test_blank_text_raises_without_a_network_call(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("gTTS must not be called with nothing to say")

    monkeypatch.setattr("gtts.gTTS", fail)
    with pytest.raises(SpeechUnavailableError, match="empty"):
        synthesize_speech("   ", "en")


def test_network_failure_becomes_speech_unavailable_error(monkeypatch):
    def broken(*args, **kwargs):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr("gtts.gTTS", broken)
    with pytest.raises(SpeechUnavailableError, match="network unreachable"):
        synthesize_speech("Your report mentions hepatic steatosis.", "en")


def test_disclaimer_text_is_included_in_what_gets_synthesized():
    """Structural check, not an audio-content check: the pipeline (app/pipeline.py)
    synthesizes speech from `final_text`, which already carries the disclaimer appended
    in app/safety/validator.py — nothing in speech.py needs to add it separately."""
    import inspect

    from app import pipeline

    source = inspect.getsource(pipeline.ExplainPipeline.run)
    assert "synthesize_speech(final_text, target_language)" in source
