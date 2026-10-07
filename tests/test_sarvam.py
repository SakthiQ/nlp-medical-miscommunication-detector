"""Sarvam AI translation and speech, with the HTTP layer mocked: no key or network needed."""

from __future__ import annotations

import base64

import pytest

from app import sarvam
from app.i18n.translator import Translator
from app.tts.speech import SpeechUnavailableError, synthesize_speech


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def sarvam_key(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")


def test_not_configured_without_a_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    assert not sarvam.is_configured()
    with pytest.raises(sarvam.SarvamError, match="SARVAM_API_KEY"):
        sarvam.translate("hello", "hi")


def test_translate_sends_documented_fields(sarvam_key, monkeypatch):
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return FakeResponse({"translated_text": "नमस्ते"})

    monkeypatch.setattr(sarvam.requests, "post", fake_post)
    assert sarvam.translate("hello", "hi") == "नमस्ते"
    assert seen["url"].endswith("/translate")
    assert seen["headers"] == {"api-subscription-key": "test-key"}
    assert seen["json"]["source_language_code"] == "en-IN"
    assert seen["json"]["target_language_code"] == "hi-IN"


def test_synthesize_decodes_base64_audio(sarvam_key, monkeypatch):
    audio = b"ID3fake-mp3"
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, json=json)
        return FakeResponse({"audios": [base64.b64encode(audio).decode()]})

    monkeypatch.setattr(sarvam.requests, "post", fake_post)
    assert sarvam.synthesize("hello", "ta") == audio
    assert seen["url"].endswith("/text-to-speech")
    assert seen["json"]["language_code"] == "ta-IN"
    assert seen["json"]["output_audio_codec"] == "mp3"


def test_http_failure_becomes_sarvam_error(sarvam_key, monkeypatch):
    def boom(*args, **kwargs):
        raise sarvam.requests.ConnectionError("down")

    monkeypatch.setattr(sarvam.requests, "post", boom)
    with pytest.raises(sarvam.SarvamError, match="down"):
        sarvam.translate("hello", "te")


def test_translator_uses_sarvam_and_skips_the_local_model(sarvam_key, monkeypatch):
    monkeypatch.setattr(sarvam, "translate", lambda text, lang: f"[{lang}] {text}")
    translator = Translator()
    assert translator._pipe is None  # NLLB not loaded
    assert translator.translate("One.\nTwo.", "ml") == "[ml] One.\n\n[ml] Two."


def test_translator_falls_back_to_nllb_when_sarvam_fails(sarvam_key, monkeypatch):
    def fail(text, lang):
        raise sarvam.SarvamError("down")

    monkeypatch.setattr(sarvam, "translate", fail)
    translator = Translator()
    monkeypatch.setattr(translator, "_load_local_model", lambda: setattr(
        translator, "_pipe", lambda line, **kw: [{"translation_text": f"nllb:{line}"}]))
    assert translator.translate("One.", "hi") == "nllb:One."


def test_speech_uses_sarvam_then_falls_back_to_gtts(sarvam_key, monkeypatch):
    monkeypatch.setattr(sarvam, "synthesize", lambda text, lang: b"sarvam-audio")
    assert synthesize_speech("hello", "hi") == b"sarvam-audio"

    def fail(text, lang):
        raise sarvam.SarvamError("down")

    monkeypatch.setattr(sarvam, "synthesize", fail)

    class FakeGTTS:
        def __init__(self, text, lang):
            pass

        def write_to_fp(self, fp):
            fp.write(b"gtts-audio")

    monkeypatch.setattr("gtts.gTTS", FakeGTTS)
    assert synthesize_speech("hello", "hi") == b"gtts-audio"


def test_speech_still_rejects_unsupported_language_with_sarvam_on(sarvam_key):
    with pytest.raises(SpeechUnavailableError, match="fr"):
        synthesize_speech("hello", "fr")
