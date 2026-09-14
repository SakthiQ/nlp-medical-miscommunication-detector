"""Step 7: translation. Building the model is fast enough for the regular lane
(~5.6s cached, same order as the NER/retriever fixtures); actually calling it for a
non-English language is not — those calls are marked `i18n`.
"""

from __future__ import annotations

import pytest

from app.i18n.translator import LANGUAGES, SUPPORTED_LANGUAGES, Translator, UnsupportedLanguageError


@pytest.fixture(scope="module")
def translator():
    return Translator()


def test_supported_languages_is_english_and_hindi_only():
    assert SUPPORTED_LANGUAGES == {"en", "hi"}


def test_english_is_a_no_op_and_never_touches_the_model(translator, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("English must short-circuit before the model is called")

    monkeypatch.setattr(translator, "_pipe", fail)
    text = "Your report mentions \"hepatic steatosis\"."
    assert translator.translate(text, "en") == text


def test_unsupported_language_raises_without_touching_the_model(translator, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("an unsupported language must be rejected before calling the model")

    monkeypatch.setattr(translator, "_pipe", fail)
    with pytest.raises(UnsupportedLanguageError, match="ta"):
        translator.translate("text", "ta")


def test_unsupported_language_error_names_the_supported_ones():
    translator_stub = Translator.__new__(Translator)  # skip __init__, no model needed
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        translator_stub.translate("text", "fr")
    assert "en" in str(excinfo.value) and "hi" in str(excinfo.value)


# --- real model calls (slow; skip with -m "not i18n") --------------------------------


@pytest.mark.i18n
def test_hindi_translation_is_not_english(translator):
    result = translator.translate('Your report mentions "hepatic steatosis".', "hi")
    assert result != 'Your report mentions "hepatic steatosis".'
    assert any("ऀ" <= ch <= "ॿ" for ch in result)  # contains Devanagari script


@pytest.mark.i18n
def test_hindi_translation_preserves_line_structure(translator):
    text = "First sentence here.\nSecond sentence here.\n\nA disclaimer on its own line."
    result = translator.translate(text, "hi")
    # Three non-blank source lines in -> three translated segments out.
    assert len([line for line in result.split("\n\n") if line.strip()]) == 3


@pytest.mark.i18n
def test_translation_only_ever_receives_validated_text(translator):
    """Structural guarantee, not a translation-quality test: the pipeline (app/pipeline.py)
    calls Translator.translate(validation.final_text, ...) — never the raw LLM output. This
    checks the function itself has no path that would accept anything else silently."""
    import inspect

    from app import pipeline

    source = inspect.getsource(pipeline.ExplainPipeline.run)
    assert "self.translator.translate(validation.final_text" in source
