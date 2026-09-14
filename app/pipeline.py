"""Orchestrates the slide's stages in order.

preprocess → NER → terminology → negation → retrieval → generation → validation →
translation → speech (optional)

Any stage failure raises `PipelineStageError` naming the stage, so the API returns a
clear error instead of a partial response. An explanation is replaced by the
deterministic template if the LLM is unreachable or unparseable, or if its output fails
the safety check; if even the template fails, nothing is returned. Speech is the one
exception to "fail loudly": it's an optional add-on, so a synthesis failure returns the
rest of the response with no audio rather than failing the whole request — see
app/tts/speech.py.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

logger = logging.getLogger(__name__)

from app.api.schemas import (
    CitationOut,
    CurationSuggestion,
    ExplainResponse,
    SuggestionOut,
    TermOut,
    ValidationOut,
)
from app.generation import explainer
from app.generation.explainer import generate_explanation
from app.generation.llm_client import LLMUnavailableError
from app.generation.template import template_explanation
from app.i18n import translator
from app.i18n.translator import SUPPORTED_LANGUAGES, Translator
from app.models import FlaggedTerm, RetrievedChunk
from app.ner import biobert_ner, preprocessing, terminology
from app.ner.biobert_ner import MedicalNER
from app.ner.preprocessing import NegationDetector, preprocess_report
from app.ner.terminology import DictionaryMatcher, identify_terms
from app.retrieval import retriever
from app.retrieval.knowledge_base import KnowledgeBase
from app.retrieval.retriever import Retriever
from app.safety import validator
from app.safety.provenance import AUDIT_LOG_PATH, record_provenance
from app.safety.semantic import semantic_advisory_flags
from app.safety.validator import validate_explanation
from app.tts import speech
from app.tts.speech import SpeechUnavailableError, synthesize_speech


class PipelineStageError(Exception):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message


class FeatureUnavailableError(PipelineStageError):
    """The request asked for something this build cannot do yet."""


class ExplainPipeline:
    def __init__(
        self,
        kb: KnowledgeBase,
        ner: MedicalNER | None = None,
        retriever: Retriever | None = None,
        translator_: Translator | None = None,
    ) -> None:
        # Everything heavy is built once here, at startup, never per request.
        self.kb = kb
        self.matcher = DictionaryMatcher(kb)
        self.ner = ner if ner is not None else MedicalNER()
        self.negation = NegationDetector()
        self.retriever = retriever if retriever is not None else Retriever(kb)
        self.translator = translator_ if translator_ is not None else Translator()

    @staticmethod
    def implementations() -> dict[str, str]:
        """Which stages are real, partial, or placeholders — reported on every response."""
        return {
            "preprocessing": preprocessing.IMPLEMENTATION,
            "ner": biobert_ner.IMPLEMENTATION,
            "terminology": terminology.IMPLEMENTATION,
            "negation": preprocessing.NEGATION_IMPLEMENTATION,
            "retrieval": retriever.IMPLEMENTATION,
            "generation": explainer.IMPLEMENTATION,
            "validation": validator.IMPLEMENTATION,
            "translation": translator.IMPLEMENTATION,
            "speech": speech.IMPLEMENTATION,
        }

    def run(
        self,
        report_text: str,
        target_language: str = "en",
        include_audio: bool = False,
    ) -> ExplainResponse:
        if target_language not in SUPPORTED_LANGUAGES:
            raise FeatureUnavailableError(
                "translation",
                f"Translation to '{target_language}' is not available yet. "
                f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}.",
            )
        timings: dict[str, float] = {}

        with _stage("preprocessing", timings):
            sentences = preprocess_report(report_text)

        with _stage("ner", timings):
            ner_entities = self.ner.extract(report_text, sentences)

        with _stage("terminology", timings):
            terms = identify_terms(self.matcher.match(sentences), ner_entities, self.kb)

        with _stage("negation", timings):
            terms = self.negation.annotate(sentences, terms)

        with _stage("retrieval", timings):
            # Explanation evidence: only each term's own exact KB link.
            retrieved: dict[str, list[RetrievedChunk]] = {}
            for term in terms:
                if term.kb_id and term.kb_id not in retrieved:
                    retrieved[term.kb_id] = self.retriever.for_term(term)
            # Curation queue: nearest entries for unvetted terms, never shown to patients.
            gap_texts = list(dict.fromkeys(t.text for t in terms if t.status == "no_vetted_definition"))
            curation = [
                CurationSuggestion(
                    text=text,
                    suggestions=[SuggestionOut(kb_id=c.kb_id, term=c.term, score=c.score) for c in chunks],
                )
                for text, chunks in zip(gap_texts, self.retriever.suggest(gap_texts))
            ]

        with _stage("generation", timings):
            # The LLM being unreachable, slow, or unparseable is an availability problem,
            # not a safety one: it falls back to the template immediately rather than
            # surfacing as a 500, the same way an unsafe LLM *output* does below.
            generation_error: str | None = None
            try:
                explanation = generate_explanation(report_text, terms, retrieved)
            except LLMUnavailableError as exc:
                generation_error = str(exc)
                explanation = template_explanation(terms, retrieved)

        with _stage("validation", timings):
            validation = validate_explanation(explanation, retrieved)
            fallback_used = generation_error is not None
            if generation_error:
                validation.flags = [f"generation_unavailable: {generation_error}", *validation.flags]
                if not validation.passed:
                    raise PipelineStageError(
                        "validation",
                        "The fallback template explanation failed safety checks, so nothing was returned.",
                    )
            elif not validation.passed and explanation.generator != "template":
                fallback = validate_explanation(template_explanation(terms, retrieved), retrieved)
                if not fallback.passed:
                    raise PipelineStageError(
                        "validation",
                        "The fallback explanation failed safety checks, so nothing was returned.",
                    )
                # Keep the rejected explanation's flags so the failure stays auditable.
                fallback.flags = validation.flags
                validation = fallback
                fallback_used = True
            elif not validation.passed:
                raise PipelineStageError(
                    "validation", "The explanation failed safety checks, so nothing was returned."
                )

            # Advisory only — logged for a human reviewer, never changes `passed` or
            # triggers a fallback. Checked against the LLM's own candidate sentences
            # (`explanation`, not `validation`'s possibly-substituted ones), so the audit
            # trail shows how the model's answer read even when it was rejected outright
            # for a structural or keyword reason above. See app/safety/semantic.py.
            if explanation.generator.startswith("llm:"):
                validation.flags = [
                    *validation.flags,
                    *semantic_advisory_flags(explanation, retrieved, self.retriever.embed),
                ]

        with _stage("translation", timings):
            # English (the common case) short-circuits inside Translator.translate before
            # touching the model, so this costs nothing when no translation is requested.
            # Any other supported language is real, and measured slow: ~15-20s per
            # sentence on this CPU (eval/translation_eval.py) — see app/i18n/translator.py.
            final_text = self.translator.translate(validation.final_text, target_language)

        with _stage("speech", timings):
            # Optional and best-effort: synthesizes from the final, already-translated
            # text (so the spoken disclaimer matches the written one), but a failure
            # here — unsupported language, or the network call to gTTS failing — must
            # never fail the whole request. See app/tts/speech.py.
            audio_base64: str | None = None
            if include_audio:
                try:
                    audio_base64 = base64.b64encode(synthesize_speech(final_text, target_language)).decode()
                except SpeechUnavailableError as exc:
                    logger.warning("Speech synthesis skipped: %s", exc)

        # Best-effort and off by default (see app/safety/provenance.py). Not timed as a
        # pipeline stage: it must never affect what the patient waits for or sees.
        provenance_id = record_provenance(
            original_text=report_text,
            terms=terms,
            retrieved=retrieved,
            explanation=explanation,
            validation=validation,
            fallback_used=fallback_used,
            generation_error=generation_error,
            timings_ms=timings,
            path=AUDIT_LOG_PATH,
        )

        return ExplainResponse(
            original_text=report_text,
            target_language=target_language,
            terms=[self._term_out(term, retrieved) for term in terms],
            explanation=final_text,
            citations=[
                CitationOut(
                    kb_id=chunks[0].kb_id,
                    term=chunks[0].term,
                    source_citation=chunks[0].source_citation,
                    source_url=chunks[0].source_url,
                )
                for chunks in retrieved.values()
                if chunks
            ],
            curation_suggestions=curation,
            validation=ValidationOut(
                passed=validation.passed and not fallback_used,
                flags=validation.flags,
                fallback_used=fallback_used,
            ),
            audio_base64=audio_base64,
            pipeline=self.implementations(),
            timings_ms=timings,
            provenance_id=provenance_id,
        )

    @staticmethod
    def _term_out(term: FlaggedTerm, retrieved: dict[str, list[RetrievedChunk]]) -> TermOut:
        chunks = retrieved.get(term.kb_id, []) if term.kb_id else []
        return TermOut(
            text=term.text,
            label=term.label,
            start=term.start,
            end=term.end,
            status=term.status,
            kb_id=term.kb_id,
            definition=chunks[0].plain_definition if chunks else None,
            source_citation=chunks[0].source_citation if chunks else None,
            negated=term.negated,
        )


@contextmanager
def _stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError(name, f"{type(exc).__name__}: {exc}") from exc
    finally:
        timings[name] = round((perf_counter() - started) * 1000, 2)
