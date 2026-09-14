"""End-to-end tests for POST /explain."""

import base64

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Explanation, ExplanationSentence
from app.safety.validator import DISCLAIMER

DECK_EXAMPLE = "Mild hepatic steatosis with elevated ALT."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.llm
def test_deck_example_end_to_end(client):
    response = client.post("/explain", json={"report_text": DECK_EXAMPLE})
    assert response.status_code == 200
    body = response.json()

    assert body["original_text"] == DECK_EXAMPLE
    assert [(t["text"], t["kb_id"]) for t in body["terms"]] == [
        ("hepatic steatosis", "kb-001"),
        ("ALT", "kb-002"),
    ]
    assert all(t["status"] == "explained" and t["definition"] for t in body["terms"])
    assert "liver" in body["explanation"]
    assert body["explanation"].endswith(DISCLAIMER)
    assert {c["kb_id"] for c in body["citations"]} == {"kb-001", "kb-002"}

    # Not asserting a single fixed validation outcome: the real LLM sometimes writes
    # "you have [finding]" phrasing that trips forbidden_diagnosis on its own (measured
    # ~3/17 on the eval corpus — see eval/generation_results.md and README section 7).
    # Both outcomes are correct system behavior: either the LLM's own phrasing passed
    # cleanly, or the safety net caught it and fell back. What must always hold is that
    # a caught problem is reflected honestly in the response, not hidden.
    if body["validation"]["fallback_used"]:
        assert body["validation"]["passed"] is False
        assert body["validation"]["flags"]  # the reason is recorded, not silently dropped
    else:
        assert body["validation"] == {"passed": True, "flags": [], "fallback_used": False}
    assert body["audio_base64"] is None


@pytest.mark.llm  # terms here are all explained and un-negated, so this calls the real LLM
def test_term_offsets_point_at_the_original_text(client):
    text = "Lipid profile shows LDL 168 mg/dL. HbA1c 7.8%. Serum creatinine 1.1 mg/dL."
    body = client.post("/explain", json={"report_text": text}).json()
    assert body["terms"], "expected terms to be found"
    for term in body["terms"]:
        assert text[term["start"]:term["end"]] == term["text"]


@pytest.mark.llm
def test_response_reports_which_stages_are_placeholders(client):
    body = client.post("/explain", json={"report_text": DECK_EXAMPLE}).json()
    assert body["pipeline"]["preprocessing"] == "real"
    assert body["pipeline"]["ner"] == "real"
    assert body["pipeline"]["negation"] == "real"
    assert body["pipeline"]["retrieval"] == "real"
    assert body["pipeline"]["generation"] == "real"
    assert body["pipeline"]["terminology"] == "partial"
    assert set(body["timings_ms"]) == {
        "preprocessing", "ner", "terminology", "negation", "retrieval",
        "generation", "validation", "translation",
    }


@pytest.mark.llm
def test_unvetted_term_found_by_ner_is_reported_honestly(client):
    # Corpus report ev-014. The NER model needs the full report as context: it misses
    # "femoral neck" when the first sentence is sent on its own.
    text = "Bone density scan shows osteopenia at the femoral neck. T-score -1.8. No osteoporosis."
    body = client.post("/explain", json={"report_text": text}).json()

    terms = {t["text"]: t for t in body["terms"]}
    assert terms["osteopenia"]["status"] == "explained"
    assert terms["femoral neck"]["status"] == "no_vetted_definition"
    assert terms["osteoporosis"]["negated"] is True
    assert 'Your report mentions "femoral neck". We do not have a checked' in body["explanation"]
    assert body["validation"]["passed"] is True


@pytest.mark.llm
def test_negated_finding_is_never_explained_as_present(client):
    text = "ECG shows normal sinus rhythm at 72 bpm. No evidence of acute ischemia."
    body = client.post("/explain", json={"report_text": text}).json()

    negated = {t["text"]: t["negated"] for t in body["terms"]}
    assert negated["normal sinus rhythm"] is False
    assert negated["ischemia"] is True
    assert 'Your report says "ischemia" was not found.' in body["explanation"]
    assert 'mentions "ischemia"' not in body["explanation"]
    assert body["validation"]["passed"] is True


def test_common_terms_are_not_flagged(client):
    text = "Patient reports fever, headache, and nausea for three days. Blood pressure was normal."
    body = client.post("/explain", json={"report_text": text}).json()
    assert body["terms"] == []
    assert "did not find any medical terms" in body["explanation"]
    assert body["explanation"].endswith(DISCLAIMER)


def test_unsupported_language_is_a_clear_error(client):
    response = client.post("/explain", json={"report_text": DECK_EXAMPLE, "target_language": "ta"})
    assert response.status_code == 501
    assert response.json()["detail"]["stage"] == "translation"


@pytest.mark.i18n  # real NLLB translation call
def test_explain_in_hindi_end_to_end(client):
    text = "No evidence of acute ischemia."  # negated -> template wording, no LLM call needed
    response = client.post("/explain", json={"report_text": text, "target_language": "hi"})
    assert response.status_code == 200
    body = response.json()

    assert body["target_language"] == "hi"
    # original_text stays in English — only the explanation is translated.
    assert body["original_text"] == text
    assert any("ऀ" <= ch <= "ॿ" for ch in body["explanation"])  # Devanagari present
    # Safety validation happens on the English explanation, before translation touches
    # it — translating cannot be a way around the checks in app/safety/validator.py.
    assert body["validation"]["passed"] is True


def test_audio_is_synthesized_when_requested(client):
    # Negated finding -> template wording, no LLM call needed, keeps this test fast.
    text = "No evidence of acute ischemia."
    response = client.post("/explain", json={"report_text": text, "include_audio": True})
    assert response.status_code == 200
    body = response.json()

    assert body["audio_base64"]
    audio = base64.b64decode(body["audio_base64"])
    assert len(audio) > 1000
    assert audio[:3] == b"ID3" or (audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0)  # ID3 tag or MPEG frame sync
    # A speech failure must never affect the explanation itself.
    assert body["validation"]["passed"] is True


def test_no_audio_by_default(client):
    # Negated finding -> template wording, no LLM call needed, keeps this test fast.
    body = client.post("/explain", json={"report_text": "No evidence of acute ischemia."}).json()
    assert body["audio_base64"] is None


def test_speech_failure_does_not_fail_the_request(client, monkeypatch):
    def broken_synthesize(*args, **kwargs):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr("app.pipeline.synthesize_speech", broken_synthesize)
    text = "No evidence of acute ischemia."
    response = client.post("/explain", json={"report_text": text, "include_audio": True})
    # A bug in the *call site* (not caught as SpeechUnavailableError) still shouldn't
    # reach here in practice — synthesize_speech only raises SpeechUnavailableError — but
    # this confirms a genuinely broken audio path fails loudly rather than silently, per
    # the project's general fail-loudly stance, while a *recognised* unavailability
    # (see the next test) degrades gracefully instead.
    assert response.status_code == 500
    assert response.json()["detail"]["stage"] == "speech"


def test_recognised_speech_unavailability_degrades_gracefully(client, monkeypatch):
    from app.tts.speech import SpeechUnavailableError

    def unavailable(*args, **kwargs):
        raise SpeechUnavailableError("gTTS request failed: network unreachable")

    monkeypatch.setattr("app.pipeline.synthesize_speech", unavailable)
    text = "No evidence of acute ischemia."
    response = client.post("/explain", json={"report_text": text, "include_audio": True})
    assert response.status_code == 200
    body = response.json()
    assert body["audio_base64"] is None
    assert body["validation"]["passed"] is True


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_blank_report_is_rejected(client, text):
    assert client.post("/explain", json={"report_text": text}).status_code == 422


@pytest.mark.llm
def test_unvetted_term_gets_curation_suggestions_but_no_explanation(client):
    # Corpus report ev-007. "renal" is nearest to hydronephrosis by embedding similarity,
    # but related is not equivalent: it must stay an "ask your doctor" term.
    text = (
        "Non-obstructing 4 mm calculus in the right renal pelvis. No hydronephrosis. "
        "Cholelithiasis noted incidentally."
    )
    body = client.post("/explain", json={"report_text": text}).json()

    status = {t["text"]: t["status"] for t in body["terms"]}
    assert status["renal"] == "no_vetted_definition"
    assert 'Your report mentions "renal". We do not have a checked' in body["explanation"]

    suggestions = {s["text"]: s["suggestions"] for s in body["curation_suggestions"]}
    assert set(suggestions) == {t for t, s in status.items() if s == "no_vetted_definition"}
    assert len(suggestions["renal"]) == 3
    assert all(0 < s["score"] < 1 for s in suggestions["renal"])


def test_stage_failure_names_the_stage(client, monkeypatch):
    def broken_retrieve(*args, **kwargs):
        raise RuntimeError("index not loaded")

    monkeypatch.setattr(client.app.state.pipeline.retriever, "for_term", broken_retrieve)
    response = client.post("/explain", json={"report_text": DECK_EXAMPLE})
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["stage"] == "retrieval"
    assert "index not loaded" in detail["message"]


def test_unsafe_generation_falls_back_to_template(client, monkeypatch):
    def unsafe_llm(original_text, terms, retrieved):
        return Explanation(
            sentences=[
                ExplanationSentence(
                    text="You have fatty liver. You should start metformin 500 mg daily.",
                    kb_ids=["kb-001"],
                    kind="definition",
                )
            ],
            generator="llm-test",
        )

    monkeypatch.setattr("app.pipeline.generate_explanation", unsafe_llm)
    body = client.post("/explain", json={"report_text": DECK_EXAMPLE}).json()

    assert body["validation"]["fallback_used"] is True
    assert body["validation"]["passed"] is False
    flags = " ".join(body["validation"]["flags"])
    for expected in ("forbidden_diagnosis", "forbidden_advice", "forbidden_dosage"):
        assert expected in flags
    assert "metformin" not in body["explanation"]
    assert "Extra fat has built up inside the liver" in body["explanation"]
