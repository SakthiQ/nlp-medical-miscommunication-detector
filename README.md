# Medical Miscommunication Detector

An NLP system that turns hard-to-read medical reports into plain-language explanations
for patients, without ever diagnosing, prescribing, or making things up.

> **Input:** `"Mild hepatic steatosis with elevated ALT."`
>
> **Output:** *Your report mentions "hepatic steatosis". What it means: Extra fat has built
> up inside the liver…* plus a definition of ALT, the source for each definition, and a
> disclaimer to discuss questions with a doctor.

---

## 1. The problem and the scope

Patients often receive reports full of terms like *cholelithiasis*, *eGFR* or
*atelectasis* with no one around to explain them. Asking a general chatbot is risky: it
can invent facts, sound like a diagnosis, or misread "No evidence of ischemia" as
"you have ischemia".

**What this system does:** explains the terms and findings already documented in the
report, using only definitions from a curated, cited knowledge base.

**What it deliberately does not do:**
- diagnose, or say what a finding means *for this patient*
- recommend treatment, medication, or doses
- predict outcomes
- replace a clinician

This scope is a feature. The honest ceiling is a vocabulary and readability layer: a
patient learns what "steatosis" and "ALT" mean, but not *why* they appear together,
because explaining that would be clinical interpretation.

---

## 2. Architecture

```
                        ┌───────────────────────── FastAPI  POST /explain ─────────────────────────┐
 Medical report ──────► │                                                                          │
                        │  1. Preprocessing ── sentence split (pysbd) with exact character offsets │
                        │          │                                                               │
                        │  2. Medical NER ──── transformer model over the whole report             │
                        │          │                                                               │
                        │  3. Terminology ──── dictionary match (KB) + NER hits, merged            │
                        │          │            • in KB      → "explained"                         │
                        │          │            • not in KB  → "no vetted definition"              │
                        │          │            • common word (fever) → not flagged                │
                        │  3b. Negation ────── NegEx: "No hydronephrosis" → negated                │
                        │          │                                                               │
                        │  4. Retrieval ────── fetch KB definitions (FAISS planned)                │
                        │          │                                                               │
                        │  5. Generation ───── LLM writes explanation from retrieved text only     │
                        │          │            (Llama 3 via Ollama; template for gaps/negation,   │
                        │          │             and as fallback if the LLM is unavailable/unsafe) │
                        │  6. Safety ───────── grounding check + blocked phrases + disclaimer      │
                        │          │            fails → deterministic template fallback            │
                        │  7. Outputs ──────── translation (Hindi) + text-to-speech (gTTS)          │
                        └──────────┼───────────────────────────────────────────────────────────────┘
                                   ▼
            JSON: highlighted terms (with offsets), explanation, citations, validation flags
```

Three routes a term can take:
- **Main route:** the term is in the knowledge base → its definition is retrieved → the
  explanation is written from it → the safety check verifies it.
- **Coverage gap:** NER finds a medical term the KB doesn't cover → the patient is told
  honestly that there is no checked explanation and to ask their doctor. Nothing is guessed.
- **Safety net:** if a generated explanation fails validation, the KB definition is shown
  word for word instead. If even that fails, nothing is returned.

### Stage-by-stage

| # | Stage | Module | Technology | Status |
|---|---|---|---|---|
| 1 | Preprocessing | `app/ner/preprocessing.py` | pysbd + clinical-abbreviation fix | Built |
| 2 | Medical NER | `app/ner/biobert_ner.py` | `d4data/biomedical-ner-all` (HuggingFace) | Built |
| 3 | Terminology identification | `app/ner/terminology.py` | Regex dictionary matcher over KB aliases | Built (partial) |
| 3b | Negation detection | `app/ner/preprocessing.py` | negspacy (NegEx, clinical termset) | Built |
| 4 | Retrieval (RAG) | `app/retrieval/retriever.py` | sentence-transformers embeddings + FAISS | Built |
| 5 | Explanation generation | `app/generation/explainer.py` | Llama 3 via Ollama for explained/un-negated terms; deterministic template for negated findings, coverage gaps, and as the fallback | Built |
| 6 | Safety & validation | `app/safety/validator.py` | Grounding + regex filters + fallback | Partial |
| 7 | Translation / speech | `app/i18n/`, `app/tts/` | NLLB (Hindi only — measured cost, see section 8); gTTS for speech (English + Hindi) | Built |
| – | Orchestration | `app/pipeline.py`, `app/api/` | FastAPI, Pydantic | Built |

Every API response includes a `pipeline` field listing which stages are real, partial,
or placeholders, so a demo can never overstate what is built.

---

## 3. User walkthrough and experience

This section follows one patient from receiving a report to talking with their doctor.
Each step pairs **what the patient experiences** with **what the system does
technically** to make that happen.

**Today vs planned:** the backend and the patient-facing page below — including
translation and the audio player in Step 7 — are all built and running. The page is
served at `/` (also reachable via FastAPI's interactive API page at `/docs`). See
section 8 for how it's built and one real bug found while testing it.

### 3.1 The patient and the situation

A 52-year-old patient gets an abdominal CT and bone-density report through a hospital
portal. The next appointment is three weeks away. The report reads:

> Non-obstructing 4 mm calculus in the right renal pelvis. No hydronephrosis.
> Cholelithiasis noted incidentally. Bone density scan shows osteopenia at the femoral neck.

They don't know what most of these words mean, whether "No hydronephrosis" is good or bad,
or which parts to worry about. What they need is to **understand what the report says**,
not a second opinion on it.

### 3.2 Step by step

#### Step 1 — Open the page and paste the report

**Patient experience:** a single text box, a language dropdown (English for now), and an
"Explain my report" button. A short line under the box says the report is not saved.
Nothing to sign up for.

**Technical:**
- The page sends `POST /explain` with `{ "report_text": "...", "target_language": "en" }`.
- Pydantic validates the request before any processing: blank or whitespace-only text is
  rejected with HTTP 422 and a clear message; text over 20,000 characters is rejected.
- Nothing is written to disk or a database **by default**. An operator can opt in to
  audit logging (`AUDIT_LOG_PATH`, roadmap step 6) for clinician review, which then
  records the report text and every validation outcome to a local file — see section 7.

#### Step 2 — Submit and wait

**Patient experience:** most of the time the result appears almost instantly. When the
report has a finding the LLM needs to phrase, it takes several seconds — the page will
need a progress message for that case (not yet built).

**Technical:**
- All heavy components (the NER model, the negation model, the retriever, the knowledge
  base and its alias index) were loaded once when the server started, so the request
  pays none of that cost.
- `app/pipeline.py` runs the stages in order and times each one. Two real, current
  measurements, because the two paths are very different:

  | Stage | This report (LLM needed) | "No evidence of ischemia. Fever, headache." (no LLM call) |
  |---|---:|---:|
  | Preprocessing | 6.96 ms | 1.49 ms |
  | NER | 38.49 ms | 25.53 ms |
  | Terminology | 0.08 ms | 0.17 ms |
  | Negation | 0.76 ms | 0.46 ms |
  | Retrieval | 0.03 ms | 0.06 ms |
  | Generation | **8,062 ms** | 0.06 ms |
  | Validation | 0.05 ms | 0.08 ms |
  | **Full round trip** | **about 8.1 s** | **about 28 ms** |

  The second report needs no LLM call at all: *ischemia* is negated and *fever*/*headache*
  are common terms, so every sentence comes from the deterministic template. Generation is
  the entire cost difference — every other stage is within a few milliseconds either way.
- These timings are returned in `timings_ms`, so latency can be monitored per stage, and a
  frontend can tell from the request pattern (or a `generating` flag, not yet added)
  whether to show a spinner.

#### Step 3 — See the report with medical terms highlighted

**Patient experience:** their original report, unchanged, with medical terms highlighted
in three visibly different styles:
- **Explained terms** (solid highlight): the system has a checked definition.
- **Not-found findings** (highlight with a "not found" badge): the report says this is
  absent, e.g. *No hydronephrosis*.
- **Unchecked terms** (dashed outline): recognised as medical, but no checked definition
  exists, so the patient is pointed to their doctor.

Everyday words like "fever" or "liver" are left alone, so the page isn't a wall of
highlights.

**Technical:** this is the core NLP pipeline. The terms returned for this report
(live run, from the build before NER switched to whole-report input, see note below):

| Text | Offsets | Label | Status | Negated | Found by |
|---|---|---|---|---|---|
| calculus | 21–29 | SYMPTOM | no vetted definition | no | NER |
| renal | 43–48 | ANATOMY | no vetted definition | no | NER |
| hydronephrosis | | DISEASE | explained (kb-041) | **yes** | dictionary |
| Cholelithiasis | | DISEASE | explained (kb-007) | no | dictionary |
| incidentally | | DESCRIPTOR | explained (kb-045) | no | dictionary |
| Bone density scan | | TEST | no vetted definition | no | NER |
| osteopenia | | DISEASE | explained (kb-025) | no | dictionary |

How each column is produced:
1. **Sentence splitting** (`preprocessing.py`): pysbd splits the text into 4 sentences
   without breaking on "4 mm" or decimals. Each sentence records its exact start and end
   character in the original text.
2. **Dictionary matching** (`terminology.py`): one compiled regex built from every KB alias,
   longest first, case-insensitive, with word boundaries. It finds *hydronephrosis*,
   *Cholelithiasis*, *incidentally* and *osteopenia* and links each to its KB id.
3. **NER** (`biobert_ner.py`): the transformer model reads the whole report and proposes
   *calculus*, *renal* and *Bone density scan*. Hits below 0.80 confidence ("right",
   "obstructing"), descriptive labels ("4 mm" as a distance) and everyday words are dropped.
4. **Merging** (`identify_terms`): dictionary matches win any overlap. NER hits that
   survive and aren't in the KB become `no_vetted_definition`.
5. **Negation** (`NegationDetector`): NegEx examines each term within its own sentence and
   marks *hydronephrosis* as negated because of the preceding "No".
6. **Offsets:** every stage keeps absolute character positions, so the page can highlight
   `report_text[start:end]` exactly, with no re-searching or guessing.

> **Honest note:** "femoral neck" is a real gap term here but was missed in this run. The
> NER model is context-sensitive and at the time was reading one sentence at a time. It
> now reads the whole report, and in the evaluation it finds "femoral neck" in the
> equivalent corpus report (ev-014).

#### Step 4 — Tap a term to see what it means

**Patient experience:** tapping *Cholelithiasis* opens a small card: *One or more hard,
stone-like lumps that have formed in the gallbladder.* Underneath, the source it came from.
Tapping *renal* shows: *We don't have a checked explanation for this term. Please ask your
doctor about it.*

**Technical:**
- Each term in the response carries `definition`, `source_citation` and `kb_id`, filled by
  the retrieval stage from the knowledge base. The card needs no second request.
- Definitions are written at a 6th–8th grade reading level and describe what a term
  *means*, never what it means *for this patient*.
- For gap terms, `definition` is `null` by design: the system never fills it with a
  nearest-looking entry.

#### Step 5 — Read the plain-language explanation

**Patient experience:** below the report, a short explanation:

> A hard, stone-like lump was found in the gallbladder.
>
> The gallbladder stone was an unexpected discovery.
>
> The bone density at the femoral neck was lower than normal.
>
> Your report mentions "calculus". We do not have a checked plain-language explanation for
> this term, so please ask your doctor about it.
>
> Your report mentions "renal". We do not have a checked plain-language explanation for this
> term, so please ask your doctor about it.
>
> Your report says "hydronephrosis" was not found. For reference, it means: Urine has backed
> up and made part of the kidney swell.
>
> Your report mentions "Bone density scan". We do not have a checked plain-language
> explanation for this term, so please ask your doctor about it.
>
> This explains findings already in your report. Please discuss any questions with your doctor.

This is the actual output of a live run, from Llama 3 via Ollama, and it passed the
safety validator unchanged.

**Technical:**
- Two generators cooperate. Llama 3 (`generation/explainer.py`) writes the sentences for
  terms that are `explained` and not negated — here, *Cholelithiasis* and *osteopenia*
  (it phrased both in its own words rather than repeating the term). Everything else —
  the negated *hydronephrosis*, and the three coverage gaps — keeps the deterministic
  template's exact wording (`generation/template.py`), unchanged from before step 5.
- This split exists because it was tested and found necessary, not assumed: asked to
  handle a negated finding itself, the model explained the underlying condition *before*
  saying it was absent, and invented a sentence `kind` outside the schema. The fixed
  wording for negation and gaps was already correct and tested, so the model is kept out
  of that decision entirely.
- Every sentence is an `ExplanationSentence` with a `kind` (`definition`, `gap` or
  `notice`) and the `kb_ids` it draws from — model-authored sentences included, since the
  model is prompted to return that tagging itself. That is what the safety layer checks.
- **Known ordering limitation:** Llama 3's sentences are placed first, then the template's,
  rather than interleaved in report order. In this example the *Bone density scan* gap
  ends up after *hydronephrosis* instead of next to *osteopenia*. Noted here rather than
  hidden; fixing it is a small ordering pass, not a design change.
- If the model is unreachable, times out, or returns text that doesn't parse into this
  schema, generation falls back to the template for every term, not just the ones that
  failed — see Step 6.

#### Step 6 — Decide whether to trust it

**Patient experience:** the page gives visible reasons to trust it, and to know its limits:
a list of sources, the "not found" wording on absent findings, honest "ask your doctor"
lines instead of guesses, and the closing disclaimer.

**Technical:** before anything is shown, `safety/validator.py` checks the explanation:
- every definition sentence must cite a KB id that was actually retrieved
- gap and notice sentences must cite nothing (they may not make medical claims)
- no diagnosis ("you have"), advice ("you should"), treatment ("start taking"),
  dosage ("500 mg", while allowing lab units like "142 mg/dL"), or prognosis wording
- the disclaimer is always appended

If a check fails, or the LLM could not be reached or parsed at all, the explanation is
replaced by the fully deterministic template and `validation.fallback_used` is set to
`true`, with a flag recording why (`generation_unavailable: ...` or the safety flags the
LLM output tripped) kept for auditing. If the template also fails, the API returns an
error and the patient sees nothing unsafe.

#### Step 7 — Switch language or listen

**Patient experience:** choose Hindi from the dropdown — the label warns up front that
it's slower — and get the explanation translated, with a progress message that adjusts
once the wait passes 30 seconds. Tick "Also read this aloud" for an audio player with
the explanation, disclaimer included, read in whichever language was selected.

**Technical:**
- Translation (NLLB, not IndicTrans2 — see section 8) runs only on the already-validated
  `final_text`, never on raw LLM output, so translation can't be a way to bypass the
  safety checks.
- Only Hindi is enabled, and it is genuinely slow: 5.6-14.7 seconds measured per sentence
  on this CPU. Tamil, Telugu, and Malayalam are measured in `eval/translation_eval.py`
  but not turned on, because that cost is the same for every language the model supports
  — see section 8 for the real numbers and why back-translation checking runs offline
  rather than on every request.
- Choosing an unsupported language (anything but English or Hindi) returns HTTP 501
  naming the `translation` stage, and the page shows "This language isn't available yet"
  instead of failing silently.
- Speech (gTTS — a free hosted service, no API key, measured under a second per call —
  see section 8) synthesizes from the same already-translated, already-validated
  `final_text`, so the disclaimer is spoken, not just written. It is genuinely optional:
  if the network call fails, the rest of the response still comes back with
  `audio_base64: null` rather than failing the whole request over an add-on feature.

#### Step 8 — Go to the appointment prepared

**Patient experience:** they arrive knowing what the report says, which finding was absent,
and which three terms (calculus, renal, bone density scan) to ask about. The gap terms
naturally form their list of questions.

**Technical:** gap terms are returned as structured data (`status:
"no_vetted_definition"`), so a future "questions for your doctor" list can be generated
directly from them. The same gap list is the curation queue for growing the knowledge base.

### 3.3 Edge cases and what the patient sees

| Situation | What the patient sees | How it's handled technically |
|---|---|---|
| Report has only everyday terms ("fever, headache") | "We did not find any medical terms in this report that need explaining." + disclaimer | KB entries marked `is_common` are matched but never flagged; the template emits a `notice` sentence |
| Blank submission | "Please paste your report" | HTTP 422 from request validation |
| Finding is absent ("No evidence of ischemia") | "Your report says 'ischemia' was not found." | NegEx negation within the term's sentence; template wording rule |
| "Rule out cholelithiasis" | Explained normally, **not** as "not found" | "Rule out" cues removed from NegEx because they signal uncertainty, not absence |
| Term the system has never seen | "No checked explanation — ask your doctor" | NER hit with no KB match → `no_vetted_definition` |
| Abbreviation inside another word ("mIU/L") | Not highlighted as "MI" (heart attack) | Word-boundary regex; covered by tests |
| Very long report (over 512 model tokens) | Terms highlighted all the way to the end | NER runs in overlapping windows (`stride=128`) and maps hits back to sentences |
| Generated text contains "you should start metformin" | Only the safe template explanation | Validator flags advice/dosage; fallback replaces it; flags kept |
| A stage crashes (e.g. model not loaded) | "Something went wrong while explaining your report" | HTTP 500 naming the failed stage; never a half-finished answer |
| Unsupported language requested (anything but English/Hindi) | "This language isn't available yet" | HTTP 501 naming the `translation` stage |
| Audio requested but the network call to gTTS fails | Explanation shows normally, no audio player | `audio_base64: null` — the rest of the response still returns; see app/tts/speech.py |

### 3.4 UX principles and how the code enforces them

| Principle | Enforced by |
|---|---|
| **Never alarm or falsely reassure.** Absent findings are clearly absent; uncertain ones aren't called absent. | Negation stage, "rule out" handling, fixed template wording |
| **Honesty over completeness.** Better "ask your doctor" than a confident guess. | `no_vetted_definition` status; `definition: null`; no nearest-match retrieval |
| **Plain language.** 6th–8th grade reading level. | Curated KB definitions; LLM prompt constraint (planned); readability scoring (planned) |
| **Show, don't replace, the report.** The patient's own text stays visible and unchanged. | Absolute character offsets on every term |
| **Every claim is traceable.** | `kb_id` per sentence, `citations` list, grounding validation |
| **Don't bury the patient in highlights.** | `is_common` flag, everyday-words stoplist, NER confidence threshold |
| **Always point back to the doctor.** | Mandatory disclaimer, appended after validation |
| **Fast when it can be; honest when it can't.** | Models loaded at startup; ~28 ms when no LLM call is needed, ~8 s when one is — no spinner yet for the slow path (see Step 2) |
| **Failures are clear, never silent or partial.** | Stage-named 500/501 errors |
| **Private by default.** | No storage of report text; local LLM planned to avoid sending reports to third parties |

---

## 4. Key design decisions (and why)

These are the choices that shaped the system.

1. **The knowledge base is the core of the system, not an add-on.** Model quality isn't
   what limits safety here; knowledge-base coverage is. A term with no curated definition
   can only be explained by guessing, so the KB and the evaluation set were built together,
   first.

2. **"Not in the knowledge base" is a real output, not a failure.** The system says
   "we don't have a checked explanation, ask your doctor" instead of retrieving the
   nearest-looking entry. Coverage becomes a metric you report instead of a hole you hide.

3. **The dictionary matcher is the precision path; NER is for finding gaps.** Finding
   known KB terms is exact string matching (100% precision on our corpus). The NER model's
   real value is spotting medical terms the KB *doesn't* cover. Where the two overlap, the
   dictionary wins.

3b. **Semantic retrieval explains nothing to a patient — it only suggests to a curator.**
   Embedding search finds the right KB entry for a lay paraphrase well (Recall@3 100% on
   12 test queries), but similarity measures *relatedness*, not equivalence. "renal"
   scores 0.634 against hydronephrosis, higher than the weakest genuine paraphrase match
   (0.412), so no similarity threshold can safely separate a real match from a related-but-
   wrong one. Explanations therefore use only a term's own exact KB link; embedding search
   returns nearest entries as `curation_suggestions`, for a human to confirm or reject —
   never shown to the patient.

4. **No stopword removal, and negation is mandatory.** Removing words like "no" and "not"
   would turn "No evidence of ischemia" into a claim that ischemia is present, which is the
   worst mistake a tool like this can make.

5. **"Rule out X" is treated as uncertain, not negated.** Standard NegEx counts it as a
   negation, but it means the doctor is *checking for* X. Telling a patient "your report
   says X was not found" would be false reassurance. "X was ruled out" is still negated.

6. **Explanations are grounded by structure, not just similarity.** Each generated
   sentence must name the KB entry it came from, and the validator checks those ids against
   what was actually retrieved. That catches a fluent, mostly-correct sentence with one
   invented claim, which similarity checks miss.

7. **Reject and fall back; never rewrite.** If output contains diagnosis or dosage
   language, it is replaced by the deterministic template, not patched with regex (a patched
   sentence can still imply the diagnosis). If the fallback also fails, the API returns an
   error rather than anything unsafe.

8. **A commonly-understood flag instead of a separate difficulty classifier.** Every KB
   term was already judged worth explaining by a human; an `is_common` flag (fever,
   headache) replaces a frequency/syllable classifier that would re-derive the same call.

9. **Fail loudly, never partially.** Each stage is wrapped so any failure returns an
   error naming the stage (HTTP 500), and features not built yet return HTTP 501.

10. **Heavy models load once at startup.** Otherwise every request would pay the
    model-loading cost.

### Engineering findings along the way

- **scispaCy was dropped.** It requires spaCy below 3.8, which has no Python 3.14 build.
  pysbd provides the abbreviation-aware sentence splitting instead.
- **pysbd needed a fix** for clinical abbreviations: it split "Pt. c/o dyspnea" and
  "5 mg. daily". A piece is now re-joined when the previous one ends in a known
  abbreviation and the next starts lowercase. Offsets are computed independently because
  pysbd can mis-locate repeated sentences.
- **NER aggregation strategy matters a lot.** The default `simple` aggregation split words
  into subword fragments ("he" + "patic steatosis") and truncated others ("hydronephrosis"
  → "hydro"). The `first` strategy labels whole words and fixed this.
- **The NER model is context-sensitive.** It found "femoral neck" in a full report but
  missed it when that sentence was sent alone. NER now runs on the whole report, with
  overlapping windows for reports longer than 512 tokens, then maps hits back to sentences.
  This raised gap detection from 3/7 to 5/7.
- **NER noise is filtered** by a 0.80 confidence threshold, a label whitelist (disease,
  symptom, test, anatomy, medication), a minimum length (drops "T" from "T-score"), and a
  small everyday-words stoplist (liver, spleen, right, left).
- **The d4data model is DistilBERT-based, not BioBERT.** The architecture slide says
  "BioBERT"; this is the checkpoint the project brief names.
- **Loading models "online" is slow even when they're already downloaded.** Both
  `transformers` and `sentence-transformers` check Hugging Face for newer files by
  default, on every start. Loading from the local cache first, and only going online if
  a model isn't cached yet, cut retriever load from 6.8 s to 0.7 s and NER load from
  1.3 s to 0.2 s, and it means a running server can't silently pick up a changed model.

---

## 5. Data

| File | Contents |
|---|---|
| `data/knowledge_base.json` | 51 entries: 46 explainable terms + 5 commonly understood. Each has an id, canonical term, aliases (ALT/SGPT, anemia/anaemia), label, plain-language definition (6th–8th grade), non-interpretive context, and source citation. |
| `data/eval_corpus.json` | 18 synthetic, non-PHI report snippets across 16 report types, with 55 hand-labelled terms: 48 covered by the KB and 7 deliberate gaps (consolidation, renal, canal stenosis, Mediastinal, femoral neck, antral, Afebrile). |
| `data/common_words.txt` | Everyday words the NER model flags but patients already know. |
| `data/retrieval_queries.json` | 12 lay-language paraphrases with their correct KB entry, plus 10 unrelated medical terms, for retrieval evaluation. |

The loader enforces that aliases are unambiguous (one alias → one entry), because an
ambiguous dictionary match silently explains the wrong term. Tests cross-check the two
files: every labelled KB id exists, every covered term is matchable, and every "gap" term
is genuinely absent from the KB.

---

## 6. Results (measured)

Term detection on the 18-report corpus (`python -m eval.ner_eval`):

| View | Gold terms | Recall | Exact-span recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| NER model alone | 55 | 74.5% | 67.3% | 78.8% | 76.6% |
| Coverage-gap detection | 7 | 71.4% (5/7) | 57.1% | n/a | n/a |
| Terminology stage (dictionary + NER) | 50 | **96.0%** | 94.0% | 81.4% | **88.1%** |

Other measured results:
- Dictionary matcher: finds exactly the labelled KB terms on all 18 reports, with no
  false matches (e.g. "last" never matches AST, "mIU" never matches MI).
- Negation: marks exactly the 4 negated findings in the corpus, and nothing else.
- Retrieval (`python -m eval.retrieval_eval`): on 12 lay-language paraphrases, Recall@1
  83.3%, Recall@3 100%, mean reciprocal rank 0.917. Full results, including why no
  similarity threshold is safe, in [eval/retrieval_results.md](eval/retrieval_results.md).
- Generation (`python -m eval.generation_eval`, Llama 3 via Ollama on the 18-report
  corpus): every one of 43 generated sentences cited a retrieved KB id — no ungrounded
  claims. **14 of 17 reports' explanations passed every safety check as generated; 3
  fell back to the template**, all for the same reason: `forbidden_diagnosis` firing on
  benign "you have [finding]" phrasing (e.g. "You have extra fat built up inside your
  liver" — a safe rephrasing of the KB definition, not a diagnosis, but the regex can't
  tell the difference from "you have diabetes"). Mean readability 7.9th grade; 23 of 43
  sentences at or below the 8.5 target. Full per-sentence output in
  [eval/generation_results.md](eval/generation_results.md).
- Latency: about 15–60 ms per request when no LLM call is needed (NER is the largest
  single stage, 10–40 ms); 10–20 s when it is, dominated entirely by CPU generation.
  Around 12 s startup once models are cached locally, more on the very first run.

Precision is strict: the labels mark only the terms a patient most needs explained, so
real medical terms the annotator left unlabelled (Urinalysis, ECG) count against it.

**Known failure modes:** NER misses common abbreviations and plain clinical words (GERD,
ischemia, hypertension, benign), which the dictionary covers when they are in the KB; it
produces partial spans ("canal" for "canal stenosis"); its output varies with surrounding
context; and it never finds "consolidation" or "Afebrile".

**Honest caveats:** the corpus is small (18 reports), synthetic, and labelled by one
person. These numbers show the method works; they are not clinical validation.

---

## 7. Safety layer

Checks run on every explanation before it reaches the patient:

| Check | Catches |
|---|---|
| Grounding | A definition sentence that cites no KB entry, or cites one that wasn't retrieved |
| Unexpected citation | A "gap" or notice sentence that makes a sourced medical claim |
| Diagnosis filter | "you have", "you are diagnosed", "diagnosis" |
| Advice filter | "you should", "you must", "I/we recommend" |
| Treatment filter | "start taking", "prescribe" |
| Dosage filter | "500 mg" (but not lab units like "142 mg/dL") |
| Prognosis filter | "will improve", "will go away" |
| Disclaimer | Always appended: *This explains findings already in your report. Please discuss any questions with your doctor.* |
| Semantic advisory *(informational only — see below)* | A definition sentence that reads as adrift from its cited definition |

Tested with a deliberately unsafe output ("You have fatty liver. You should start
metformin 500 mg daily."): all three violations are flagged and the template fallback
is shown instead.

**This isn't just a synthetic test case.** Running the real LLM over the 18-report
corpus, 3 of 17 explanations tripped `forbidden_diagnosis` on their own, not from an
adversarial prompt — the model's own natural phrasing ("you have X"). The filter is
deliberately blunt: it can't distinguish a safe rephrasing of a finding from an actual
diagnosis, so it blocks both, and about 1 in 6 real explanations pay for that caution
with a fallback to the plainer template wording. That trade — a noticeably higher
fallback rate in exchange for never needing to trust the model's judgment on this — was a
concrete argument for the smarter, structure-aware check step 6 was supposed to add.

### Why step 6 didn't add a stricter semantic check

The natural next step looked like: embed each generated sentence, compare it to its
cited definition, and reject the explanation if the similarity is too low — a "does this
sentence actually match what it claims to be based on" check, stronger than "does it cite
a valid id." Measured first, built second (`eval/semantic_grounding_eval.py`, real LLM
output over the 18-report corpus):

| Set | Similarity range |
|---|---|
| Faithful (sentence vs. its own cited definition) | 0.265 - 0.970 (median 0.719) |
| Mismatched (sentence vs. a random *other* definition) | -0.000 - 0.495 |

**The ranges overlap.** The worst faithful sentence (0.265) scores lower than the worst
mismatched one (0.495) — the same shape of finding as retrieval's "no threshold is safe"
result in step 4, for the same reason: embedding similarity measures relatedness, not
equivalence. A threshold tight enough to catch the worst mismatched sentences would also
reject a meaningful share of genuinely faithful ones; a threshold loose enough to spare
faithful sentences misses the more plausible-sounding wrong ones entirely.

So `app/safety/semantic.py` implements the check, but **advisory only**: it never sets
`validation.passed` to false and never triggers the template fallback by itself. It adds
a `semantic_advisory: ...` flag to the audit trail when a sentence scores below 0.25 (just
under the measured faithful floor), for a human reviewer to glance at — not to gate the
response on. The structural citation check above remains the sole gate, exactly as the
original project brief itself recommended ("structured output... far stronger than
post-hoc similarity"). Full numbers, including the specific sentences that would have been
false positives or false negatives under an enforced cutoff, in
[eval/semantic_grounding_results.md](eval/semantic_grounding_results.md).

### Provenance / audit logging

Per the project brief: *"Log every validation event with a provenance record... for
clinician auditability."* `app/safety/provenance.py` implements this as **opt-in, off by
default** — set `AUDIT_LOG_PATH` to a file path to enable it. When enabled, every request
appends one JSON line recording the original report text, the terms found, the retrieved
KB ids, the LLM's raw output, the final text shown, every validation flag (structural,
keyword, and semantic-advisory), and per-stage timings. A logging failure is caught and
logged as a warning, never raised — an audit trail must not be able to take the product
down.

This is a real, deliberate exception to the "nothing is stored" claim made elsewhere in
this README (section 3, Step 1; section 13) — auditability and "we store nothing" are in
direct tension, and logging is opt-in specifically so that tension is a deployment choice,
not a decision baked in silently. A local JSON-lines file is a prototype's audit trail,
not a production one: no access control, no retention limit, no encryption at rest. See
section 13.

---

## 8. Translation and the demo page

### Translation

Model: `facebook/nllb-200-distilled-600M`, not IndicTrans2. IndicTrans2 needs its own
custom tokenizer library (IndicTransToolkit) with a separate preprocessing pipeline; NLLB
covers the same languages through the standard `transformers` pipeline API already used
everywhere else in this project, at the cost of being general-purpose rather than tuned
specifically for Indian languages.

**Only Hindi is enabled — a measured decision, not a placeholder gap.** Before wiring
anything in, I measured real latency on real generated sentences
(`eval/translation_eval.py`):

| Metric | Value |
|---|---:|
| Forward translation (min / median / max) | 5.6s / 8.5s / 14.7s |
| Round trip, forward + back (min / median / max) | 9.3s / 15.3s / 24.3s |

These are **per sentence** — `app/i18n/translator.py` translates `final_text` line by
line (see its docstring for why), and a real explanation is several lines, so a live
request's translation stage runs this forward-translation cost once per sentence in the
response, not once total. That cost is the same for every language NLLB supports. Adding Tamil, Telugu, and
Malayalam is not more engineering work — the model already covers them, and the code path
is a one-line addition to a dict in `app/i18n/translator.py` — it is roughly four times
more waiting per non-English request on this CPU-only machine. That is a product decision
about acceptable latency, not a code change, so this build makes it explicitly rather than
quietly shipping four slow languages and calling it done. Full numbers, and the two other
languages measured for comparison but not enabled, in
[eval/translation_results.md](eval/translation_results.md).

**A real translation-quality finding, not just a latency one.** Round-tripping "hepatic
steatosis" through Telugu or Malayalam and back came out as "hepatic **stenosis**" — a
narrowing of a vessel, a genuinely different condition from a fatty deposit — even though
the forward translation, read as a script, is a reasonable phonetic rendering of the
clinical term. This is exactly why back-translation does not run on every request: it
would cost roughly double the latency above for a QA signal that can itself be
unreliable, and it is never shown to the patient anyway. It stays an offline check in
`eval/translation_eval.py`, with that specific finding documented rather than smoothed
over — the same "advisory, not a live gate" reasoning as the semantic grounding check in
section 7.

Translation only ever receives the already-validated `final_text`, never raw LLM output —
translating cannot be a way to route around the checks in `app/safety/validator.py`.

### Speech

`gTTS` — a free, no-API-key call to Google Translate's TTS endpoint — synthesizes audio
from the same `final_text` translation already produced, so a Hindi request gets Hindi
audio, English gets English audio, and the spoken disclaimer always matches the written
one without any separate step to guarantee that.

The trade-off pattern here is the opposite of translation's, and worth naming precisely
because of that contrast: gTTS is a hosted network call, not local CPU inference, so it
measured under a second — nothing like NLLB's cost. That's why speech didn't end up on
the trim list the roadmap planned for it, even though the original plan expected it
would. The cost that *is* real here is different: an external network dependency at
request time, and (like the LLM and translation) a failure mode the code has to handle
rather than assume away. `synthesize_speech` raises `SpeechUnavailableError` for an
unsupported language or a failed network call; `app/pipeline.py` catches it and returns
the rest of the response with `audio_base64: null` rather than failing the request over
an optional add-on — the same "don't let a secondary feature take down the primary one"
pattern already used for provenance logging (see above).

### Demo page

A single self-contained page (`app/static/index.html`, served at `/`) matching the
walkthrough in section 3: paste a report, submit, see it highlighted three ways, click a
term for its definition, read the plain-language explanation, see the sources. Built and
checked against the running app — not just described. One real bug found that way and
fixed before it shipped: the first version inserted a line break after every highlighted
term so its definition could sit inline, which fractured the patient's own sentences onto
broken lines (*"No hydronephrosis[break]. Cholelithiasis[break] noted incidentally[break]."*)
— a direct violation of "show the patient's own report, unchanged." Fixed by moving the
term detail into one shared panel below the report text instead of inline, so the report
itself is never touched.

Verified with the browser, not just the API: submitting a report with "Also read this
aloud" checked returns real audio (confirmed `readyState: 4`, an 18.5-second clip for a
negated-finding explanation) that plays in an `<audio>` element built from the response's
base64 payload via a `Blob` and an object URL — freed on the next submission rather than
left to accumulate.

Not a production frontend: no build step, no framework, inline CSS and JS in one file, on
purpose — this is a demo to show the pipeline working, not a shippable patient product.
The language dropdown offers English and Hindi, with the Hindi option's label setting
latency expectations up front rather than leaving the patient guessing, the progress
message switches to a longer, translation-specific one after 30 seconds of waiting, and
an "Also read this aloud" checkbox requests audio without forcing it on every request.

---

## 9. API

`POST /explain`

```json
{ "report_text": "Mild hepatic steatosis with elevated ALT.", "target_language": "en", "include_audio": false }
```

Response (abridged, real capture — and it happens to show the fallback firing: on this
particular run the LLM wrote "you have..." phrasing, `forbidden_diagnosis` caught it, and
the response fell back to the template rather than showing anything unsafe):

```json
{
  "terms": [
    { "text": "hepatic steatosis", "start": 5, "end": 22, "status": "explained",
      "kb_id": "kb-001", "label": "DISEASE", "negated": false,
      "definition": "Extra fat has built up inside the liver. …" },
    { "text": "ALT", "start": 37, "end": 40, "status": "explained", "kb_id": "kb-002", "label": "BIOMARKER" }
  ],
  "explanation": "Your report mentions \"hepatic steatosis\". What it means: … Please discuss any questions with your doctor.",
  "citations": [ { "kb_id": "kb-001", "source_citation": "…" } ],
  "curation_suggestions": [],
  "validation": { "passed": false, "flags": ["forbidden_diagnosis: sentence 0"], "fallback_used": true },
  "pipeline": { "preprocessing": "real", "ner": "real", "retrieval": "real", "generation": "real", "translation": "real" },
  "timings_ms": { "preprocessing": 6.5, "ner": 41.3, "negation": 0.8, "generation": 7225.4 }
}
```

`curation_suggestions` is present, and non-empty, only when a term has no vetted
definition — see section 3, Step 8. `start`/`end` are character offsets into the original
report, so a frontend can highlight each term exactly. Errors: `422` blank input, `501`
unbuilt feature (e.g. Tamil, audio), `500` stage failure naming the stage.

---

## 10. Project structure

```
app/
  main.py                 FastAPI app; builds the pipeline once at startup
  pipeline.py             Runs the stages in order, times them, handles failures and fallback
  models.py               Data passed between stages (Sentence, Entity, FlaggedTerm, …)
  api/                    Route and request/response schemas
  ner/preprocessing.py    Sentence splitting + negation detection
  ner/biobert_ner.py      Transformer NER (coverage detection)
  ner/terminology.py      Dictionary matcher + merge logic
  retrieval/              Knowledge base loader, corpus loader, retriever
  generation/             Llama 3 explainer (via Ollama) + deterministic template/fallback
  safety/validator.py     Grounding and forbidden-content checks (the enforced gate)
  safety/semantic.py      Embedding-similarity advisory check (logged, never enforced)
  safety/provenance.py    Opt-in audit logging (AUDIT_LOG_PATH), off by default
  i18n/translator.py      NLLB translation (Hindi only — see section 8)
  tts/speech.py           gTTS speech synthesis (English + Hindi), best-effort
  static/index.html       The demo page served at / (section 8)
data/                     Knowledge base, evaluation corpus, common-words stoplist
eval/ner_eval.py                 Term-detection evaluation → eval/ner_results.md
eval/retrieval_eval.py           Retrieval evaluation → eval/retrieval_results.md
eval/generation_eval.py          Generation grounding/readability check → eval/generation_results.md
eval/semantic_grounding_eval.py  Semantic-check calibration → eval/semantic_grounding_results.md
eval/translation_eval.py         Translation latency/quality → eval/translation_results.md
tests/                    Unit and end-to-end tests (real-LLM tests marked `llm`,
                          real-translation tests marked `i18n`, both slow)
```

---

## 11. Roadmap

| Step | Work | Status |
|---|---|---|
| 0 | End-to-end skeleton: `/explain` through every stage | Done |
| KB | Knowledge base + evaluation corpus, cross-validated | Done |
| 1 | Sentence splitting + negation detection | Done |
| 2 | Medical NER for coverage detection, with evaluation | Done |
| 3 | Terminology: finalise merging and labels | Mostly done |
| 4 | RAG: sentence-transformer embeddings + FAISS, Recall@3 | Done |
| 5 | Llama 3 via Ollama, prompted to use only retrieved text and cite KB ids | Done |
| 6 | Safety: semantic grounding checks, provenance/audit logging | Done — semantic check is advisory by design (measured, not enforced; see section 7); audit logging is opt-in |
| 7 | Translation (one Indian language first), speech, demo web page | Done |
| 8 | Full evaluation report with readability scores and human review | NER, retrieval, generation, and translation all measured; human faithfulness review still needed |

Planning notes:
- **Llama 3 on this machine, measured:** 31.5 GB RAM, no NVIDIA GPU, Llama 3 8B (Q4_0)
  via Ollama. CPU generation runs 10-20 s for one or two findings, longer for more —
  see [eval/generation_results.md](eval/generation_results.md) for the per-report numbers
  and the README's own measured examples in section 3. The LLM sits behind one interface
  (`app/generation/llm_client.py`) so a smaller model or a hosted API can be swapped in by
  changing `LLM_MODEL`/`OLLAMA_HOST`, without touching the prompt or parsing logic.
- **Knowledge base growth:** seed from MedlinePlus Health Topics (public domain), UMLS
  synonyms and the Consumer Health Vocabulary; prioritise by term frequency; use an LLM to
  draft definitions with a human approving each; turn logged coverage gaps into the
  curation queue. Target: 300–500 reviewed terms.
- **Trim order if time runs short:** speech first, then extra languages. Keep the demo
  page, the safety layer, and the evaluation numbers. What actually happened differs from
  the plan in an informative way: speech turned out cheap enough to keep — gTTS is a
  hosted call, measured under a second, nothing like NLLB's local-CPU cost — so it shipped
  after all. The extra-language cut is the one that held, and for exactly the reason
  anticipated: that cost is real CPU latency (eval/translation_eval.py), not a guess that
  turned out to be avoidable.

---

## 12. Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The first start downloads the NER model (`d4data/biomedical-ner-all`, ~260 MB), the
embedding model (`all-MiniLM-L6-v2`, ~90 MB), and the translation model
(`nllb-200-distilled-600M`, ~2.4 GB) — all cached locally afterward. Requires Python
3.11+; developed on Python 3.14.

Generation also needs [Ollama](https://ollama.com) running locally with the model
pulled:

```bash
ollama pull llama3
```

If Ollama isn't running, `/explain` still works — it falls back to the deterministic
template automatically (`validation.fallback_used: true`, with a
`generation_unavailable` flag saying why). Speech (gTTS) needs internet access at request
time, not just at startup, since it's a hosted call — see section 8 for what happens if
that call fails.

Audit logging is off by default. To enable it (see section 7):

```bash
export AUDIT_LOG_PATH=data/audit_log.jsonl   # PowerShell: $env:AUDIT_LOG_PATH = "data/audit_log.jsonl"
```

The demo page (section 3) is served at `/` once the server is running.

```bash
curl -X POST http://127.0.0.1:8000/explain -H "Content-Type: application/json" -d "{\"report_text\": \"Mild hepatic steatosis with elevated ALT.\"}"
python -m pytest                                 # everything: real Ollama + NLLB + gTTS calls included (~4-5 min)
python -m pytest -m "not llm and not i18n"       # fast lane: no Ollama/NLLB calls, still builds those models once (~80-90s cold, faster once the OS file cache is warm)
python -m eval.ner_eval                          # NER evaluation → eval/ner_results.md
python -m eval.retrieval_eval                    # retrieval evaluation → eval/retrieval_results.md
python -m eval.generation_eval                   # generation evaluation → eval/generation_results.md
python -m eval.semantic_grounding_eval           # semantic-check calibration → eval/semantic_grounding_results.md
python -m eval.translation_eval                  # translation latency/quality → eval/translation_results.md
```

---

## 13. Known limitations

- Source citations name their source, but `source_url` is empty until each link is
  verified. Several entries cite the MedlinePlus Medical Encyclopedia, which is licensed
  from A.D.A.M. and must not be copied; those definitions need re-sourcing from
  public-domain MedlinePlus Health Topics.
- Ambiguous abbreviations (PE, MS) are not handled yet and are left out of the dictionary.
- The evaluation corpus is small, synthetic and single-annotator.
- Generated explanations have not had the human faithfulness review the project brief
  calls for (Phase 5's acceptance criteria: reading 10 explanations by hand to confirm
  every claim traces to its source). `eval/generation_eval.py` checks grounding and
  readability automatically, which is a triage pass, not a substitute for that review.
- The LLM's sentences are placed before the template's, not interleaved in report order
  (see section 3, Step 5) — a readability polish item, not a safety one.
- Only Hindi is translated; Tamil, Telugu, and Malayalam are measured but deliberately not
  enabled (see section 8) — a genuine scope gap against the original brief, made
  explicitly for a measured reason (latency) rather than left unstated.
- Translation quality has not had a human review either. Machine back-translation found
  one real error class on its own (steatosis/stenosis confusion — section 8), but
  back-translation is an imperfect check; a fluent speaker has not reviewed the Hindi
  output.
- The demo page is unauthenticated and has no rate limiting — anyone who can reach the
  server can submit reports and trigger LLM/translation/speech calls that cost real CPU
  time and (for speech) send report-derived text to Google's TTS endpoint.
- Speech sends the validated explanation text to a third-party hosted API (Google
  Translate's TTS endpoint) to be spoken — unlike generation and translation, which run
  entirely locally. A production deployment handling real patient data would need either
  a self-hosted TTS engine or an explicit, disclosed exception to "nothing leaves this
  server," not the silent one this prototype currently has.
- This is a prototype, not a clinical product: it is not HIPAA-compliant and stores
  nothing by default. Generation already runs on a local Llama 3 rather than a hosted
  API, which avoids sending report text to a third party — but Ollama itself has no
  authentication or encryption on its default local port, so a real deployment still
  needs the network-level protections a production system would add regardless.
- The semantic grounding check (section 7) is deliberately advisory-only — it was
  measured and found unable to safely gate on its own (overlapping score ranges between
  faithful and mismatched sentences). It adds a logged signal, not a stronger guarantee;
  the structural citation check remains the only thing standing between the model and the
  patient.
- Audit logging (section 7), when an operator enables it, stores the full original report
  text and every generated sentence in a local JSON-lines file with no access control, no
  retention policy, and no encryption at rest — appropriate for a prototype's local
  development use, not for a real deployment handling real patient reports.

---

## 14. Interview preparation

**Why not just ask ChatGPT to explain the report?**
It can invent facts, drift into diagnosis, and misread negation. This system constrains
every claim to a cited knowledge base entry, checks it, and falls back to verbatim
definitions when a check fails.

**Walk me through the user experience.**
The patient pastes a report and sees it unchanged, with terms highlighted three ways:
explained, not found, and unchecked. Tapping a term shows its definition and source.
Below that is a one-line-per-term explanation ending with a disclaimer. Technically, every
highlight comes from absolute character offsets kept through each stage, and the
explanation is validated before it's shown. Terms the system can't vouch for become a
list of questions for the doctor instead of guesses. (Section 3 has the full walkthrough.)

**How do you prevent hallucination?**
Four layers: the generator (Llama 3, via Ollama) only sees retrieved KB text; it must tag
each sentence with the KB id it used, and the validator checks those ids against what was
actually retrieved; the model is kept out of the two cases that most need to be exactly
right — negated findings and coverage gaps — which stay on fixed, tested wording instead;
and any failure, whether unsafe content or the LLM being unreachable, falls back to the
deterministic template. Terms with no KB entry are reported as gaps rather than explained.

**Why use both a dictionary and an NER model?**
The dictionary is exact for known terms; NER generalises to unknown ones. On our corpus,
NER alone reaches 76.6% F1, and the combination reaches 96% recall and 88.1% F1.

**What was the hardest bug?**
The NER model returned subword fragments and truncated terms with default aggregation,
and it was context-sensitive: it found terms in full reports but not in isolated sentences.
Switching to `first` aggregation and running over the full report with a sliding window
fixed both. Gap detection rose from 3/7 to 5/7.

**How do you handle negation?**
NegEx via negspacy, judged per sentence, with two custom changes: added phrasings like
"is absent", and removed "rule out", because that signals uncertainty, not absence.

**How would you scale the knowledge base?**
Import from public-domain sources, draft with an LLM, approve with a human, prioritise by
frequency, and feed logged coverage gaps back into curation.

**Why doesn't retrieval explain terms the dictionary misses, if it can find them?**
It can find them — Recall@3 is 100% on paraphrases — but a similarity score can't tell
a true match from a related-but-wrong one. "renal" scores higher against hydronephrosis
(0.634) than the weakest genuine paraphrase match scores against its correct entry
(0.412), so no threshold separates them. Explanations use only a term's own curated KB
link; semantic neighbours become suggestions for a human curator instead.

**How would you evaluate it properly?**
Real NER and retrieval metrics (done), grounding and readability checks on generated
explanations (done — `eval/generation_eval.py`), a human faithfulness review of those
explanations (not yet done — this is different from grounding: a sentence can cite the
right source and still subtly misstate it), and ideally a before/after comprehension test
with volunteers, reporting measured and projected results separately.

**Why does only part of the explanation come from the LLM?**
Because testing showed it shouldn't write all of it. Asked to phrase a negated finding,
it explained the condition before saying it was absent — the exact mistake the project
can't afford. So the model only writes sentences for terms that are cleanly explained and
not negated; coverage gaps and negated findings keep their tested, fixed wording. It's a
narrower job for the model, chosen because a wider one was tried and got the highest-risk
case wrong.

**Is the safety filter too strict, or about right?**
Measured, not guessed: 3 of 17 real LLM explanations from the eval corpus fell back to
the template, every time because of `forbidden_diagnosis` catching "you have [finding]"
phrasing that was actually safe — the model rephrasing "extra fat in the liver" as "you
have extra fat in your liver." The regex can't tell that apart from a real diagnosis, so
it blocks both. That's a real cost — a noticeably higher fallback rate than a smarter
check would need — but it's the right trade for a first version: it never has to trust
the model's judgment on the one thing that most needs to be right. A structure-aware
check (roadmap step 6) is the way to bring that rate down without loosening the filter.

**What happens if the LLM is slow or down?**
Generation is wrapped separately from the safety check: an unreachable Ollama server, a
timeout, or a response that won't parse all raise the same `LLMUnavailableError` and fall
back to the template immediately, flagged as `generation_unavailable` for audit — same
outcome as unsafe LLM output failing validation, just a different, earlier trigger. The
one thing that's genuinely slow is generation itself: 10-20s per explanation on an 8B
model with no GPU, measured, not estimated.

**Why is the semantic grounding check advisory instead of enforced?**
Because I measured before building, not after: on real LLM output over the eval corpus,
a faithful sentence's similarity to its own definition ranged 0.265-0.970, and a sentence
paired with a wrong-but-plausible definition ranged up to 0.495 — the ranges overlap, so
no fixed threshold separates them without both false positives and false negatives. The
same failure mode as retrieval's "no threshold is safe" finding, for the same reason:
similarity measures relatedness, not equivalence. So it logs a signal for a human
reviewer instead of gating the response, and the structural citation check — which is
exact, not fuzzy — stays the only thing that can reject an explanation.

**Why is only one language translated, when the brief asked for several?**
Because I measured the cost of the others before promising them. NLLB translates Tamil,
Telugu, and Malayalam through the exact same code path as Hindi — turning them on is a
one-line change — but each one costs the same CPU latency as Hindi does: seconds to tens
of seconds per sentence, worse for longer text. Shipping all four silently would have
meant a demo where selecting a language sometimes takes over a minute with no explanation.
I built one language properly, with the cost measured and disclosed
(`eval/translation_eval.py`), rather than four languages with an unstated cost.

**Walk me through a real bug you found testing your own work.**
Building the demo page, I put each highlighted term's definition inline in the report
text via a line break, so it would sit right under the term. Testing it in a browser
showed the actual defect: it fractured the patient's own sentences onto broken lines —
"No hydronephrosis" then a stray period alone on the next line. That's not cosmetic on
this project specifically — "show the patient's own report, unchanged" is a stated
design principle (section 3.4), and I'd broken it. Fixed by moving the definition into
one shared panel below the report instead of inline, so the report text is never touched.
I wouldn't have caught it from reading the code; it only showed up once I actually
clicked through the running page.

**Why was speech easy when translation was hard?**
Different cost shape, not less engineering care. NLLB translation runs the model locally
on this machine's CPU, so its cost scales with hardware I don't control — measured at
5.6-14.7 seconds per sentence, the reason only one language ships. gTTS is a hosted API
call: Google's servers do the work, so it came back under a second regardless of load
here. That's also its real cost, just a different kind: report-derived text now leaves
this server to a third party, something generation and translation deliberately don't do.
Cheap latency and free privacy aren't the same thing, and the README says so rather than
only reporting the speed.

**What would production need?**
HTTPS, no persistent storage of raw reports by default (this build already defaults to
that; audit logging is opt-in but has no access control, retention limit, or encryption
at rest yet), an on-premises LLM (already true here — Llama 3 via local Ollama), GPU
acceleration for translation (CPU latency is workable for a demo, not for a real product),
a self-hosted TTS engine instead of a third-party API call for speech, clinician review of
the KB and of the Hindi translation output, authentication and rate limiting on the demo
page, and a much larger, multi-annotator evaluation.
