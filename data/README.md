# Data: knowledge base & evaluation corpus

These two files are built together on purpose. The KB defines what the system can
safely explain; the corpus defines what we measure it against. Curating them in
isolation is how you end up with an evaluation full of terms you never covered.

## Current state

| Metric | Value |
| --- | --- |
| KB entries | 51 (46 explainable, 5 commonly-understood) |
| Corpus snippets | 18, across 16 report types |
| Gold-annotated terms | 55 (48 covered, 7 deliberate gaps) |
| Corpus coverage | 87.3% |

## `knowledge_base.json`

One entry per term the system is allowed to explain.

| Field | Purpose |
| --- | --- |
| `id` | Stable citation handle. Each generated sentence is tagged with the id it used, so the safety layer can verify grounding. |
| `term` | Canonical form. |
| `aliases` | Every surface form the dictionary matcher should catch — abbreviations (`ALT`/`SGPT`), spelling variants (`anemia`/`anaemia`), and lay synonyms. |
| `label` | DISEASE / SYMPTOM / TEST / DRUG / BIOMARKER / ANATOMY / DESCRIPTOR. |
| `plain_definition` | Patient-facing text, 6th–8th grade reading level. This is the only text the generator may draw a claim from. |
| `clinical_context` | Non-interpretive framing (how the term is usually reported). Never a diagnosis. |
| `source_citation` | Human-readable source. Required and non-empty — enforced by test. |
| `source_url` | Verified permalink to the cited page: it resolves and supports the definition. `null` only where no public-domain source has been found yet. Do not invent URLs. |
| `is_common` | `true` = matched but never flagged for explanation. This replaces a separate difficulty classifier. |

### Curation rules

1. **No interpretation.** Definitions say what a term *means*, never what it means *for this patient*.
2. **Aliases must be globally unambiguous.** One alias may map to exactly one entry; the loader raises `KnowledgeBaseError` otherwise.
3. **Every entry needs a real citation.** If you cannot source it, it does not go in.
4. **Adding an entry means adding corpus coverage for it**, or it is untested.
5. **Do not copy MedlinePlus Medical Encyclopedia text.** It is licensed from A.D.A.M., not public domain. Write definitions in your own words, and cite public-domain sources: MedlinePlus Health Topics and Medical Tests (NLM), the NCI Dictionary of Cancer Terms, or other NIH institute pages. Five entries still cite the Encyclopedia because no public-domain page supporting them has been found yet: `kb-008` cardiomegaly, `kb-028` spondylosis, `kb-036` normal sinus rhythm, `kb-045` incidental finding, `kb-046` unremarkable.

### Adding entries from MedlinePlus

```bash
python -m tools.kb_candidates --download   # newest Health Topics XML + Medical Tests index -> data/raw/ (not committed)
python -m tools.kb_candidates              # -> data/kb_candidates.json, one draft per wanted term
python -m tools.kb_candidates --promote    # approved drafts -> knowledge_base.json
```

Wanted terms are every coverage gap in `eval_corpus.json` plus `kb_wishlist.txt`. They
are looked up first in MedlinePlus Medical Tests (so "sodium" means the blood test, not
the dietary-sodium topic), then in Health Topics. Each draft has the citation, URL and
suggested aliases filled in from the matching page, and that page's own text alongside
as reference. The tool never writes a
definition. A reviewer writes `plain_definition` and `clinical_context`, picks `label`,
removes any alias that isn't the same thing as the term, and sets `"approved": true`.

| Status | Meaning |
| --- | --- |
| `matched` | The term is a topic's title or one of its "also called" names. |
| `related_topic` | The term is only a "see" cross-reference on a broader topic (hyponatremia → Sodium). Its names are not offered as aliases, because they can be a different condition (that topic lists Hyperkalemia). |
| `partial` | No exact match; `suggestions` lists topics containing the term. |
| `not_found` | No MedlinePlus topic. Source it elsewhere (NCI dictionary, RadLex), or leave it as a gap. |
| `already_in_kb` | Nothing to add. |

Re-running keeps every draft a reviewer has touched. `--promote` validates the combined
KB (unique ids, unambiguous aliases) before writing anything, and then rule 4 still
applies: add a corpus snippet for each new entry.

## `eval_corpus.json`

Synthetic, non-PHI report snippets with hand-annotated gold terms.

A gold term with `kb_id: null` is a **deliberate coverage gap**. The correct system
behaviour is to detect the term and report that no vetted definition exists — not to
retrieve the nearest-looking entry and explain the wrong thing.

Current gaps: `consolidation`, `renal`, `canal stenosis`, `Mediastinal`,
`femoral neck`, `antral`, `Afebrile`.

## Validation

```bash
python -m pytest tests/test_knowledge_base.py -v
```

The suite cross-validates the two files: every `kb_id` resolves, every gold surface
occurs in its snippet, every covered term is reachable by dictionary lookup, every gap
is genuinely absent from the KB, and the corpus keeps enough gaps to exercise that path.
