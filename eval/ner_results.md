# Term detection results

Measured on `data/eval_corpus.json` (18 synthetic reports, hand-annotated) with `d4data/biomedical-ner-all`, aggregation `first`, confidence threshold 0.8.
Regenerate with `python -m eval.ner_eval`.

| View | Gold terms | Recall (overlap) | Recall (exact span) | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NER model alone | 55 | 74.5% | 67.3% | 78.8% | 76.6% |
| Coverage-gap detection | 7 | 71.4% | 57.1% | n/a | n/a |
| Terminology stage (dictionary + NER) | 50 | 96.0% | 94.0% | 81.4% | 88.1% |

Precision here is strict: the gold labels mark only the terms a patient most needs explained, so a correct medical term the annotator left unlabelled (for example "Urinalysis") counts against it.

## NER model misses

- ev-001: ALT
- ev-003: consolidation
- ev-004: edema
- ev-004: hypertension
- ev-005: dyslipidemia
- ev-006: anemia
- ev-007: incidentally
- ev-010: ischemia
- ev-013: benign
- ev-014: osteoporosis
- ev-015: GERD
- ev-016: Afebrile
- ev-016: cough
- ev-017: unremarkable

## Coverage gaps not detected

- ev-003: consolidation
- ev-016: Afebrile

## Terms flagged that are not in the gold labels

- ev-002: Fasting glucose
- ev-003: basilar
- ev-004: pedal
- ev-005: Lipid profile
- ev-006: Complete blood count
- ev-007: calculus
- ev-010: ECG
- ev-012: Urinalysis
- ev-013: pulmonary
- ev-014: Bone density scan
- ev-015: Endoscopy
