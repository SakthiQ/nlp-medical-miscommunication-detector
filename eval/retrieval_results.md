# Retrieval results

Model `sentence-transformers/all-MiniLM-L6-v2`, FAISS inner product over normalised vectors (cosine), 46 indexed KB entries (term, aliases and definition).
Regenerate with `python -m eval.retrieval_eval`.

## Paraphrase retrieval

| Metric | Value |
| --- | ---: |
| Recall@1 | 10/12 (83.3%) |
| Recall@3 | 12/12 (100.0%) |
| Mean reciprocal rank | 0.917 |
| Score of the correct entry (min / median) | 0.412 / 0.623 |

| Query | Expected | Rank | Top 3 |
| --- | --- | ---: | --- |
| fat in the liver | kb-001 | 1 | kb-001 hepatic steatosis (0.662), kb-042 hepatomegaly (0.508), kb-019 triglycerides (0.365) |
| stones in the gallbladder | kb-007 | 1 | kb-007 cholelithiasis (0.672), kb-006 nephrolithiasis (0.517), kb-034 gastroesophageal reflux disease (0.404) |
| low red blood cell count | kb-021 | 2 | kb-024 thrombocytopenia (0.493), kb-021 anemia (0.447), kb-023 leukocytosis (0.381) |
| fluid around the lung | kb-010 | 1 | kb-010 pleural effusion (0.672), kb-009 atelectasis (0.503), kb-041 hydronephrosis (0.419) |
| thin fragile bones | kb-026 | 1 | kb-026 osteoporosis (0.555), kb-025 osteopenia (0.416), kb-028 spondylosis (0.339) |
| average blood sugar test | kb-004 | 1 | kb-004 hemoglobin A1c (0.593), kb-003 aspartate aminotransferase (0.268), kb-022 hemoglobin (0.251) |
| reduced kidney filtering | kb-016 | 2 | kb-041 hydronephrosis (0.450), kb-016 estimated glomerular filtration rate (0.412), kb-015 creatinine (0.386) |
| swollen ankles | kb-011 | 1 | kb-011 edema (0.669), kb-041 hydronephrosis (0.453), kb-044 lymphadenopathy (0.305) |
| underactive thyroid gland | kb-032 | 1 | kb-032 hypothyroidism (0.731), kb-033 thyroid-stimulating hormone (0.438), kb-044 lymphadenopathy (0.322) |
| heart looks bigger than normal | kb-008 | 1 | kb-008 cardiomegaly (0.626), kb-036 normal sinus rhythm (0.344), kb-042 hepatomegaly (0.333) |
| urine backed up in the kidney | kb-041 | 1 | kb-041 hydronephrosis (0.620), kb-039 proteinuria (0.467), kb-015 creatinine (0.447) |
| bad cholesterol | kb-017 | 1 | kb-017 low-density lipoprotein cholesterol (0.602), kb-020 dyslipidemia (0.503), kb-018 high-density lipoprotein cholesterol (0.479) |

## Could a threshold let retrieval explain unvetted terms?

Each unrelated term's nearest entry. None of these entries is a correct explanation.

| Term | Nearest entry | Score |
| --- | --- | ---: |
| consolidation | kb-009 atelectasis | 0.129 |
| renal | kb-041 hydronephrosis | 0.634 |
| canal stenosis | kb-012 dyspnea | 0.285 |
| Mediastinal | kb-010 pleural effusion | 0.186 |
| femoral neck | kb-028 spondylosis | 0.300 |
| antral | kb-008 cardiomegaly | 0.196 |
| Afebrile | kb-009 atelectasis | 0.187 |
| Bone density scan | kb-025 osteopenia | 0.426 |
| Urinalysis | kb-041 hydronephrosis | 0.538 |
| ECG | kb-013 tachycardia | 0.351 |

The highest-scoring unrelated term, "renal", reaches 0.634 against hydronephrosis. A threshold above that would accept only 5 of 12 paraphrases, while the weakest correct match scores 0.412. The score ranges overlap, so no threshold is safe.

This is expected: embedding similarity measures relatedness, not equivalence. "renal" is about the kidney, so it lands near a kidney condition, but it is not that condition. The system therefore never explains a term through a semantic match. Explanations use exact KB links only; semantic neighbours are returned as `curation_suggestions` for a human curator to confirm or reject.
