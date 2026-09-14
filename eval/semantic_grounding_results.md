# Semantic grounding results

Model `sentence-transformers/all-MiniLM-L6-v2`. Measures whether embedding similarity between a generated sentence and its cited KB definition can distinguish a faithful paraphrase from a sentence paired with a plausible but wrong definition.
Regenerate with `python -m eval.semantic_grounding_eval`. Requires Ollama running.

| Set | Stats |
| --- | --- |
| Faithful (sentence vs. its own cited definition) | n=42  min=0.265  median=0.719  max=0.970 |
| Mismatched (sentence vs. a random other definition) | n=42  min=-0.000  median=0.165  max=0.495 |

**The ranges overlap: worst faithful score (0.265) is below the worst mismatched score (0.495).** No fixed threshold separates the two sets without both false positives (rejecting real paraphrases) and false negatives (missing wrong-definition sentences).

At the advisory threshold used in code (0.25), 0/42 faithful sentences would be flagged (false positives if this were an enforced cutoff instead of an advisory one).

## Lowest-scoring faithful pairs (would be flagged first if enforced)

| Report | Score | KB id | Sentence |
| --- | ---: | --- | --- |
| ev-007 | 0.265 | kb-045 | The gallbladder stone was an unexpected discovery. |
| ev-017 | 0.420 | kb-046 | The liver looked normal. |
| ev-001 | 0.428 | kb-002 | Your liver enzyme levels are higher than normal, which could be a sign that your liver is irritated. |
| ev-013 | 0.519 | kb-029 | The pulmonary nodule in the right upper lobe is not cancerous. |
| ev-010 | 0.532 | kb-038 | Your report says "ischemia" was not found. For reference, it means: An area of the body is not getting as much blood flow as it needs. |
| ev-013 | 0.548 | kb-031 | A pulmonary nodule is a small rounded lump of tissue in the lung. |
| ev-009 | 0.559 | kb-033 | Your thyroid-stimulating hormone level is higher than normal, which means your thyroid gland is being signaled to work more than usual. |
| ev-012 | 0.569 | kb-040 | A special test called microalbuminuria would be helpful to get more information. |

## Highest-scoring mismatched pairs (would slip past an enforced cutoff)

| Report | Score | Wrong KB id | Sentence |
| --- | ---: | --- | --- |
| ev-006 | 0.495 | kb-019 | A blood test measured the amount of hemoglobin in your red blood cells, which is the part that carries oxygen. |
| ev-015 | 0.431 | kb-037 | The lining of the stomach is irritated or inflamed. |
| ev-010 | 0.388 | kb-005 | The heart's electrical pattern is following its normal path at a normal rate. |
| ev-004 | 0.369 | kb-021 | You may be feeling like you're not getting enough air, which is known as dyspnea. |
| ev-007 | 0.351 | kb-041 | A hard, stone-like lump was found in the gallbladder. |
| ev-010 | 0.343 | kb-045 | Your report says "ischemia" was not found. For reference, it means: An area of the body is not getting as much blood flow as it needs. |
| ev-011 | 0.252 | kb-032 | The liver enzyme AST was also found to be higher than usual, which can be a sign that the liver is irritated. |
| ev-001 | 0.249 | kb-026 | You have extra fat built up inside your liver, which is a common finding that can be detected by chance on a scan. |

## Conclusion

Same finding as retrieval (step 4), for the same reason: embedding similarity measures relatedness, not equivalence. This is why `app/safety/semantic.py`'s check is advisory-only — logged for a human reviewer, never used to reject an explanation on its own. The structural citation check in `app/safety/validator.py` remains the sole gate.
