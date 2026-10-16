{{problem}}

## Parent program (score {{score}})
```cpp
{{code}}
```

## Belief document (current)
{{beliefs}}

## What happened this iteration
- A candidate modification of the parent program was generated (diff below) and evaluated.
- Predicted delta (delta_hat): {{delta_hat}}; predicted probability that the full evaluation is valid: {{p_valid}}
- Estimator's reasoning: {{estimator_reason}}
- Beliefs the estimator cited: {{cited_beliefs}}
- Actual delta: {{delta}} (score {{score}} -> {{new_score}}; full evaluation status: {{status}})
- Prediction error e = delta - delta_hat = {{error}}{{trigger_note}}

## Evidence from the evaluation (70 test cases)
{{case_evidence}}

## Code change (candidate vs. the parent program above)
{{code_evidence}}

## Task
You maintain the belief document. Its purpose is to make FUTURE predictions of score changes more accurate for modifications that have not been tried yet. First analyze why the prediction was off: was a belief wrong or too confident, was a belief applied outside its conditions, or was a belief missing? Then edit the document with a short list of operations (usually 1-3; an empty list is allowed if nothing should change).

Rules:
- Each belief must be falsifiable: under which conditions, which kind of modification, what expected effect on the score (with a magnitude, in score points).
- Prefer general lessons about kinds of modifications over facts about this single change.
- At most {{cap}} beliefs in total (currently {{n_beliefs}}); MERGE or DELETE before adding when the document is nearly full.
- "evidence" says whether this iteration supports ("for") or contradicts ("against") the belief.

Allowed operations (each needs a one-sentence "reason"):
- {"op": "ADD", "statement": "...", "conditions": "...", "predicted_effect": "...", "confidence": 0.0-1.0, "evidence": "for"|"against", "reason": "..."}
- {"op": "UPDATE", "id": "b001", <any of: "statement", "conditions", "predicted_effect", "confidence">, "evidence": "for"|"against"|"none", "reason": "..."}
- {"op": "MERGE", "ids": ["b001", "b002"], "statement": "...", "conditions": "...", "predicted_effect": "...", "confidence": 0.0-1.0, "evidence": "for"|"against"|"none", "reason": "..."}
- {"op": "DELETE", "id": "b001", "reason": "..."}

Output ONLY a JSON object of this form:
{"analysis": "<what the prediction got wrong and why>", "ops": [ ... ]}
