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
- A belief is ONE short, general, falsifiable claim about a KIND of modification: under which conditions, which kind of change, and the expected effect on the score in points. Example:
  {"statement": "For cases with n > 3000, placing small pieces (k <= 2) last to fill gaps beats placing by descending area", "conditions": "the current solution is a skyline-type packer", "predicted_effect": "+1 to +3"}
- Length limits: "statement" at most 200 characters, "conditions" at most 150, "predicted_effect" at most 40 and it must be a score change in points (e.g. "+1 to +3", "-0.5 to 0", "about 0"). Answers that break a limit are rejected.
- Do NOT put into a belief: what happened in this iteration, measured timings or case sizes, names of variables or functions of this program, instructions to the estimator, or your analysis. The analysis goes into "analysis"; one concrete measurement may go into "evidence_note" (at most 150 characters, e.g. "n=2198 took 2.26 s"), which is kept as evidence but not shown in prompts.
- UPDATE a belief by rewriting it as a better short claim (narrow its conditions, change its effect or confidence); do not append to it.
- At most {{cap}} beliefs in total (currently {{n_beliefs}}); MERGE or DELETE before adding when the document is nearly full.
- "evidence" says whether this iteration supports ("for") or contradicts ("against") the belief.

Allowed operations (each needs a one-sentence "reason"; "evidence_note" is optional):
- {"op": "ADD", "statement": "...", "conditions": "...", "predicted_effect": "...", "confidence": 0.0-1.0, "evidence": "for"|"against", "evidence_note": "...", "reason": "..."}
- {"op": "UPDATE", "id": "b001", <any of: "statement", "conditions", "predicted_effect", "confidence">, "evidence": "for"|"against"|"none", "evidence_note": "...", "reason": "..."}
- {"op": "MERGE", "ids": ["b001", "b002"], "statement": "...", "conditions": "...", "predicted_effect": "...", "confidence": 0.0-1.0, "evidence": "for"|"against"|"none", "evidence_note": "...", "reason": "..."}
- {"op": "DELETE", "id": "b001", "reason": "..."}

Output ONLY a JSON object of this form:
{"analysis": "<what the prediction got wrong and why>", "ops": [ ... ]}
