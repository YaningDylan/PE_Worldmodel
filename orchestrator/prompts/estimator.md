{{problem}}

## Current solution (score {{score}})
```cpp
{{code}}
```
{{belief_section}}
## Candidate plan
{{plan}}

## Task
Predict delta = (score of the program after this plan is implemented) - (current score {{score}}), in score points on the same scale as the scores above. Assume the plan is implemented faithfully and the program stays valid (implementation failures are accounted for separately). A negative prediction is allowed.
First decide which beliefs (if any) are relevant to this plan and how they apply, then give your point estimate and the reasoning behind it.

Output ONLY a JSON object of this form:
{"relevant_beliefs": ["<belief id>", ...], "delta_hat": <number>, "reason": "<why you expect this change>"}
