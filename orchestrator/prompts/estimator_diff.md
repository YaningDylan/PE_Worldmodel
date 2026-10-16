{{problem}}

## Parent program (score {{score}})
```cpp
{{code}}
```

### Parent evaluation output
{{artifacts}}
{{belief_section}}
## Candidate modification (unified diff, parent -> candidate)
```diff
{{diff}}
```

## Candidate's quick check
{{stage1}}

## Task
Predict delta = (score of the candidate on the full evaluation over the 70 hidden test cases) - (parent score {{score}}), in score points on the same scale. The quick check only ran the 5 smallest public cases (n about 105); the full evaluation has n up to 10000 with 2 s and 256 MB per case. If any case fails (invalid packing, crash, time or memory limit), the candidate scores 0, i.e. delta = -{{score}}. delta_hat is your expected value of delta, including that risk; p_valid is your probability that the full evaluation is valid. A negative prediction is allowed.
First decide which beliefs (if any) are relevant to this modification and how they apply, then give your estimates and the reasoning behind them.

Output ONLY a JSON object of this form:
{"cited_beliefs": ["<belief id>", ...], "p_valid": <number in [0, 1]>, "delta_hat": <number>, "reason": "<why you expect this change>"}
