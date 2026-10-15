{{problem}}

## Current solution (score {{score}})
```cpp
{{code}}
```

## Recent iterations (oldest first)
{{history}}
{{belief_section}}
## Task
Propose {{K}} candidate modification plans for the current solution. Each plan must be:
- a concrete change that an engineer can implement in one rewrite of this file (say what to change in the algorithm and how),
- clearly different from the other plans (different mechanisms, not variations of one idea),
- at most 150 words, including the mechanism by which it should change the score.
Plans may be bold or incremental. Do not propose things already tried unless you change them meaningfully.

Output ONLY a JSON object of this form:
{"plans": [{"id": "p1", "plan": "..."}, ...]}
with exactly {{K}} plans and ids p1..p{{K}}.
