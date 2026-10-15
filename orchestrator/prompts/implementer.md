{{problem}}

## Current solution (score {{score}})
```cpp
{{code}}
```

## Plan to implement
{{plan}}

## Task
Rewrite the program to implement the plan faithfully. Requirements:
- Output one complete, self-contained C++17 program (a single file) that reads stdin and writes stdout in the required format. Do not omit any part of the file.
- It must ALWAYS produce a valid packing for every input (keep a simple, safe fallback if your new logic might fail).
- Deterministic: if you use randomness, use a fixed seed (e.g. std::mt19937 rng(12345)); never seed from time or addresses.
- Prefer iteration-count budgets over wall-clock budgets. If you use a wall-clock budget, the whole run must finish within 1.5 seconds per test case even for n = 10000.
- Memory below 256 MB. No threads, files, or system calls.

Answer in exactly this format and nothing else:
<what_changed>one sentence summarizing what you changed</what_changed>
```cpp
// complete program
```
