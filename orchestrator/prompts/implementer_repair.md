{{problem}}

## Plan being implemented
{{plan}}

## Your previous program
```cpp
{{code}}
```

## Evaluation result (it compiled, but the submission failed)
{{failure_report}}

## Task
Fix the program so that it produces a VALID packing within the limits on every test case, while keeping the implementation of the plan. Typical causes:
- "Overlap at cell (x,y)": two pieces share a cell - the occupancy bookkeeping does not match the cells actually output (check the reflect -> rotate -> translate convention: reflect (x,y)->(-x,y), one clockwise step (x,y)->(y,-x), then add (X,Y)).
- "Out of bounds": a transformed cell falls outside [0,W) x [0,H) - recompute W and H from the final placement, or normalize each oriented shape so its minimum x and y are 0 before choosing (X,Y).
- Timeouts: the algorithm is too slow for n = 10000 pieces - reduce the search (fewer candidates, early exit) and keep total runtime under 1.5 s.
A robust pattern is to place cells into an explicit occupancy grid, assert no conflict before committing a placement, and compute W and H from the occupied cells at the end.

Same requirements as before: one complete self-contained C++17 file, valid output for every input, fixed random seed, no threads/files/system calls, at most 1.5 s wall clock if you use a time budget, memory below 256 MB.

Answer in exactly this format and nothing else:
<what_changed>one sentence describing this repair</what_changed>
```cpp
// complete program
```
