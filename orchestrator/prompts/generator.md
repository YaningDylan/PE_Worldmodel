# Current Program Information
- Fitness: {{fitness}}
- Focus areas: {{focus_areas}}

{{artifacts}}{{belief_section}}

# Program Evolution History
## Previous Attempts

{{previous_attempts}}

## Top Performing Programs

{{top_programs}}

{{inspirations}}

# Current Program
```cpp
{{code}}
```

# Task
Suggest improvements to the program that will improve its FITNESS SCORE.
The system maintains diversity across these dimensions: complexity, diversity
Different solutions with similar fitness but different features are valuable.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
