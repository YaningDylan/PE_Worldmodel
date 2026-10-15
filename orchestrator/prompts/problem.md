## Problem: Polyomino Packing with reflections (Frontier-CS algorithmic problem 0)

Given n polyominoes (100 <= n <= 10000; each has 1..10 unit cells, 4-connected, possibly with holes), place ALL of them - each with an optional reflection, a rotation by a multiple of 90 degrees, and an integer translation - inside one axis-aligned W x H rectangle with no overlapping cells, minimizing the area W*H.

Input (stdin): line 1: n. Then for each piece i = 1..n: a line with k_i, followed by k_i lines "x y" (cell coordinates in the piece's local frame; may be negative).

Output (stdout): line 1: "W H". Then exactly n lines "X_i Y_i R_i F_i", one per piece, in input order.
- F_i in {0,1}: if 1, first reflect every cell across the y-axis: (x, y) -> (-x, y).
- R_i in {0,1,2,3}: then rotate R_i times by 90 degrees clockwise about the origin; one step maps (x, y) -> (y, -x).
- Then translate: final cell = (x + X_i, y + Y_i).
- Every final cell must satisfy 0 <= x < W and 0 <= y < H, and no two cells may coincide.

Test data: 70 hidden test cases. Each piece is drawn uniformly at random (with replacement) from the set of all polyomino shapes of size 1..10; n = 10^p with p ~ Uniform(2, 4), so n ranges from 100 to 10000.

Limits per test case: 2 s time, 256 MB memory. The submission is one C++17 file compiled with `g++ -O2 -pipe -std=gnu++17`.

Scoring: for each test case, density = (sum of k_i) / (W*H), a number in (0, 1]. {{score_scale}} Higher is better. If ANY test case fails (overlap, out of bounds, malformed output, crash, time or memory limit), the whole submission is invalid.
