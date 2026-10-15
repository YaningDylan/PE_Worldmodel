"""Scripted mock LLM that produces well-formed (and occasionally malformed) role outputs.

Used by `run.py --mock` and the unit tests to exercise the whole loop without a GPU.
Implementer programs carry `// MOCK_SCORE=` / `// MOCK_STATUS=` markers read by MockEvaluator.
"""
from __future__ import annotations

import json
import random
import re
import threading

from .llm import MockLLM

PROGRAM = """#include <bits/stdc++.h>
// MOCK_SCORE={score:.4f}
// {status}
int main() {{ return 0; }}
"""


def scripted_llm(seed: int = 0, p_bad_json: float = 0.1, p_compile_error: float = 0.15,
                 p_invalid: float = 0.05) -> MockLLM:
    rng = random.Random(seed)
    lock = threading.Lock()

    def respond(tag, messages):
        user = messages[-1]["content"]
        with lock:
            r = rng.random()
            if tag in ("proposer", "estimator", "selector", "librarian") and r < p_bad_json:
                return "Sorry, here is my answer: plans are p1 and p2."  # unparseable
            if tag == "proposer":
                k = int(re.search(r"Propose (\d+) candidate", user).group(1))
                return json.dumps({"plans": [{"id": f"p{i + 1}", "plan": f"mock plan {rng.randint(0, 9999)}"}
                                             for i in range(k)]})
            if tag == "estimator":
                ids = re.findall(r'"id": "(b\d+)"', user)
                rel = rng.sample(ids, min(len(ids), rng.randint(0, 2)))
                return json.dumps({"relevant_beliefs": rel, "delta_hat": round(rng.uniform(-2, 3), 2),
                                   "reason": "mock reason"})
            if tag == "selector":
                ids = re.findall(r"^### (p\d+)$", user, re.M)
                return "```json\n" + json.dumps({"chosen_id": rng.choice(ids), "rationale": "mock"}) + "\n```"
            if tag in ("implementer", "implementer_fix", "implementer_repair"):
                m = re.search(r"Current solution \(score ([-0-9.]+)\)", user)
                base = float(m.group(1)) if m else 50.0
                status = "MOCK_STATUS=ok"
                if tag == "implementer" and r < p_compile_error:
                    status = "MOCK_STATUS=compile_error"
                elif tag == "implementer" and r > 1 - p_invalid:
                    status = "MOCK_STATUS=invalid"
                code = PROGRAM.format(score=base + rng.gauss(0.3, 2.0), status=status)
                return f"<what_changed>mock change</what_changed>\n```cpp\n{code}```"
            if tag == "librarian":
                ids = re.findall(r'"id": "(b\d+)"', user)
                ops = [{"op": "ADD", "statement": f"mock belief {rng.randint(0, 999)}",
                        "conditions": "always", "predicted_effect": "+1", "confidence": 0.5,
                        "evidence": "for", "reason": "mock"}]
                if ids:
                    ops.append({"op": "UPDATE", "id": rng.choice(ids), "confidence": round(rng.random(), 2),
                                "evidence": "against", "reason": "mock"})
                if len(ids) >= 2 and rng.random() < 0.3:
                    ops.append({"op": "MERGE", "ids": ids[:2], "statement": "merged", "conditions": "c",
                                "predicted_effect": "0", "confidence": 0.4, "reason": "mock"})
                if rng.random() < 0.2:
                    ops.append({"op": "DELETE", "id": "b999", "reason": "invalid id on purpose"})
                return json.dumps({"analysis": "mock analysis", "ops": ops})
            raise ValueError(f"unknown tag {tag}")

    return MockLLM(respond, model="mock-scripted")
