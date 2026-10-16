"""Scripted mock LLM for checking the loop without a GPU (`run.py --mock-llm`).

Generator: small random SEARCH/REPLACE edits of the current program shown in the prompt (perturb a numeric
literal, insert comment lines, occasionally a compile-breaking line), plus some responses without a usable
diff. The edited programs are evaluated for real, so scores and validity change. Estimator / Librarian:
well-formed (occasionally malformed) JSON.
"""
from __future__ import annotations

import json
import random
import re
import threading

from .llm import MockLLM

_PROG_RE = re.compile(r"# Current Program\n```cpp\n(.*?)\n```\n\n# Task", re.DOTALL)
_NUM_RE = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def _block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def scripted_llm(seed: int = 0, p_no_diff: float = 0.08, p_no_match: float = 0.08, p_compile_error: float = 0.08,
                 p_bad_json: float = 0.1) -> MockLLM:
    rng = random.Random(seed)
    lock = threading.Lock()

    def generator(user: str) -> str:
        code = _PROG_RE.search(user).group(1)
        lines = code.split("\n")
        r = rng.random()
        if r < p_no_diff:
            return "The program looks fine; I would keep it as it is."
        if r < p_no_diff + p_no_match:
            return "Tweak:\n" + _block("int this_line_does_not_exist = 0;", "int x = 1;")
        body = [i for i, ln in enumerate(lines) if ln.strip() and not ln.lstrip().startswith(("//", "#"))
                and lines.count(ln) == 1]
        i = rng.choice(body)
        if r < p_no_diff + p_no_match + p_compile_error:
            return "Refactor:\n" + _block(lines[i], lines[i] + "\n    this is not C++;")
        out = []
        nums = [j for j in body if _NUM_RE.search(lines[j]) and "include" not in lines[j]]
        if nums and rng.random() < 0.7:
            j = rng.choice(nums)
            ms = list(_NUM_RE.finditer(lines[j]))
            m = rng.choice(ms)
            v = int(m.group(1))
            nv = max(0, v + rng.choice([-2, -1, 1, 2, max(1, v // 4)]))
            out.append(_block(lines[j], lines[j][:m.start()] + str(nv) + lines[j][m.end():]))
        if not out or rng.random() < 0.5:
            k = rng.choice(body)
            note = "\n".join(f"    // mock note {rng.randint(0, 10**6)}" + "." * rng.randint(0, 60)
                             for _ in range(rng.randint(1, 12)))
            if all(lines[k] not in b for b in out):
                out.append(_block(lines[k], lines[k] + "\n" + note))
        return "Here are my changes.\n\n" + "\n".join(out)

    def respond(tag, messages):
        user = messages[-1]["content"]
        with lock:
            if tag == "generator":
                return generator(user)
            if rng.random() < p_bad_json:
                return "I am not sure; maybe +1."  # unparseable
            ids = re.findall(r'"id": "(b\d+)"', user)
            if tag in ("estimator", "estimator_emptyB"):
                return json.dumps({"cited_beliefs": rng.sample(ids, min(len(ids), rng.randint(0, 2))),
                                   "p_valid": round(rng.uniform(0.5, 1.0), 2),
                                   "delta_hat": round(rng.uniform(-2, 3), 2), "reason": "mock reason"})
            if tag == "librarian":
                ops = [{"op": "ADD", "statement": f"mock belief {rng.randint(0, 999)}", "conditions": "always",
                        "predicted_effect": "+1", "confidence": 0.5, "evidence": "for", "reason": "mock"}]
                if ids:
                    ops.append({"op": "UPDATE", "id": rng.choice(ids), "confidence": round(rng.random(), 2),
                                "evidence": "against", "reason": "mock"})
                if len(ids) >= 2 and rng.random() < 0.3:
                    ops.append({"op": "MERGE", "ids": ids[:2], "statement": "merged", "conditions": "c",
                                "predicted_effect": "0", "confidence": 0.4, "reason": "mock"})
                if len(ids) >= 25:
                    ops.append({"op": "DELETE", "id": ids[0], "reason": "keep the document under the cap"})
                return json.dumps({"analysis": "mock analysis", "ops": ops})
            raise ValueError(f"unknown tag {tag}")

    return MockLLM(respond, model="mock-scripted")
