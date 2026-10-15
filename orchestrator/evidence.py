"""Evidence derived from evaluations and code, used in logs, repair prompts and the Librarian's input."""
from __future__ import annotations

import difflib
import functools
import re

from .config import PE_ROOT

TESTDATA = PE_ROOT / "third_party" / "Frontier-CS" / "algorithmic" / "problems" / "0" / "testdata"
_EPS = 1e-12


@functools.lru_cache(maxsize=1)
def case_sizes() -> dict[int, int]:
    """Number of pieces n in each test case (first token of {case}.in)."""
    out = {}
    for p in TESTDATA.glob("*.in"):
        try:
            with open(p) as f:
                out[int(p.stem)] = int(f.readline().split()[0])
        except (ValueError, IndexError, OSError):
            pass
    return out


def compare_cases(parent: dict | None, new: dict | None) -> dict | None:
    """Per-case density comparison of two OK evaluations (score points = 100 * density)."""
    if not parent or not new or parent.get("status") != "ok" or new.get("status") != "ok":
        return None
    pc = {c["case"]: c.get("ratio") for c in parent.get("per_case") or []}
    nc = {c["case"]: c.get("ratio") for c in new.get("per_case") or []}
    common = sorted(k for k in pc if k in nc and pc[k] is not None and nc[k] is not None)
    if not common:
        return None
    d = [100.0 * (nc[k] - pc[k]) for k in common]
    improved = sum(x > 100 * _EPS for x in d)
    worsened = sum(x < -100 * _EPS for x in d)
    return {"n": len(common), "identical": improved == 0 and worsened == 0, "improved": improved,
            "worsened": worsened, "unchanged": len(common) - improved - worsened,
            "mean_delta": sum(d) / len(d), "min_delta": min(d), "max_delta": max(d)}


_FUNC_RE = re.compile(
    r"^\s{0,4}(?!(?:if|for|while|switch|return|else|do|case|sizeof|catch)\b)"
    r"(?:[\w:<>,\*&~\[\]]+\s+)+[\*&]?([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:\{.*)?$")
_LAMBDA_RE = re.compile(r"^\s*(?:auto|function<[^>]*>)\s+([A-Za-z_]\w*)\s*=\s*\[")
_STRUCT_RE = re.compile(r"^\s*(?:struct|class)\s+([A-Za-z_]\w*)")


def _enclosing(lines: list[str]) -> list[str | None]:
    """For every line, the name of the innermost function/lambda/struct header seen so far (heuristic)."""
    cur, out = None, []
    for ln in lines:
        m = _FUNC_RE.match(ln) or _LAMBDA_RE.match(ln) or _STRUCT_RE.match(ln)
        if m and not ln.strip().startswith(("//", "#")):
            cur = m.group(1)
        out.append(cur)
    return out


def code_diff(old: str, new: str, max_lines: int = 150) -> dict:
    a, b = old.splitlines(), new.splitlines()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ea, eb = _enclosing(a), _enclosing(b)
    added = removed = 0
    funcs: dict[str, None] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        removed += i2 - i1
        added += j2 - j1
        for name in ea[i1:i2] + eb[j1:j2]:
            if name:
                funcs[name] = None
    diff = list(difflib.unified_diff(a, b, "before.cpp", "after.cpp", lineterm="", n=2))
    return {"added": added, "removed": removed, "old_lines": len(a), "new_lines": len(b),
            "changed_functions": list(funcs), "diff": "\n".join(diff[:max_lines]) if max_lines > 0 else "",
            "truncated": len(diff) > max_lines > 0}


def failure_report(res: dict, max_cases: int = 5) -> str:
    """What the repair prompt shows after a compiled program failed: counts by kind, first checker
    messages (with coordinates), CPU time of timed-out cases, and each case's size n."""
    cases = res.get("per_case") or []
    sizes = case_sizes()
    counts = res.get("status_counts") or {}
    lines = [f"Overall status: {res.get('status')}. Per-case outcome counts over {len(cases)} cases: "
             + ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))]
    bad = [c for c in cases if c.get("status") != "ok"]
    inval = [c for c in bad if c.get("status") == "invalid"]
    slow = [c for c in bad if c.get("status") in ("timeout", "memory_limit")]
    other = [c for c in bad if c.get("status") not in ("invalid", "timeout", "memory_limit")]
    if inval:
        lines.append("First rejected outputs (checker message):")
        for c in inval[:max_cases]:
            lines.append(f"- case {c['case']} (n={sizes.get(c['case'], '?')}): {(c.get('msg') or '').strip()[:240]}")
    if slow:
        lines.append(f"Cases over the limit (2 s CPU, 256 MB): {len(slow)}; slowest:")
        for c in sorted(slow, key=lambda c: -(c.get('time_s') or 0))[:max_cases]:
            lines.append(f"- case {c['case']} (n={sizes.get(c['case'], '?')}): {c.get('status')}, "
                         f"CPU {c.get('time_s', 0):.2f} s, memory {c.get('mem_mb', 0):.0f} MB")
    if other:
        lines.append("Crashed cases:")
        for c in other[:max_cases]:
            lines.append(f"- case {c['case']} (n={sizes.get(c['case'], '?')}): {(c.get('msg') or '').strip()[:200]}")
    ok_sizes = [sizes.get(c["case"]) for c in cases if c.get("status") == "ok" and sizes.get(c["case"])]
    if ok_sizes:
        lines.append(f"Cases that passed: {len(ok_sizes)} (n from {min(ok_sizes)} to {max(ok_sizes)}).")
    return "\n".join(lines)
