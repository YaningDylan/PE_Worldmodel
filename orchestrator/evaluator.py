"""Evaluator adapters: the local judge-compatible evaluator (eval/local_eval.py) and a mock."""
from __future__ import annotations

import importlib.util
import re
import time
import traceback

from .config import PE_ROOT

OK = "ok"
FAIL_STATUSES = ("compile_error", "runtime_error", "timeout", "invalid", "no_code", "internal_error")
INFRA_RETRIES = 2  # internal_error = evaluator/infrastructure failure, not the candidate's fault


def _load_local_eval():
    path = PE_ROOT / "eval" / "local_eval.py"
    spec = importlib.util.spec_from_file_location("pe_local_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LocalEvaluator:
    def __init__(self, workers: int | None = None, time_limit_s: float = 2.0, mem_mb: int = 256):
        self.mod = _load_local_eval()
        self.kw = {"workers": workers, "time_limit_s": time_limit_s, "mem_mb": mem_mb}

    def evaluate(self, code: str, tag: str | None = None) -> dict:
        res: dict = {}
        for attempt in range(INFRA_RETRIES + 1):
            try:
                res = self.mod.evaluate_code(code, tag=tag, **self.kw)
            except Exception:  # noqa: BLE001 - never crash the loop
                res = {"status": "internal_error", "score": None, "per_case": [],
                       "stderr_tail": traceback.format_exc()[-4000:], "compile_log": ""}
            res["infra_attempts"] = attempt + 1
            res["judge"] = "local"
            if res.get("status") != "internal_error":
                break
        cases = res.get("per_case") or []
        if res.get("status") not in ("ok", "compile_error", "internal_error") and cases:
            counts: dict[str, int] = {}
            for c in cases:
                k = c.get("status", "runtime_error")
                counts[k] = counts.get(k, 0) + 1
            res["status_counts"] = counts
            res["status"] = submission_status(counts, len(cases))
        return res


TIME_LIMIT_NS = 2_000_000_000
MEM_LIMIT_B = 256 * 2**20


def classify_judge_case(c: dict) -> str:
    """Per-case status from the official judge's result fields.

    server.js overwrites every case's `status` with 'Correct'/'Wrong Answer' (by scoreRatio == 1), so
    the run verdict is recovered from the remaining fields: a parsed checker Ratio > 0 means a valid
    packing; a checker message means the output was rejected; otherwise the program did not finish
    normally (TLE by CPU time, MLE by memory, else runtime error). A case with time 0, memory 0 and no
    message never ran (known flake of the hosted judge) -> infra.
    """
    msg = c.get("msg") or ""
    if (c.get("scoreRatio") or 0) > 0:
        return "ok"
    if not msg and not c.get("time") and not c.get("memory"):
        return "infra"
    if re.search(r"wrong answer|Ratio:|points|FAIL|wrong output format|Unexpected end of file|Extra data", msg):
        return "invalid"
    if (c.get("time") or 0) >= TIME_LIMIT_NS:
        return "timeout"
    if (c.get("memory") or 0) >= MEM_LIMIT_B:
        return "memory_limit"
    return "runtime_error"


# per-case failure kind -> submission status; list order breaks ties in the majority vote
_CASE_TO_SUBMISSION = {"invalid": "invalid", "timeout": "timeout", "memory_limit": "runtime_error",
                       "runtime_error": "runtime_error"}
_TIE_ORDER = ["invalid", "timeout", "memory_limit", "runtime_error"]


def submission_status(counts: dict[str, int], n_cases: int) -> str:
    """ok iff every case is ok; any never-run case -> internal_error (retried); otherwise the most
    frequent failure kind (ties: invalid > timeout > memory > runtime). Per-kind counts are kept
    separately in `status_counts`."""
    if n_cases and counts.get("ok", 0) == n_cases:
        return "ok"
    if counts.get("infra"):
        return "internal_error"
    failing = [(counts.get(k, 0), -i, k) for i, k in enumerate(_TIE_ORDER) if counts.get(k)]
    if not failing:
        return "internal_error"
    return _CASE_TO_SUBMISSION[max(failing)[2]]


def judge_result_to_eval(res: dict, judge_type: str) -> dict:
    """Convert a server.js /result payload into the local_eval result format."""
    if res.get("status") != "done":
        err = str(res.get("error") or res.get("status"))
        # a real compile error carries g++ diagnostics; "compile failed: Time Limit Exceeded" etc. (no
        # diagnostics) means the compile step hit the judge's resource limits under load -> retry
        status = "compile_error" if re.search(r"error:", err) else "internal_error"
        return {"status": status, "score": None, "score_unbounded": None, "official_score": None, "per_case": [],
                "compile_log": err[-6000:], "stderr_tail": err[-2000:], "n_cases": 0, "judge": judge_type}
    per_case = []
    for i, c in enumerate(res.get("cases") or []):
        per_case.append({"case": i + 1, "status": classify_judge_case(c), "ratio": c.get("scoreRatio"),
                         "time_s": (c.get("time") or 0) / 1e9, "mem_mb": (c.get("memory") or 0) / 2**20,
                         "msg": (c.get("msg") or "")[:300]})
    counts: dict[str, int] = {}
    for c in per_case:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    status = submission_status(counts, len(per_case))
    bad = [c for c in per_case if c["status"] != "ok"][:5]
    return {"status": status, "score": res.get("score") if status == "ok" else None,
            "score_unbounded": res.get("scoreUnbounded") if status == "ok" else None,
            "official_score": res.get("score"), "per_case": per_case, "status_counts": counts,
            "n_cases": len(per_case), "compile_log": "", "judge": judge_type,
            "stderr_tail": "\n".join(f"[case {c['case']}: {c['status']}] {c['msg']}" for c in bad)}


class JudgeEvaluator:
    """Evaluate through a running Frontier-CS judge server (official Docker image or the native copy).

    All runs in a job share one judge; result dicts carry `judge` = judge_type.
    """

    def __init__(self, url: str, judge_type: str, timeout_s: float = 1800.0, poll_s: float = 0.5):
        import requests
        self.requests = requests
        self.url = url.rstrip("/")
        self.judge_type = judge_type
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    def _submit(self, code: str) -> dict:
        t0 = time.time()
        r = self.requests.post(f"{self.url}/submit", files={"code": ("solution.cpp", code)},
                               data={"pid": "0", "lang": "cpp"}, timeout=60)
        r.raise_for_status()
        sid = r.json()["sid"]
        while time.time() - t0 < self.timeout_s:
            rr = self.requests.get(f"{self.url}/result/{sid}", timeout=30)
            if rr.status_code == 200:
                res = rr.json()
                if res.get("status") in ("done", "error"):
                    res["_sid"], res["_wall_s"] = sid, time.time() - t0
                    return res
            time.sleep(self.poll_s)
        return {"status": "client_timeout", "_sid": sid, "_wall_s": time.time() - t0}

    def evaluate(self, code: str, tag: str | None = None) -> dict:
        out: dict = {}
        for attempt in range(INFRA_RETRIES + 1):
            try:
                raw = self._submit(code)
                out = judge_result_to_eval(raw, self.judge_type)
                out["sid"], out["eval_wall_s"] = raw.get("_sid"), round(raw.get("_wall_s", 0.0), 3)
            except Exception:  # noqa: BLE001 - never crash the loop
                out = {"status": "internal_error", "score": None, "per_case": [], "compile_log": "",
                       "stderr_tail": traceback.format_exc()[-4000:], "judge": self.judge_type}
            out["infra_attempts"] = attempt + 1
            out["tag"] = tag
            if out.get("status") != "internal_error":
                break
        return out


class MockEvaluator:
    """Score read from markers in the code: `// MOCK_SCORE=71.5`, `// MOCK_STATUS=compile_error`."""

    def __init__(self, default_score: float = 50.0):
        self.default_score = default_score
        self.calls: list[str] = []

    def evaluate(self, code: str, tag: str | None = None) -> dict:
        self.calls.append(code)
        st = re.search(r"MOCK_STATUS=(\w+)", code)
        if st and st.group(1) != OK:
            status = st.group(1)
            if status == "compile_error":
                return {"status": status, "score": None, "per_case": [], "stderr_tail": "mock failure",
                        "compile_log": "mock.cpp:1: error: mock", "judge": "mock"}
            per_case = [{"case": i, "status": status if i % 3 == 0 else "ok", "ratio": None if i % 3 == 0 else 0.5,
                         "time_s": 2.5 if status == "timeout" else 0.1, "mem_mb": 10.0,
                         "msg": "wrong answer Overlap at cell (3,4)" if status == "invalid" else ""}
                        for i in range(1, 71)]
            counts = {"ok": sum(c["status"] == "ok" for c in per_case)}
            counts[status] = 70 - counts["ok"]
            return {"status": status, "score": None, "per_case": per_case, "status_counts": counts,
                    "stderr_tail": "mock failure", "compile_log": "", "judge": "mock"}
        sc = re.search(r"MOCK_SCORE=([-+0-9.eE]+)", code)
        score = float(sc.group(1)) if sc else self.default_score
        per_case = [{"case": i, "status": OK, "ratio": score / 100.0, "time_s": 0.1, "mem_mb": 10.0, "msg": ""}
                    for i in range(1, 71)]
        return {"status": OK, "score": score, "score_unbounded": score, "per_case": per_case,
                "status_counts": {"ok": 70}, "stderr_tail": "", "compile_log": "", "n_cases": 70,
                "eval_wall_s": 0.0, "judge": "mock"}
