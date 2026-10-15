"""The five roles (plan section 5.4). Each role is an independent call to the same model.

Every call is logged to calls.jsonl with the full prompt, the raw output (content and
reasoning), the parsed result and token usage. Output that fails to parse is retried
`parse_retries` times with a fresh seed; the final failure is returned as None.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .beliefs import BeliefDoc
from .llm import Sampling
from .parsing import ParseError, extract_cpp, extract_json, extract_tag, to_float

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_TPL_RE = re.compile(r"\{\{(\w+)\}\}")
ROLE_NAMES = {"proposer": "Proposer", "estimator": "Estimator", "selector": "Selector",
              "implementer": "Implementer", "implementer_fix": "Implementer",
              "implementer_repair": "Implementer", "librarian": "Librarian"}


def render(name: str, **kw) -> str:
    tpl = (PROMPT_DIR / f"{name}.md").read_text()
    missing = sorted(set(_TPL_RE.findall(tpl)) - set(kw))
    if missing:
        raise KeyError(f"template {name} missing values for {missing}")
    # single pass: substituted values (e.g. C++ code with '{{') are never re-scanned
    return _TPL_RE.sub(lambda m: str(kw[m.group(1)]), tpl)


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


@dataclass
class Solution:
    code: str
    score: float


def fmt_score(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def fmt_delta(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.4f}"


class RoleRunner:
    """Holds clients/sampling/logging; `step`, `b_version`, `iter_uid` tag every call."""

    def __init__(self, clients: dict, sampling: dict[str, Sampling], logger, problem_text: str,
                 run_seed: int, parse_retries: int = 1):
        self.clients = clients            # role -> client (falls back to clients["default"])
        self.sampling = sampling          # role -> Sampling
        self.logger = logger
        self.problem = problem_text
        self.run_seed = run_seed
        self.parse_retries = parse_retries
        self.step = 0
        self.b_version = 0
        self.iter_uid = ""

    def begin(self, step: int, b_version: int, iter_uid: str) -> None:
        self.step, self.b_version, self.iter_uid = step, b_version, iter_uid

    def _client(self, role: str):
        return self.clients.get(role) or self.clients["default"]

    def call(self, role: str, user: str, parse: Callable[[str], object], idx: int = 0,
             extra_log: dict | None = None):
        client = self._client(role)
        sampling = self.sampling[role]
        system = render("system", role_name=ROLE_NAMES[role])
        for attempt in range(self.parse_retries + 1):
            content = user if attempt == 0 else (
                user + "\n\n(Your previous answer could not be parsed. Answer strictly in the requested output format.)")
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            seed = stable_seed(self.run_seed, self.step, role, idx, attempt)
            t0 = time.time()
            resp = client.chat(messages, sampling, seed=seed, tag=role)
            rec = {"role": role, "iter": self.step, "iter_uid": self.iter_uid, "idx": idx,
                   "attempt": attempt, "model": resp.model, "B_version": self.b_version,
                   "seed": seed, "sampling": sampling.to_dict(), "messages": messages,
                   "content": resp.content, "reasoning": resp.reasoning,
                   "finish_reason": resp.finish_reason, "tokens_in": resp.tokens_in,
                   "tokens_out": resp.tokens_out, "latency_s": round(time.time() - t0, 3)}
            if extra_log:
                rec.update(extra_log)
            try:
                parsed = parse(resp.content)
                rec.update(parse_ok=True, parse_error=None, parsed=parsed)
                self.logger.log_call(rec)
                return parsed
            except (ParseError, ValueError, KeyError, TypeError) as e:
                rec.update(parse_ok=False, parse_error=f"{type(e).__name__}: {e}", parsed=None)
                self.logger.log_call(rec)
        return None


# ---------------------------------------------------------------- parsers

def parse_plans(K: int) -> Callable[[str], list[dict]]:
    def _p(text: str) -> list[dict]:
        obj = extract_json(text)
        plans = obj.get("plans")
        if not isinstance(plans, list):
            raise ParseError("'plans' must be a list")
        out = []
        for p in plans:
            if isinstance(p, str):
                body = p
            elif isinstance(p, dict):
                body = p.get("plan")
                if not isinstance(body, str):
                    body = " ".join(str(v) for k, v in p.items() if k != "id" and isinstance(v, str))
            else:
                continue
            body = (body or "").strip()
            if body:
                out.append({"id": f"p{len(out) + 1}", "plan": body,
                            "orig_id": p.get("id") if isinstance(p, dict) else None})
        if not out:
            raise ParseError("no usable plans")
        return out[:K]
    return _p


def parse_estimate(text: str) -> dict:
    obj = extract_json(text)
    if "delta_hat" not in obj:
        raise ParseError("missing delta_hat")
    rel = obj.get("relevant_beliefs") or []
    if isinstance(rel, str):
        rel = [rel]
    rel = [str(r.get("id", r)) if isinstance(r, dict) else str(r) for r in rel]
    return {"relevant_beliefs": rel, "delta_hat": to_float(obj["delta_hat"]),
            "reason": str(obj.get("reason", "")).strip()}


def parse_selection(valid_ids: list[str]) -> Callable[[str], dict]:
    def _p(text: str) -> dict:
        obj = extract_json(text)
        raw = str(obj.get("chosen_id", "")).strip()
        cid = match_plan_id(raw, valid_ids)
        if cid is None:
            raise ParseError(f"chosen_id {raw!r} not in {valid_ids}")
        return {"chosen_id": cid, "rationale": str(obj.get("rationale", "")).strip()}
    return _p


def match_plan_id(raw: str, valid_ids: list[str]) -> str | None:
    if raw in valid_ids:
        return raw
    m = re.search(r"(\d+)", raw)
    if m and f"p{int(m.group(1))}" in valid_ids:
        return f"p{int(m.group(1))}"
    return None


def parse_impl(text: str) -> dict:
    code = extract_cpp(text)
    if code is None:
        raise ParseError("no C++ program found")
    return {"code": code, "what_changed": extract_tag(text, "what_changed") or ""}


def parse_librarian(text: str) -> dict:
    obj = extract_json(text)
    ops = obj.get("ops", [])
    if not isinstance(ops, list):
        raise ParseError("'ops' must be a list")
    return {"analysis": str(obj.get("analysis", "")).strip(), "ops": ops}


# ---------------------------------------------------------------- prompt pieces

def belief_section(belief: BeliefDoc | None) -> str:
    if belief is None:
        return ""
    return ("\n## Belief document (lessons recorded so far about which modifications change the "
            "score and by how much; it may be incomplete or wrong)\n" + belief.render() + "\n")


def history_text(history: list[dict], show_estimates: bool) -> str:
    if not history:
        return "(none yet - this is the first iteration)"
    lines = []
    for h in history:
        s = f"- iter {h['iter']}: plan: {h['plan']}"
        if show_estimates and h.get("delta_hat") is not None:
            s += f" | predicted delta: {fmt_delta(h['delta_hat'])}"
        if h.get("status") != "ok":
            s += f" | result: implementation failed ({h.get('status')})"
        else:
            s += (f" | actual delta: {fmt_delta(h['delta'])} -> "
                  + ("accepted as new best" if h.get("accepted") else "not accepted"))
        lines.append(s)
    return "\n".join(lines)


def candidates_text(plans: list[dict], estimates: dict | None) -> str:
    blocks = []
    for p in plans:
        b = f"### {p['id']}\n{p['plan']}"
        if estimates is not None:
            est = estimates.get(p["id"])
            if est is None:
                b += "\n- predicted delta: (estimate unavailable)"
            else:
                b += (f"\n- predicted delta: {fmt_delta(est['delta_hat'])}"
                      f"\n- estimator reasoning: {est['reason']}")
                if est.get("relevant_beliefs"):
                    b += f"\n- beliefs cited: {', '.join(est['relevant_beliefs'])}"
        blocks.append(b)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- roles

def propose(rr: RoleRunner, sol: Solution, K: int, history: list[dict], belief: BeliefDoc | None,
            show_estimates: bool) -> list[dict] | None:
    user = render("proposer", problem=rr.problem, score=fmt_score(sol.score), code=sol.code.rstrip(),
                  history=history_text(history, show_estimates), belief_section=belief_section(belief),
                  K=K)
    return rr.call("proposer", user, parse_plans(K))


def estimate(rr: RoleRunner, sol: Solution, plan: dict, belief: BeliefDoc | None, idx: int = 0
             ) -> dict | None:
    user = render("estimator", problem=rr.problem, score=fmt_score(sol.score), code=sol.code.rstrip(),
                  belief_section=belief_section(belief), plan=plan["plan"])
    return rr.call("estimator", user, parse_estimate, idx=idx, extra_log={"plan_id": plan["id"]})


def select(rr: RoleRunner, sol: Solution, plans: list[dict], estimates: dict | None,
           belief: BeliefDoc | None, remaining: int) -> dict | None:
    hint = ("Each plan comes with a predicted score change (delta_hat) and the estimator's reasoning. "
            "Predictions can be wrong; weigh them together with the belief document and your own judgement."
            if estimates is not None else "")
    user = render("selector", problem=rr.problem, score=fmt_score(sol.score), code=sol.code.rstrip(),
                  belief_section=belief_section(belief), candidates=candidates_text(plans, estimates),
                  remaining=remaining, selector_hint=hint)
    return rr.call("selector", user, parse_selection([p["id"] for p in plans]))


def implement(rr: RoleRunner, sol: Solution, plan: dict) -> dict | None:
    user = render("implementer", problem=rr.problem, score=fmt_score(sol.score),
                  code=sol.code.rstrip(), plan=plan["plan"])
    return rr.call("implementer", user, parse_impl)


def implement_fix(rr: RoleRunner, plan: dict, code: str, compile_log: str, idx: int) -> dict | None:
    user = render("implementer_fix", problem=rr.problem, plan=plan["plan"], code=code.rstrip(),
                  compile_log=(compile_log or "").strip()[-6000:])
    return rr.call("implementer_fix", user, parse_impl, idx=idx)


def implement_repair(rr: RoleRunner, plan: dict, code: str, failure_report: str, idx: int) -> dict | None:
    """Repair after a compiled program failed on the judge (invalid output / timeout / crash)."""
    user = render("implementer_repair", problem=rr.problem, plan=plan["plan"], code=code.rstrip(),
                  failure_report=failure_report)
    return rr.call("implementer_repair", user, parse_impl, idx=idx)


def case_evidence_text(cmp: dict | None) -> str:
    if not cmp:
        return "(per-case comparison unavailable)"
    lines = [f"- Output identical to the previous solution on all {cmp['n']} cases (every per-case density equal): "
             f"{'YES' if cmp['identical'] else 'NO'}",
             f"- Cases improved / worsened / unchanged: {cmp['improved']} / {cmp['worsened']} / {cmp['unchanged']}",
             f"- Per-case score change (points): mean {cmp['mean_delta']:+.4f}, min {cmp['min_delta']:+.4f}, "
             f"max {cmp['max_delta']:+.4f}"]
    return "\n".join(lines)


def code_evidence_text(code_diff: dict | None) -> str:
    if not code_diff:
        return "(no code change information)"
    funcs = ", ".join(code_diff["changed_functions"]) or "(none identified)"
    out = (f"- Diff size: +{code_diff['added']} / -{code_diff['removed']} lines "
           f"({code_diff['old_lines']} -> {code_diff['new_lines']} lines)\n- Changed functions: {funcs}")
    if code_diff.get("diff"):
        out += f"\n- Unified diff{' (truncated)' if code_diff.get('truncated') else ''}:\n```diff\n{code_diff['diff']}\n```"
    return out


def librarian(rr: RoleRunner, sol: Solution, belief: BeliefDoc, rec: dict, tau: float, rule: str) -> dict | None:
    cited = belief.subset(rec.get("relevant_beliefs") or [])
    cited_txt = "; ".join(f"[{b['id']}] {b['statement']}" for b in cited) or "(none)"
    note = (f" (|e| > tau = {tau:g}: the prediction was significantly off)" if rule == "error" else "")
    user = render("librarian", problem=rr.problem, score=fmt_score(sol.score), code=sol.code.rstrip(),
                  beliefs=belief.render(), plan=rec["chosen_plan"], what_changed=rec.get("what_changed") or "(none)",
                  delta_hat=fmt_delta(rec["delta_hat"]), estimator_reason=rec.get("estimator_reason") or "(none)",
                  cited_beliefs=cited_txt, delta=fmt_delta(rec["delta"]), new_score=fmt_score(rec["f_x_new"]),
                  error=fmt_delta(rec["e"]), trigger_note=note,
                  case_evidence=case_evidence_text(rec.get("case_cmp")),
                  code_evidence=code_evidence_text(rec.get("code_diff")),
                  cap=belief.cap, n_beliefs=len(belief))
    return rr.call("librarian", user, parse_librarian)
