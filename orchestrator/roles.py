"""LLM roles: the Generator (OpenEvolve's diff prompt and SEARCH/REPLACE format) and, for potential estimation
(PE), the Estimator and the Librarian. Each call is an independent request to the same model.

Every call is logged to calls.jsonl with the full prompt, the raw output (content and reasoning), the parsed
result and token usage. Estimator / Librarian output that fails to parse is retried `parse_retries` times with
a fresh seed; Generator output is never retried (a response without a usable diff is a generation failure).
A failed request (server unreachable after the client's retries) raises LLMError, which aborts the iteration.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
import time
from pathlib import Path
from typing import Callable

from .beliefs import BeliefDoc, check_fields
from .db import Program
from .llm import Sampling
from .parsing import ParseError, extract_json, strip_think, to_float

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_TPL_RE = re.compile(r"\{\{(\w+)\}\}")
ROLE_NAMES = {"generator": "Generator", "estimator": "Estimator", "librarian": "Librarian"}


class LLMError(RuntimeError):
    """The model server failed (not the candidate's fault): the iteration is aborted and redone on resume."""


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


def fmt_score(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def fmt_delta(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.4f}"


class RoleRunner:
    """Holds clients/sampling/logging; `step`, `b_version`, `iter_uid` tag every call."""

    def __init__(self, clients: dict, sampling: dict[str, Sampling], logger, problem_text: str,
                 run_seed: int, parse_retries: int = 1, max_context_tokens: int = 131072):
        self.clients = clients            # role -> client (falls back to clients["default"])
        self.sampling = sampling          # role -> Sampling
        self.logger = logger
        self.problem = problem_text
        self.run_seed = run_seed
        self.parse_retries = parse_retries
        self.max_context = max_context_tokens
        self.step, self.b_version, self.iter_uid = 0, 0, ""

    def begin(self, step: int, b_version: int, iter_uid: str) -> None:
        self.step, self.b_version, self.iter_uid = step, b_version, iter_uid

    def call(self, role: str, user: str, parse: Callable[[str], object], idx: int = 0, system: str | None = None,
             log_role: str | None = None, retries: int | None = None, extra_log: dict | None = None):
        """Returns (parsed or None, last response or None)."""
        client = self.clients.get(role) or self.clients["default"]
        system = system if system is not None else render("system", role_name=ROLE_NAMES[role])
        log_role = log_role or role
        # keep prompt + completion inside the context window (~3 characters per token for code-heavy prompts)
        est_in = (len(system) + len(user)) // 3
        sampling = dataclasses.replace(self.sampling[role], max_tokens=min(
            self.sampling[role].max_tokens, self.max_context - est_in - 512))
        resp, err = None, ""
        for attempt in range((self.parse_retries if retries is None else retries) + 1):
            content = user if attempt == 0 else (
                user + f"\n\n(Your previous answer could not be used: {err}. Answer strictly in the requested "
                "output format.)")
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            seed = stable_seed(self.run_seed, self.step, log_role, idx, attempt)
            rec = {"role": log_role, "iter": self.step, "iter_uid": self.iter_uid, "idx": idx, "attempt": attempt,
                   "B_version": self.b_version, "seed": seed, "sampling": sampling.to_dict(), "messages": messages,
                   **(extra_log or {})}
            if sampling.max_tokens < 4096:
                self.logger.log_call({**rec, "parse_ok": False, "parse_error": "prompt_too_long", "tokens_in": 0,
                                      "tokens_out": 0, "est_prompt_tokens": est_in})
                return None, None
            t0 = time.time()
            try:
                resp = client.chat(messages, sampling, seed=seed, tag=log_role)
            except Exception as e:  # noqa: BLE001
                self.logger.log_call({**rec, "parse_ok": False, "parse_error": f"request failed: {e!r}"[:2000],
                                      "tokens_in": 0, "tokens_out": 0, "latency_s": round(time.time() - t0, 3)})
                raise LLMError(f"{log_role} request failed: {e!r}") from e
            rec.update(model=resp.model, content=resp.content, reasoning=resp.reasoning,
                       finish_reason=resp.finish_reason, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                       latency_s=round(time.time() - t0, 3))
            try:
                parsed = parse(resp.content)
                self.logger.log_call({**rec, "parse_ok": True, "parse_error": None, "parsed": parsed})
                return parsed, resp
            except (ParseError, ValueError, KeyError, TypeError) as e:
                err = str(e)[:1000]
                self.logger.log_call({**rec, "parse_ok": False, "parse_error": f"{type(e).__name__}: {e}",
                                      "parsed": None})
        return None, resp


# ---------------------------------------------------------------- SEARCH/REPLACE diffs (OpenEvolve code_utils)

DIFF_RE = re.compile(r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE", re.DOTALL)
MARKER_RE = re.compile(r"^[ \t]*(<<<<<<< SEARCH|=======|>>>>>>> REPLACE)[ \t]*\r?$", re.MULTILINE)


def extract_diffs(text: str) -> list[tuple[str, str]]:
    """(search, replace) blocks; ValueError on a delimiter outside a block or a malformed block."""
    matches = list(DIFF_RE.finditer(text))
    cursor = 0
    for m in matches:
        if MARKER_RE.search(text[cursor:m.start()]):
            raise ValueError("Unmatched SEARCH/REPLACE delimiter outside a diff block")
        if [x.group(1) for x in MARKER_RE.finditer(m.group(0))] != ["<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"]:
            raise ValueError("Malformed SEARCH/REPLACE delimiter sequence")
        cursor = m.end()
    if MARKER_RE.search(text[cursor:]):
        raise ValueError("Unmatched SEARCH/REPLACE delimiter outside a diff block")
    return [(m.group(1).rstrip(), m.group(2).rstrip()) for m in matches]


def _find(lines: list[str], search: list[str]) -> int:
    n = len(search)
    for i in range(len(lines) - n + 1):
        if lines[i:i + n] == search:
            return i
    stripped = [s.rstrip() for s in search]
    for i in range(len(lines) - n + 1):
        if [s.rstrip() for s in lines[i:i + n]] == stripped:
            return i
    return -1


def apply_diff_blocks(code: str, blocks: list[tuple[str, str]]) -> tuple[str, int]:
    lines, applied = code.split("\n"), 0
    for search, replace in blocks:
        s = search.split("\n")
        i = _find(lines, s)
        if i >= 0:
            lines[i:i + len(s)] = replace.split("\n")
            applied += 1
    return "\n".join(lines), applied


def _block_lines(lines: list[str], max_len: int = 100, max_lines: int = 30) -> str:
    out = ["  " + (s.rstrip()[:max_len - 3] + "..." if len(s.rstrip()) > max_len else s.rstrip())
           for s in lines[:max_lines]]
    if len(lines) > max_lines:
        out.append(f"  ... ({len(lines) - max_lines} more lines)")
    return "\n".join(out) if out else "  (empty)"


def format_diff_summary(blocks: list[tuple[str, str]]) -> str:
    out = []
    for i, (s, r) in enumerate(blocks):
        sl, rl = s.strip().split("\n"), r.strip().split("\n")
        if len(sl) == 1 and len(rl) == 1:
            out.append(f"Change {i + 1}: '{sl[0]}' to '{rl[0]}'")
        else:
            out.append(f"Change {i + 1}: Replace:\n{_block_lines(sl)}\nwith:\n{_block_lines(rl)}")
    return "\n".join(out)


def parse_generation(parent_code: str, text: str, max_code_length: int) -> dict:
    """Generator response -> {gen_status, code, n_blocks, n_applied, changes}."""
    out = {"gen_status": "ok", "code": None, "n_blocks": 0, "n_applied": 0, "changes": ""}
    try:
        blocks = extract_diffs(strip_think(text or ""))
    except ValueError as e:
        return {**out, "gen_status": "no_valid_diff", "detail": str(e)}
    out["n_blocks"] = len(blocks)
    if not blocks:
        return {**out, "gen_status": "no_valid_diff"}
    code, applied = apply_diff_blocks(parent_code, blocks)
    out["n_applied"] = applied
    if applied == 0:
        return {**out, "gen_status": "search_not_matching"}
    if code == parent_code:
        return {**out, "gen_status": "no_change"}
    if len(code) > max_code_length:
        return {**out, "gen_status": "code_too_long", "detail": f"{len(code)} > {max_code_length}"}
    return {**out, "code": code, "changes": format_diff_summary(blocks)}


# ---------------------------------------------------------------- Generator prompt (OpenEvolve prompt sampler)

def render_artifacts(art: dict | None, max_chars: int = 20480) -> str:
    if not art:
        return ""
    secs = []
    for k, v in art.items():
        s = str(v)
        secs.append(f"### {k}\n```\n{s[:max_chars] + chr(10) + '... (truncated)' if len(s) > max_chars else s}\n```")
    return "## Last Execution Output\n\n" + "\n\n".join(secs)


def _program_type(p: Program) -> str:
    if p.migrant:
        return "Migrant"
    s = p.score
    return ("High-Performer" if s >= 0.8 else "Alternative" if s >= 0.6 else "Experimental" if s >= 0.4
            else "Exploratory")


def _unique_features(p: Program, limit: int = 3) -> str:
    f = []
    if p.changes and len(p.changes) < 100:
        f.append("Modification:")   # OpenEvolve's fragment has no {changes} placeholder; reproduced as is
    if p.score >= 0.9:
        f.append(f"Excellent combined_score ({p.score:.3f})")
    elif p.score <= 0.3:
        f.append("Alternative combined_score approach")
    low, nl = p.code.lower(), len(p.code.split("\n"))
    if "for" in low and "while" in low:
        f.append("Mixed iteration strategies")
    if nl <= 10:
        f.append("Concise implementation")
    elif nl >= 50:
        f.append("Comprehensive implementation")
    return ", ".join(f[:limit]) or "approach to the problem"


def generator_prompt(rr: RoleRunner, parent: Program, inspirations: list[Program], island_top: list[Program],
                     rng, cfg: dict) -> tuple[str, str]:
    """(system, user). island_top: the parent island's programs by score (OpenEvolve: previous attempts = top
    num_top; top programs = top num_top + num_diverse, of which the last num_diverse are sampled as 'diverse')."""
    nt, nd = int(cfg["num_top_programs"]), int(cfg["num_diverse_programs"])
    prev, top = island_top[:nt], island_top[:nt + nd]
    areas = []
    if prev:
        a, b = prev[-1].score, parent.score
        areas.append(f"Fitness improved: {a:.4f} → {b:.4f}" if b > a else
                     f"Fitness declined: {a:.4f} → {b:.4f}. Consider revising recent changes." if b < a else
                     f"Fitness unchanged at {b:.4f}")
    lim = cfg.get("suggest_simplification_after_chars")
    if lim and len(parent.code) > lim:
        areas.append(f"Consider simplifying - code length exceeds {lim} characters")
    areas = areas or ["Focus on improving fitness while maintaining diversity"]

    attempts = []
    for i, p in enumerate(reversed(prev)):
        ps = p.parent_score if p.parent_score is not None else 0.0
        outcome = ("Improvement in all metrics" if p.score > ps else "Regression in all metrics" if p.score < ps
                   else "Mixed results")
        attempts.append(f"### Attempt {len(prev) - i}\n- Changes: {p.changes or 'Unknown changes'}\n"
                        f"- Metrics: combined_score: {p.score:.4f}\n- Outcome: {outcome}")
    tops = [f"### Program {i + 1} (Score: {p.score:.4f})\n```cpp\n{p.code}\n```\n"
            f"Key features: Performs well on combined_score ({p.score:.4f})" for i, p in enumerate(top[:nt])]
    diverse = rng.sample(top[nt:], min(nd, len(top[nt:]))) if nd > 0 and len(top) > nt else []
    top_txt = "\n\n".join(tops)
    if diverse:
        top_txt += "\n\n## Diverse Programs\n\n" + "\n\n".join(
            f"### Program D{i + 1} (Score: {p.score:.4f})\n```cpp\n{p.code}\n```\n"
            "Key features: Alternative approach to combined_score" for i, p in enumerate(diverse))
    shown = {p.id for p in top[:nt]} | {p.id for p in diverse}
    insp = [p for p in inspirations if p.id not in shown]
    insp_txt = ""
    if insp:
        insp_txt = ("## Inspiration Programs\n\nThese programs represent diverse approaches and creative solutions "
                    "that may inspire new ideas:\n\n" + "\n\n".join(
                        f"### Inspiration {i + 1} (Score: {p.score:.4f}, Type: {_program_type(p)})\n```cpp\n"
                        f"{p.code}\n```\nUnique approach: {_unique_features(p)}" for i, p in enumerate(insp)))
    system = render("generator_system", problem=rr.problem)
    user = render("generator", fitness=f"{parent.score:.4f}", focus_areas="\n".join(f"- {a}" for a in areas),
                  artifacts=render_artifacts(parent.artifacts, int(cfg.get("max_artifact_chars", 20480))),
                  previous_attempts="\n\n".join(attempts), top_programs=top_txt, inspirations=insp_txt,
                  code=parent.code)
    return system, user


def generate(rr: RoleRunner, system: str, user: str, parent_code: str, idx: int, max_code_length: int) -> dict:
    _, resp = rr.call("generator", user, lambda t: t, idx=idx, system=system, retries=0)
    if resp is None:
        return {"gen_status": "prompt_too_long", "code": None, "n_blocks": 0, "n_applied": 0, "changes": "",
                "finish_reason": None, "tokens_out": 0}
    out = parse_generation(parent_code, resp.content, max_code_length)
    out.update(finish_reason=resp.finish_reason, tokens_out=resp.tokens_out)
    return out


# ---------------------------------------------------------------- Estimator / Librarian (PE)

def belief_section(belief: BeliefDoc | None) -> str:
    if belief is None:
        return ""
    return ("\n## Belief document (lessons recorded so far about which modifications change the "
            "score and by how much; it may be incomplete or wrong)\n" + belief.render() + "\n")


def parse_estimate(text: str) -> dict:
    obj = extract_json(text)
    if "delta_hat" not in obj:
        raise ParseError("missing delta_hat")
    cited = obj.get("cited_beliefs", obj.get("relevant_beliefs")) or []
    if isinstance(cited, str):
        cited = [cited]
    cited = [str(r.get("id", r)) if isinstance(r, dict) else str(r) for r in cited]
    pv = obj.get("p_valid")
    p_valid = None
    if pv is not None:
        p_valid = to_float(pv)
        p_valid = min(1.0, max(0.0, p_valid / 100.0 if p_valid > 1.0 else p_valid))
    return {"delta_hat": to_float(obj["delta_hat"]), "p_valid": p_valid,
            "reason": str(obj.get("reason", "")).strip(), "cited_beliefs": cited}


def estimate(rr: RoleRunner, parent: Program, diff: str, stage1_text: str, belief: BeliefDoc | None, idx: int,
             log_role: str = "estimator") -> dict | None:
    user = render("estimator_diff", problem=rr.problem, score=fmt_score(parent.score), code=parent.code.rstrip(),
                  artifacts=render_artifacts(parent.artifacts) or "(none)", belief_section=belief_section(belief),
                  diff=diff, stage1=stage1_text)
    est, _ = rr.call("estimator", user, parse_estimate, idx=idx, log_role=log_role)
    return est


def parse_librarian(text: str, strict: bool = True) -> dict:
    """strict: reject the answer if any op breaks the belief format (the retry tells the Librarian why)."""
    obj = extract_json(text)
    ops = obj.get("ops", [])
    if not isinstance(ops, list):
        raise ParseError("'ops' must be a list")
    if strict:
        errs = [f"op {i} ({op.get('op')}): {'; '.join(e)}" for i, op in enumerate(ops) if isinstance(op, dict)
                and str(op.get("op", "")).upper() != "DELETE" and (e := check_fields(op))]
        if errs:
            raise ParseError("belief format violated - " + " | ".join(errs) + ". Keep each belief one short general "
                             "claim; put the analysis in 'analysis' and measurements in 'evidence_note'")
    return {"analysis": str(obj.get("analysis", "")).strip(), "ops": ops}


def case_evidence_text(cmp: dict | None) -> str:
    if not cmp:
        return "(per-case comparison unavailable)"
    return "\n".join([
        f"- Output identical to the parent on all {cmp['n']} cases (every per-case density equal): "
        f"{'YES' if cmp['identical'] else 'NO'}",
        f"- Cases improved / worsened / unchanged: {cmp['improved']} / {cmp['worsened']} / {cmp['unchanged']}",
        f"- Per-case score change (points): mean {cmp['mean_delta']:+.4f}, min {cmp['min_delta']:+.4f}, "
        f"max {cmp['max_delta']:+.4f}"])


def code_evidence_text(code_diff: dict | None) -> str:
    if not code_diff:
        return "(no code change information)"
    funcs = ", ".join(code_diff["changed_functions"]) or "(none identified)"
    out = (f"- Diff size: +{code_diff['added']} / -{code_diff['removed']} lines "
           f"({code_diff['old_lines']} -> {code_diff['new_lines']} lines)\n- Changed functions: {funcs}")
    if code_diff.get("diff"):
        out += f"\n- Unified diff{' (truncated)' if code_diff.get('truncated') else ''}:\n```diff\n{code_diff['diff']}\n```"
    return out


def librarian(rr: RoleRunner, parent: Program, belief: BeliefDoc, sel: dict, tau: float, case_evidence: str,
              code_evidence: str) -> dict | None:
    """sel: the selected candidate's record (delta_hat, p_valid, reason, cited_beliefs, delta, score, status, e)."""
    cited = belief.subset(sel.get("cited_beliefs") or [])
    user = render("librarian", problem=rr.problem, score=fmt_score(parent.score), code=parent.code.rstrip(),
                  beliefs=belief.render(), delta_hat=fmt_delta(sel["delta_hat"]),
                  p_valid="n/a" if sel.get("p_valid") is None else f"{sel['p_valid']:.2f}",
                  estimator_reason=sel.get("reason") or "(none)",
                  cited_beliefs="; ".join(f"[{b['id']}] {b['statement']}" for b in cited) or "(none)",
                  delta=fmt_delta(sel["delta"]), new_score=fmt_score(sel["score"]), status=sel["status"],
                  error=fmt_delta(sel["e"]), trigger_note=f" (|e| > tau = {tau:g}: the prediction was significantly off)",
                  case_evidence=case_evidence, code_evidence=code_evidence, cap=belief.cap, n_beliefs=len(belief))
    lib, resp = rr.call("librarian", user, parse_librarian)
    if lib is None and resp is not None:  # still violating after the retry: apply its well-formed ops only
        try:
            lib = parse_librarian(resp.content, strict=False)
            lib["format_violation"] = True
        except (ParseError, ValueError):
            lib = None
    return lib
