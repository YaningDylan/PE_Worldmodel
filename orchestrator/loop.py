"""The research loop (plan section 5.2):

  propose K plans -> estimate delta_hat per plan (reads B_t) -> model selects one ->
  implement -> evaluate, with <= max_compile_fixes repairs after compile errors and
  <= max_runtime_fixes repairs after invalid output / timeout / crash -> e = delta - delta_hat ->
  if the update rule fires, the Librarian edits B (structured ops) -> accept iff delta > margin.

Implementation failures consume budget but never produce an error e or a B update.
State is checkpointed after every iteration; re-running with the same run dir resumes.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import evidence, roles
from .beliefs import BeliefDoc
from .config import PE_ROOT, ROLES, resolve_path, sampling_for
from .roles import RoleRunner, Solution
from .runlog import RunLogger, code_hash


def _git_head(path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _eval_summary(res: dict) -> dict:
    return {k: res.get(k) for k in ("status", "score", "score_unbounded", "official_score", "n_cases",
                                    "eval_wall_s", "judge", "sid", "infra_attempts", "status_counts")}


RUNTIME_REPAIRABLE = ("invalid", "timeout", "runtime_error")


def infra_ids() -> dict:
    """Which job / vLLM server / judge produced a record (several runs share one server and judge)."""
    return {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "vllm_server": os.environ.get("PE_VLLM_RUN_DIR"), "vllm_variant": os.environ.get("PE_VLLM_VARIANT"),
            "judge_id": os.environ.get("PE_JUDGE_ID")}


class ResearchLoop:
    def __init__(self, cfg: dict, clients: dict, evaluator, run_dir, problem_text: str | None = None):
        self.cfg = cfg
        self.lp = cfg["loop"]
        self.T = int(cfg["run"]["budget_T"])
        self.log = RunLogger(run_dir)
        self.evaluator = evaluator
        self.clients = clients
        if problem_text is None:
            problem_text = roles.render("problem", score_scale=cfg["problem"]["score_scale"])
        sampling = {r: sampling_for(cfg, r) for r in ROLES}
        self.rr = RoleRunner(clients, sampling, self.log, problem_text, int(cfg["run"]["seed"]),
                             int(self.lp.get("parse_retries", 1)))
        self.state: dict | None = None
        self.belief: BeliefDoc | None = None
        self._pending_code: str | None = None

    # ------------------------------------------------------------ setup / resume
    def _initial_belief(self) -> BeliefDoc:
        cap = int(self.lp["belief_cap"])
        init = resolve_path(self.lp.get("init_belief"))
        if init is not None:
            b = BeliefDoc.load(init, cap)
            return BeliefDoc(b.beliefs, 0, b.next_id, cap)  # re-versioned from 0 for this run
        return BeliefDoc(cap=cap)

    def setup(self) -> None:
        st = self.log.read_json("state.json")
        if st is not None:
            self.state = st
            self.belief = BeliefDoc.load(self.log.b_dir / f"B_v{st['b_version']}.json",
                                         int(self.lp["belief_cap"]))
            self.state["resumes"] = st.get("resumes", 0) + 1
            return
        init_path = resolve_path(self.cfg["run"]["init_solution"])
        code = init_path.read_text()
        res = self.evaluator.evaluate(code, tag="init")
        h = self.log.save_solution(code, res)
        if res.get("status") != "ok":
            raise RuntimeError(f"initial solution failed to evaluate: {_eval_summary(res)}")
        self.belief = self._initial_belief()
        self.belief.save(self.log.b_dir)
        meta = {
            **infra_ids(), "judge_type": self.cfg["eval"].get("judge_type") if self.cfg["eval"]["backend"] == "judge"
            else self.cfg["eval"]["backend"],
            "frontier_cs_commit": _git_head(PE_ROOT / "third_party" / "Frontier-CS"),
            "llm": {r: c.server_info() for r, c in self.clients.items()},
            "init_solution_hash": h, "init_score": res["score"],
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.log.write_config({**self.cfg, "meta": meta})
        self.state = {"step": 0, "best_code": code, "best_hash": h, "best_score": res["score"],
                      "init_score": res["score"], "b_version": self.belief.version, "history": [],
                      "tokens_in": 0, "tokens_out": 0, "n_improvements": 0, "n_impl_fail": 0,
                      "n_b_updates": 0, "resumes": 0, "wall_s": 0.0}
        self.log.write_json("state.json", self.state)

    # ------------------------------------------------------------ main loop
    def run(self, on_iter=None) -> dict:
        if self.state is None:
            self.setup()
        while self.state["step"] < self.T:
            t0 = time.time()
            rec = self.iterate(self.state["step"] + 1)
            self.state["wall_s"] += time.time() - t0
            self._commit(rec)
            if on_iter is not None:
                on_iter(rec, self.state)
        summary = {k: self.state[k] for k in ("step", "best_score", "init_score", "best_hash", "b_version",
                                              "n_improvements", "n_impl_fail", "n_b_updates",
                                              "tokens_in", "tokens_out", "wall_s", "resumes")}
        summary["condition"] = self.cfg["run"]["condition"]
        summary["seed"] = self.cfg["run"]["seed"]
        summary["calls_by_role"] = self._calls_by_role()
        self.log.write_json("summary.json", summary)
        return summary

    def _commit(self, rec: dict) -> None:
        st = self.state
        self.log.log_iter(rec)
        st["step"] = rec["iter"]
        st["tokens_in"] += rec["tokens_in"]
        st["tokens_out"] += rec["tokens_out"]
        st["n_impl_fail"] += int(rec["impl_status"] != "ok")
        st["n_b_updates"] += int(bool(rec["B_updated"]))
        st["b_version"] = rec["B_version_after"]
        if rec["accepted"]:
            st["best_code"] = self._pending_code
            st["best_hash"] = rec["x_new_hash"]
            st["best_score"] = rec["f_x_new"]
            st["n_improvements"] += 1
        st["history"].append({"iter": rec["iter"], "plan": rec["chosen_plan"], "delta_hat": rec["delta_hat"],
                              "delta": rec["delta"], "status": rec["impl_status"], "accepted": rec["accepted"]})
        self.log.write_json("state.json", st)

    # ------------------------------------------------------------ one iteration
    def iterate(self, step: int) -> dict:
        lp, st = self.lp, self.state
        iter_uid = uuid.uuid4().hex[:12]
        b_before = self.belief.version
        self.rr.begin(step, b_before, iter_uid)
        calls_before = self._token_counter()
        sol = Solution(st["best_code"], st["best_score"])
        shown_b = self.belief if lp["use_belief"] else None
        timing = {}
        K = int(lp["K"])

        # 1. propose
        t = time.time()
        hist = st["history"][-int(lp["history_len"]):] if int(lp["history_len"]) > 0 else []
        plans = roles.propose(self.rr, sol, K, hist, shown_b, show_estimates=lp["use_estimator"])
        timing["propose_s"] = time.time() - t
        if not plans:
            return self._failed_iteration(step, iter_uid, b_before, "no_plans", timing, calls_before)

        # 2. estimate (independently per plan)
        estimates = None
        t = time.time()
        if lp["use_estimator"]:
            if lp.get("estimator_parallel", True) and len(plans) > 1:
                with ThreadPoolExecutor(max_workers=len(plans)) as ex:
                    outs = list(ex.map(lambda ip: roles.estimate(self.rr, sol, ip[1], shown_b, idx=ip[0]),
                                       enumerate(plans)))
            else:
                outs = [roles.estimate(self.rr, sol, p, shown_b, idx=i) for i, p in enumerate(plans)]
            estimates = {p["id"]: o for p, o in zip(plans, outs)}
        timing["estimate_s"] = time.time() - t
        argmax_id = None
        if estimates:
            valid = [(e["delta_hat"], pid) for pid, e in estimates.items() if e is not None]
            if valid:
                argmax_id = max(valid, key=lambda x: x[0])[1]  # ties -> lowest plan id order kept by max

        # 3. select
        t = time.time()
        selection, selector_fallback = None, False
        if len(plans) == 1 or lp["selector"] == "first":
            chosen_id = plans[0]["id"]
        else:
            selection = roles.select(self.rr, sol, plans, estimates, shown_b, remaining=self.T - step)
            if selection is None:
                selector_fallback = True
                chosen_id = argmax_id or plans[0]["id"]
            else:
                chosen_id = selection["chosen_id"]
        timing["select_s"] = time.time() - t
        chosen = next(p for p in plans if p["id"] == chosen_id)
        est = estimates.get(chosen_id) if estimates else None
        delta_hat = est["delta_hat"] if est else None

        # 4-5. implement (+ repairs) and evaluate
        t = time.time()
        impl = self._implement_and_eval(sol, chosen, step)
        timing["implement_eval_s"] = time.time() - t
        impl_status, res, code = impl["status"], impl["res"], impl["code"]
        f_new = res.get("score") if impl_status == "ok" else None
        case_cmp = evidence.compare_cases(self._load_eval(st["best_hash"]), res) if impl_status == "ok" else None
        code_diff = (evidence.code_diff(sol.code, code, int(lp.get("librarian_diff_lines", 150)))
                     if code else None)
        delta = (f_new - sol.score) if f_new is not None else None
        e = (delta - delta_hat) if (delta is not None and delta_hat is not None) else None
        accepted = delta is not None and delta > float(lp["accept_margin"])
        self._pending_code = code if accepted else None

        rec = {
            "iter": step, "iter_uid": iter_uid, "x_t_hash": st["best_hash"], "f_x_t": sol.score,
            "plans": plans, "delta_hats": {pid: (o["delta_hat"] if o else None) for pid, o in estimates.items()}
            if estimates else None,
            "estimates": estimates, "chosen_id": chosen_id, "argmax_id": argmax_id,
            "selector_rationale": selection["rationale"] if selection else None,
            "selector_fallback": selector_fallback, "chosen_plan": chosen["plan"], "delta_hat": delta_hat,
            "estimator_reason": est["reason"] if est else None,
            "relevant_beliefs": est["relevant_beliefs"] if est else None,
            "impl_status": impl_status, "impl_status_first": impl["first_status"],
            "n_compile_fixes": impl["n_compile_fixes"], "n_runtime_fixes": impl["n_runtime_fixes"],
            "repairs": impl["repairs"], "what_changed": impl["what_changed"], "repair_notes": impl["repair_notes"],
            "x_new_hash": code_hash(code) if code else None, "eval": _eval_summary(res) if res else None,
            "eval_trace": impl["trace"], "output_identical": case_cmp["identical"] if case_cmp else None,
            "case_cmp": case_cmp, "code_diff": code_diff,
            "f_x_new": f_new, "delta": delta, "e": e, "accepted": accepted,
            "B_version_before": b_before, "infra": infra_ids(),
        }

        # 6-7. belief update
        t = time.time()
        rec.update(self._maybe_update_belief(sol, rec))
        timing["librarian_s"] = time.time() - t
        rec["timing"] = {k: round(v, 2) for k, v in timing.items()}
        rec.update(self._token_counter(since=calls_before))
        return rec

    def _load_eval(self, h: str) -> dict | None:
        p = self.log.sol_dir / f"{h}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _implement_and_eval(self, sol: Solution, plan: dict, step: int) -> dict:
        """Implement, evaluate, and repair. The original what_changed is kept; each repair's own
        summary goes to repair_notes. repairs[i] records what triggered repair i and the status of
        the evaluation that followed it."""
        lp = self.lp
        out = {"status": "no_code", "first_status": "no_code", "res": {}, "code": None, "what_changed": None,
               "repair_notes": [], "repairs": [], "trace": [], "n_compile_fixes": 0, "n_runtime_fixes": 0}
        impl = roles.implement(self.rr, sol, plan)
        if impl is None:
            return out
        code = impl["code"]
        out["what_changed"] = impl["what_changed"]
        max_c, max_r = int(lp["max_compile_fixes"]), int(lp.get("max_runtime_fixes", 0))
        while True:
            res = self.evaluator.evaluate(code, tag=f"it{step}_eval{len(out['trace'])}")
            h = self.log.save_solution(code, res)
            out["trace"].append({"hash": h, **_eval_summary(res)})
            if out["repairs"]:
                out["repairs"][-1]["status_after"] = res.get("status")
                out["repairs"][-1]["hash_after"] = h
            status = res.get("status", "internal_error")
            if status == "compile_error" and out["n_compile_fixes"] < max_c:
                kind = "compile"
                out["n_compile_fixes"] += 1
            elif status in RUNTIME_REPAIRABLE and out["n_runtime_fixes"] < max_r:
                kind = "runtime"
                out["n_runtime_fixes"] += 1
            else:
                break
            idx = len(out["repairs"]) + 1
            if kind == "compile":
                fixed = roles.implement_fix(self.rr, plan, code, res.get("compile_log", ""), idx=idx)
            else:
                fixed = roles.implement_repair(self.rr, plan, code, evidence.failure_report(res), idx=idx)
            out["repairs"].append({"attempt": idx, "kind": kind, "trigger_status": status, "trigger_hash": h,
                                   "trigger_counts": res.get("status_counts"), "parsed": fixed is not None,
                                   "status_after": None, "hash_after": None})
            if fixed is None:
                break
            out["repair_notes"].append(fixed["what_changed"])
            code = fixed["code"]
        out.update(status=status, first_status=out["trace"][0]["status"], res=res, code=code)
        return out

    def _maybe_update_belief(self, sol: Solution, rec: dict) -> dict:
        lp = self.lp
        rule = lp["update_rule"]
        e = rec["e"]
        fire = e is not None and (rule == "always" or (rule == "error" and abs(e) > float(lp["tau"])))
        out = {"librarian_called": False, "B_updated": False, "B_version_after": self.belief.version,
               "librarian_analysis": None, "belief_ops": None}
        if not fire:
            return out
        lib = roles.librarian(self.rr, sol, self.belief, rec, float(lp["tau"]), rule)
        out["librarian_called"] = True
        if lib is None:
            return out
        ref = f"{self.cfg['run'].get('run_id', 'run')}/iter{rec['iter']:02d}"
        new_b, results = self.belief.apply_ops(lib["ops"], step=rec["iter"], evidence_ref=ref)
        out["librarian_analysis"] = lib["analysis"]
        out["belief_ops"] = [{"op": op, "result": r} for op, r in zip(lib["ops"], results)]
        self.log.log_belief_change({"iter": rec["iter"], "iter_uid": rec["iter_uid"],
                                    "version_before": self.belief.version, "version_after": new_b.version,
                                    "e": e, "analysis": lib["analysis"], "ops": out["belief_ops"]})
        if new_b.version != self.belief.version:
            new_b.save(self.log.b_dir)
            self.belief = new_b
            out["B_updated"] = True
        out["B_version_after"] = self.belief.version
        return out

    def _failed_iteration(self, step, iter_uid, b_before, status, timing, calls_before) -> dict:
        self._pending_code = None
        rec = {"iter": step, "iter_uid": iter_uid, "x_t_hash": self.state["best_hash"],
               "f_x_t": self.state["best_score"], "plans": None, "delta_hats": None, "estimates": None,
               "chosen_id": None, "argmax_id": None, "selector_rationale": None, "selector_fallback": False,
               "chosen_plan": None, "delta_hat": None, "estimator_reason": None, "relevant_beliefs": None,
               "impl_status": status, "impl_status_first": status, "n_compile_fixes": 0, "n_runtime_fixes": 0,
               "repairs": [], "what_changed": None, "repair_notes": [], "x_new_hash": None,
               "eval": None, "eval_trace": [], "output_identical": None, "case_cmp": None, "code_diff": None,
               "f_x_new": None, "delta": None, "e": None, "accepted": False,
               "B_version_before": b_before, "librarian_called": False, "B_updated": False,
               "B_version_after": b_before, "librarian_analysis": None, "belief_ops": None, "infra": infra_ids(),
               "timing": {k: round(v, 2) for k, v in timing.items()}}
        rec.update(self._token_counter(since=calls_before))
        return rec

    def _calls_by_role(self) -> dict:
        """Per role: calls, output tokens, runaways (hit max_tokens), parse failures."""
        out: dict = {}
        p = self.log.dir / "calls.jsonl"
        if not p.exists():
            return out
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                r = out.setdefault(c["role"], {"calls": 0, "tokens_out": 0, "runaway": 0, "parse_fail": 0})
                r["calls"] += 1
                r["tokens_out"] += c.get("tokens_out", 0)
                r["runaway"] += int(c.get("finish_reason") == "length")
                r["parse_fail"] += int(not c.get("parse_ok", True))
        return out

    # token accounting via the call log of the current iteration
    def _token_counter(self, since: dict | None = None) -> dict:
        tin = tout = n = 0
        p = self.log.dir / "calls.jsonl"
        if p.exists() and since is not None:
            with open(p) as f:
                f.seek(since["offset"])
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        tin += c.get("tokens_in", 0)
                        tout += c.get("tokens_out", 0)
                        n += 1
            return {"tokens_in": tin, "tokens_out": tout, "n_calls": n}
        return {"offset": p.stat().st_size if p.exists() else 0}
