"""OpenEvolve-style search with an optional potential-estimation (PE) gate.

Iteration t works on island t % num_islands:
  parent, inspirations = db.sample(island)           (OpenEvolve's storage and sampling, db.py)
  the Generator prompt is built from them; with loop.belief_to_generator it also shows B (when non-empty),
  otherwise it is the same in every condition
  N candidates: independent generations from that prompt, SEARCH/REPLACE diffs applied to the parent
  stage 1: compile + the 5 smallest public cases, for every candidate whose diff applied
  select:  single (vanilla, N = 1) | argmax of the Estimator's delta_hat over stage-1 passers (PE) |
           random among stage-1 passers; uniform among the generated candidates when none passed stage 1
  the selected candidate gets the full 70-case evaluation (the budget unit) if it passed stage 1; a stage-1
  failure is stored with score 0 and its stage-1 artifacts, without a full evaluation
  db.add(child, island); PE: Librarian when |delta - delta_hat| > tau (delta relative to the parent);
  db.tick(island)   (generation counter, migration)
A generation failure (no diff applied) adds nothing. Labeled mode (eval_all): every stage-1 passer also gets a
full evaluation (not budgeted, never inserted) and a second Estimator call with an empty B; both are only logged.

Resume: iters.jsonl, db_state.json, B and state.json are written when an iteration commits. An iteration cut
by the job time limit or by an infrastructure failure (judge or model server) is not committed and is redone.
"""
from __future__ import annotations

import os
import random
import socket
import subprocess
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import evidence, roles
from .beliefs import BeliefDoc
from .config import PE_ROOT, ROLES, resolve_path, sampling_for
from .db import Program, ProgramDB
from .roles import RoleRunner, stable_seed
from .runlog import RunLogger, code_hash


class InfraError(RuntimeError):
    """Evaluation infrastructure failed (not the candidate's fault): abort the iteration."""


def _git_head(path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def infra_ids() -> dict:
    """Which job / vLLM server / judge produced a record (several runs share one server and judge)."""
    return {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "vllm_server": os.environ.get("PE_VLLM_RUN_DIR"), "vllm_variant": os.environ.get("PE_VLLM_VARIANT"),
            "judge_id": os.environ.get("PE_JUDGE_ID")}


def _eval_summary(res: dict | None) -> dict | None:
    if res is None:
        return None
    return {k: res.get(k) for k in ("status", "score", "official_score", "n_cases", "eval_wall_s", "judge", "sid",
                                    "infra_attempts", "status_counts")}


def _pmap(fn, items, workers=None):
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=workers or len(items)) as ex:
        return list(ex.map(fn, items))


class EvolveLoop:
    def __init__(self, cfg: dict, clients: dict, stage1_eval, full_eval, run_dir):
        self.cfg, self.lp = cfg, cfg["loop"]
        self.seed = int(cfg["run"]["seed"])
        self.log = RunLogger(run_dir)
        self.stage1_eval, self.full_eval = stage1_eval, full_eval
        self.clients = clients
        problem = roles.render("problem", score_scale=cfg["problem"]["score_scale"])
        self.rr = RoleRunner(clients, {r: sampling_for(cfg, r) for r in ROLES}, self.log, problem, self.seed,
                             int(self.lp["parse_retries"]), int(self.lp["max_context_tokens"]))
        self.db: ProgramDB | None = None
        self.belief: BeliefDoc | None = None
        self.state: dict | None = None

    # ------------------------------------------------------------------ evaluation
    def _stage1(self, code: str, tag: str) -> dict:
        res = self.stage1_eval.evaluate(code, tag=tag)
        if res.get("status") == "internal_error":
            raise InfraError(f"stage-1 evaluator failed ({tag}): {(res.get('stderr_tail') or '')[-300:]}")
        return res

    def _full(self, code: str, tag: str) -> dict:
        res = self.full_eval.evaluate(code, tag=tag)
        if res.get("status") == "internal_error":
            raise InfraError(f"full evaluation failed ({tag}): {(res.get('stderr_tail') or '')[-300:]}")
        return res

    # ------------------------------------------------------------------ setup / resume
    def setup(self) -> None:
        cap = int(self.lp["belief_cap"])
        st = self.log.read_json("state.json")
        if st is not None:
            self.state = st
            self.db = ProgramDB.from_dict(self.log.read_json("db_state.json"), self.seed)
            self.belief = BeliefDoc.load(self.log.b_dir / f"B_v{st['b_version']}.json", cap)
            self.state["resumes"] = st.get("resumes", 0) + 1
            self.log.truncate_iters(st["t"])  # records of an iteration that did not commit
            return
        code = resolve_path(self.cfg["run"]["init_solution"]).read_text()
        s1 = self._stage1(code, "init_stage1")
        full = self._full(code, "init")
        h = self.log.save_solution(code, full)
        self.log.save_eval(h, s1, ".s1")
        if s1.get("status") != "ok" or full.get("status") != "ok":
            raise RuntimeError(f"initial program failed: stage1={s1.get('status')} full={_eval_summary(full)}")
        self.db = ProgramDB(self.seed, **self.cfg["db"])
        init = Program(id=self.db.new_id(), code=code, score=float(full["score"]),
                       artifacts=evidence.artifacts(s1, full), iteration=-1, code_hash=h, eval_hash=h)
        self.db.add(init, 0)
        init_b = resolve_path(self.lp.get("init_belief"))
        self.belief = BeliefDoc.load(init_b, cap) if init_b else BeliefDoc(cap=cap)
        self.belief = BeliefDoc(self.belief.beliefs, 0, self.belief.next_id, cap)
        self.belief.save(self.log.b_dir)
        meta = {**infra_ids(), "frontier_cs_commit": _git_head(PE_ROOT / "third_party" / "Frontier-CS"),
                "repo_commit": _git_head(PE_ROOT), "llm": {r: c.server_info() for r, c in self.clients.items()},
                "init_hash": h, "init_score": init.score, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.log.write_config({**self.cfg, "meta": meta})
        self.state = {"t": 0, "full_evals": 0, "best_score": init.score, "init_score": init.score,
                      "b_version": 0, "tokens_in": 0, "tokens_out": 0, "n_calls": 0, "resumes": 0, "wall_s": 0.0,
                      "done": False}
        self._save()

    def _save(self) -> None:
        self.log.write_json("db_state.json", self.db.to_dict(with_code=True))
        self.log.write_json("state.json", self.state)

    # ------------------------------------------------------------------ main loop
    def run(self, on_iter=None, deadline: float | None = None) -> dict | None:
        """Run to the budget. Returns None (run not finished) when `deadline` (epoch seconds) passes: no new
        iteration is started after it, so a job's time limit does not cut one in the middle."""
        if self.state is None:
            self.setup()
        budget, max_iters = int(self.cfg["run"]["full_evals"]), int(self.cfg["run"]["max_iters"])
        while self.state["full_evals"] < budget and self.state["t"] < max_iters:
            if deadline is not None and time.time() >= deadline:
                return None
            t0 = time.time()
            rec, child, new_b = self.iterate(self.state["t"])
            self._commit(rec, child, new_b, time.time() - t0)
            if on_iter is not None:
                on_iter(rec, self.state)
        self.state["done"] = True
        self._save()
        summary = self.summary()
        self.log.write_json("summary.json", summary)
        return summary

    def _commit(self, rec: dict, child: Program | None, new_b: BeliefDoc | None, wall: float) -> None:
        st, t = self.state, rec["iteration"]
        if child is not None:
            self.db.add(child, rec["island"])
            rec["migrated"] = self.db.tick(rec["island"])
        if new_b is not None:
            new_b.save(self.log.b_dir)
            self.log.log_belief_change(rec["librarian"]["changelog"])
            self.belief = new_b
        rec["B_version_after"] = self.belief.version
        rec["db"] = self.db.health()
        rec["wall_s"] = round(wall, 2)
        st["t"] = t + 1
        st["full_evals"] += rec["budget_used"]
        st["best_score"] = self.db.best.score
        st["b_version"] = self.belief.version
        st["wall_s"] += wall
        for k in ("tokens_in", "tokens_out", "n_calls"):
            st[k] += rec["tokens"][k]
        rec["full_evals_after"], rec["best_score_after"] = st["full_evals"], st["best_score"]
        self.log.log_iter(rec)
        self._save()
        if st["t"] % 10 == 0:
            self.log.write_json(f"db_snapshots/db_it{st['t']:03d}.json", self.db.to_dict(with_code=False))

    # ------------------------------------------------------------------ one iteration
    def iterate(self, t: int):
        lp, db = self.lp, self.db
        island = t % db.n
        uid = uuid.uuid4().hex[:12]
        self.rr.begin(t, self.belief.version, uid)
        calls_before = self.log.calls_offset()
        timing: dict[str, float] = {}
        tick = time.time()

        parent, insp, mode = db.sample(island)
        pc = self.cfg["prompt"]
        top = db.top(parent.island, int(pc["num_top_programs"]) + int(pc["num_diverse_programs"]))
        gen_b = self.belief if lp.get("belief_to_generator") and len(self.belief) > 0 else None
        system, user = roles.generator_prompt(self.rr, parent, insp, top, random.Random(stable_seed(
            self.seed, t, "prompt")), pc, gen_b)
        N, max_len = int(lp["N"]), int(pc["max_code_length"])
        gens = _pmap(lambda i: roles.generate(self.rr, system, user, parent.code, i, max_len), range(N))
        timing["generate_s"] = time.time() - tick
        cands = []
        for i, g in enumerate(gens):
            c = {"idx": i, **{k: v for k, v in g.items() if k != "code"}, "hash": None, "stage1": None,
                 "passed": False, "est": None, "est_emptyB": None, "full": None, "delta": None, "selected": False}
            if g["code"]:
                c["hash"] = self.log.save_solution(g["code"], None)
                c["_code"] = g["code"]
            cands.append(c)
        applied = [c for c in cands if c["hash"]]

        tick = time.time()
        s1s = _pmap(lambda c: self._stage1(c["_code"], f"it{t}_c{c['idx']}_s1"), applied)
        for c, r in zip(applied, s1s):
            self.log.save_eval(c["hash"], r, ".s1")
            c["_s1"] = r
            c["stage1"] = {"status": r.get("status"), "status_counts": r.get("status_counts"),
                           "max_time_s": max((x.get("time_s") or 0) for x in r.get("per_case") or [{}])}
            c["passed"] = r.get("status") == "ok"
        passed = [c for c in applied if c["passed"]]
        timing["stage1_s"] = time.time() - tick

        # Estimator (with the current B, and in labeled mode also with an empty B) + labeled full evaluations
        tick = time.time()
        jobs = []
        if lp["use_estimator"]:
            empty = BeliefDoc(cap=self.belief.cap)
            for c in passed:
                diff = evidence.code_diff(parent.code, c["_code"], int(lp["estimator_diff_lines"]))
                d = diff["diff"] + ("\n... (diff truncated)" if diff["truncated"] else "")
                s1txt = "\n".join(f"- {k}: {v}" for k, v in evidence.artifacts(c["_s1"]).items())
                jobs.append((c, "est", lambda c=c, d=d, s=s1txt: roles.estimate(self.rr, parent, d, s, self.belief,
                                                                                 c["idx"])))
                if lp["eval_all"]:
                    jobs.append((c, "est_emptyB", lambda c=c, d=d, s=s1txt: roles.estimate(
                        self.rr, parent, d, s, empty, c["idx"], log_role="estimator_emptyB")))
        if lp["eval_all"]:
            for c in passed:
                jobs.append((c, "_full", lambda c=c: self._full(c["_code"], f"it{t}_c{c['idx']}_full")))
        for (c, key, _), out in zip(jobs, _pmap(lambda j: j[2](), jobs)):
            c[key] = out
        for c in passed:
            if c.get("_full") is not None:
                self.log.save_eval(c["hash"], c["_full"])
                sc = float(c["_full"]["score"]) if c["_full"].get("status") == "ok" else 0.0
                c["full"] = {"status": c["_full"].get("status"), "score": sc}
                c["delta"] = sc - parent.score
        timing["estimate_label_s"] = time.time() - tick

        # select
        rng = random.Random(stable_seed(self.seed, t, "select"))
        rule = lp["select"]
        if not applied:
            pick, how = None, "generation_failure"
        elif rule == "single":
            pick, how = applied[0], "single"
        elif not passed:
            pick, how = rng.choice(applied), "random_no_stage1_passer"
        elif rule == "argmax" and any(c["est"] for c in passed):
            pick, how = max((c for c in passed if c["est"]), key=lambda c: c["est"]["delta_hat"]), "argmax_delta_hat"
        else:
            pick, how = rng.choice(passed), ("random" if rule == "random" else "random_no_estimate")

        rec = {"iteration": t, "iter_uid": uid, "island": island, "sampling_mode": mode,
               "parent": {"id": parent.id, "score": parent.score, "island": parent.island, "status": parent.status,
                          "migrant": parent.migrant, "hash": parent.code_hash},
               "inspiration_ids": [p.id for p in insp], "prompt_top_ids": [p.id for p in top],
               "B_version": self.belief.version, "gen_B_version": gen_b.version if gen_b is not None else None,
               "N": N, "selection_rule": how, "selected_idx": None,
               "child": None, "delta_hat": None, "delta": None, "e": None, "budget_used": 0,
               "librarian": {"fired": False}, "migrated": False, "infra": infra_ids()}
        child, new_b = None, None
        if pick is not None:
            pick["selected"] = True
            rec["selected_idx"] = pick["idx"]
            tick = time.time()
            full = None
            if pick["passed"]:
                full = pick.get("_full") or self._full(pick["_code"], f"it{t}_c{pick['idx']}_full")
                self.log.save_eval(pick["hash"], full)
                rec["budget_used"] = 1
                score, status = (float(full["score"]) if full.get("status") == "ok" else 0.0), full.get("status")
            else:
                score, status = 0.0, pick["_s1"].get("status")
            timing["full_eval_s"] = time.time() - tick
            child = Program(id=db.new_id(), code=pick["_code"], score=score,
                            artifacts=evidence.artifacts(pick["_s1"], full), parent_id=parent.id, island=island,
                            iteration=t, generation=parent.generation + 1, status=status, changes=pick["changes"],
                            parent_score=parent.score, code_hash=pick["hash"],
                            eval_hash=pick["hash"] if full is not None else None)
            delta = score - parent.score
            dh = pick["est"]["delta_hat"] if pick["est"] else None
            rec.update(child={"id": child.id, "hash": child.code_hash, "score": score, "status": status,
                              "full_eval": _eval_summary(full)},
                       delta=delta, delta_hat=dh, e=(delta - dh) if dh is not None else None)
            tick = time.time()
            new_b = self._maybe_librarian(rec, parent, pick, full)
            timing["librarian_s"] = time.time() - tick
        rec["candidates"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands]
        rec["timing"] = {k: round(v, 2) for k, v in timing.items()}
        rec["tokens"] = self.log.tokens_since(calls_before)
        return rec, child, new_b

    def _maybe_librarian(self, rec: dict, parent: Program, pick: dict, full: dict | None) -> BeliefDoc | None:
        tau = float(self.lp["tau"])
        if self.lp["update_rule"] != "error" or rec["e"] is None or abs(rec["e"]) <= tau:
            return None
        rec["librarian"]["fired"] = True
        if full is not None and full.get("status") == "ok":
            par = self.log.load_eval(parent.eval_hash) if parent.eval_hash else None
            cases = roles.case_evidence_text(evidence.compare_cases(par, full))
        elif full is not None:
            cases = evidence.failure_report(full)
        else:
            cases = "(no full evaluation)"
        diff = evidence.code_diff(parent.code, pick["_code"], int(self.lp["librarian_diff_lines"]))
        sel = {**pick["est"], "delta": rec["delta"], "score": rec["child"]["score"], "status": rec["child"]["status"],
               "e": rec["e"]}
        lib = roles.librarian(self.rr, parent, self.belief, sel, tau, cases, roles.code_evidence_text(diff))
        if lib is None:
            rec["librarian"]["parse_failed"] = True
            return None
        ref = f"{self.cfg['run'].get('run_id', 'run')}/iter{rec['iteration']:03d}"
        new_b, results = self.belief.apply_ops(lib["ops"], step=rec["iteration"], evidence_ref=ref)
        ops = [{"op": op, "result": r} for op, r in zip(lib["ops"], results)]
        rec["librarian"].update(analysis=lib["analysis"], ops=ops, changelog={
            "iteration": rec["iteration"], "iter_uid": rec["iter_uid"], "version_before": self.belief.version,
            "version_after": new_b.version, "delta_hat": rec["delta_hat"], "delta": rec["delta"], "e": rec["e"],
            "analysis": lib["analysis"], "ops": ops})
        return new_b if new_b.version != self.belief.version else None

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        its = self.log.read_iters()
        modes = Counter(r["sampling_mode"].split("(")[0] for r in its)
        gen, s1, full, sel = Counter(), Counter(), Counter(), Counter()
        for r in its:
            for c in r["candidates"]:
                gen[c["gen_status"]] += 1
                if c["stage1"]:
                    s1[c["stage1"]["status"]] += 1
            ch = r["child"]
            sel["generation_failure" if ch is None else
                "stage1_fail" if ch["full_eval"] is None else f"full_{ch['status']}"] += 1
        ops = Counter(o["result"]["op"] for r in its for o in (r["librarian"].get("ops") or [])
                      if o["result"]["ok"])
        st = self.state
        return {"condition": self.cfg["run"]["condition"], "seed": self.seed, "iterations": len(its),
                "full_evals": st["full_evals"], "init_score": st["init_score"], "best_score": st["best_score"],
                "best_id": self.db.best_id, "best_hash": self.db.best.code_hash,
                "sampling_modes": dict(modes), "sampling_mode_fractions": {k: round(v / max(1, len(its)), 3)
                                                                           for k, v in modes.items()},
                "db": self.db.health(), "candidates_gen_status": dict(gen), "candidates_stage1_status": dict(s1),
                "selected_outcome": dict(sel), "B_version": st["b_version"],
                "librarian_calls": sum(r["librarian"]["fired"] for r in its), "belief_ops": dict(ops),
                "tokens_in": st["tokens_in"], "tokens_out": st["tokens_out"], "n_calls": st["n_calls"],
                "calls_by_role": self.log.calls_by_role(), "wall_s": round(st["wall_s"], 1),
                "resumes": st["resumes"]}
