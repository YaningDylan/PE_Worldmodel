"""Probe set (plan section 6): a fixed set of (x, a, delta) to measure how well an Estimator
reading a given belief doc B predicts score changes of modifications it has never seen.

  build:  python -m orchestrator.probe build --snapshot base=solutions/baseline_naive.cpp ... \
              --out probe/probe_set.jsonl --plans 20 --keep 15 [--set llm.model=Qwen/Qwen3.6-27B ...]
  eval:   python -m orchestrator.probe eval --probe probe/probe_set.jsonl \
              (--belief B.json | --belief empty | --run-dir runs/X [--every 5]) --out probe/results/NAME.json

Probe content never enters any run's B or Librarian input: build/eval only write under probe/.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import roles
from .beliefs import BeliefDoc
from .config import ROLES, load_config, resolve_path, sampling_for
from .roles import RoleRunner, Solution
from .run import build_clients, build_evaluator
from .runlog import RunLogger, code_hash


# ------------------------------------------------------------------ metrics
def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return None if va == 0 or vb == 0 else cov / (va * vb)


def sign(x: float, eps: float) -> int:
    return 0 if abs(x) <= eps else (1 if x > 0 else -1)


def metrics(rows: list[dict], key: str = "delta_hat", eps: float = 0.0) -> dict:
    """rows: {snapshot, delta, <key>}; rows with key None are counted as missing."""
    ok = [r for r in rows if r.get(key) is not None]
    if not ok:
        return {"n": 0, "n_missing": len(rows)}
    err = [r[key] - r["delta"] for r in ok]
    by_snap: dict[str, list[dict]] = {}
    for r in ok:
        by_snap.setdefault(r["snapshot"], []).append(r)
    rhos = {s: spearman([r[key] for r in rs], [r["delta"] for r in rs]) for s, rs in by_snap.items()}
    valid_rhos = [v for v in rhos.values() if v is not None]
    return {
        "n": len(ok), "n_missing": len(rows) - len(ok),
        "mae": sum(abs(e) for e in err) / len(err),
        "rmse": math.sqrt(sum(e * e for e in err) / len(err)),
        "bias": sum(err) / len(err),  # mean(delta_hat - delta); > 0 = over-optimistic
        "direction_acc": sum(sign(r[key], eps) == sign(r["delta"], eps) for r in ok) / len(ok),
        "spearman_within_snapshot_mean": sum(valid_rhos) / len(valid_rhos) if valid_rhos else None,
        "spearman_by_snapshot": rhos,
    }


# ------------------------------------------------------------------ build
def _words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def dedupe(plans: list[str], thresh: float = 0.6) -> list[str]:
    kept: list[str] = []
    for p in plans:
        w = _words(p)
        if all(len(w & _words(q)) / max(1, len(w | _words(q))) < thresh for q in kept):
            kept.append(p)
    return kept


def build(args, cfg) -> None:
    out = resolve_path(args.out)
    work = out.parent / f"build_{out.stem}"
    log = RunLogger(work)
    clients = build_clients(cfg)
    sampling = {r: sampling_for(cfg, r) for r in ROLES}
    problem = roles.render("problem", score_scale=cfg["problem"]["score_scale"])
    evaluator = build_evaluator(cfg)
    eval_lock = threading.Lock()  # one evaluation at a time: keeps timing noise comparable to runs
    K = int(cfg["loop"]["K"])
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["snapshot"], r["plan_idx"]))

    for snap in args.snapshot:
        name, path = snap.split("=", 1)
        code = resolve_path(path).read_text()
        with eval_lock:
            base = evaluator.evaluate(code, tag=f"probe_{name}_base")
        if base.get("status") != "ok":
            raise RuntimeError(f"snapshot {name} does not evaluate: {base.get('status')}")
        sol = Solution(code, base["score"])
        log.save_solution(code, base)
        plan_file = work / f"plans_{name}.json"
        if plan_file.exists():
            plans = json.loads(plan_file.read_text())
        else:
            raw: list[str] = []
            for call in range(math.ceil(args.plans / K)):
                rr = RoleRunner(clients, sampling, log, problem, run_seed=args.seed)
                rr.begin(step=call, b_version=-1, iter_uid=f"probe_{name}_propose{call}")
                got = roles.propose(rr, sol, K, [], None, show_estimates=False) or []
                raw += [p["plan"] for p in got]
            plans = dedupe(raw)[: args.keep]
            plan_file.write_text(json.dumps(plans, indent=1))
        print(f"[probe] {name}: f={sol.score:.4f} plans={len(plans)}", flush=True)

        def one(i_plan):
            i, plan = i_plan
            if (name, i) in done:
                return None
            rr = RoleRunner(clients, sampling, log, problem, run_seed=args.seed)
            rr.begin(step=1000 + i, b_version=-1, iter_uid=f"probe_{name}_{i}")
            p = {"id": "p1", "plan": plan}
            impl = roles.implement(rr, sol, p)
            status, res, new_code, fixes = "no_code", {}, None, 0
            if impl is not None:
                new_code = impl["code"]
                while True:
                    with eval_lock:
                        res = evaluator.evaluate(new_code, tag=f"probe_{name}_{i}")
                    log.save_solution(new_code, res)
                    status = res.get("status", "internal_error")
                    if status != "compile_error" or fixes >= int(cfg["loop"]["max_compile_fixes"]):
                        break
                    fixes += 1
                    fx = roles.implement_fix(rr, p, new_code, res.get("compile_log", ""), idx=fixes)
                    if fx is None:
                        break
                    new_code = fx["code"]
            f_new = res.get("score") if status == "ok" else None
            return {"snapshot": name, "snapshot_hash": code_hash(code), "snapshot_path": path, "plan_idx": i,
                    "f_x": sol.score, "plan": plan, "impl_status": status, "n_compile_fixes": fixes,
                    "what_changed": impl.get("what_changed") if impl else None,
                    "x_new_hash": code_hash(new_code) if new_code else None, "f_x_new": f_new,
                    "delta": (f_new - sol.score) if f_new is not None else None,
                    "implementer_model": cfg["llm"].get("roles", {}).get("implementer", {}).get(
                        "model", cfg["llm"]["model"])}

        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            for rec in ex.map(one, enumerate(plans)):
                if rec is not None:
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"[probe] {name}#{rec['plan_idx']} {rec['impl_status']} delta={rec['delta']}", flush=True)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l]
    fail = sum(r["impl_status"] != "ok" for r in rows)
    print(f"[probe] total={len(rows)} impl_fail={fail} ({fail / max(1, len(rows)):.0%})", flush=True)


# ------------------------------------------------------------------ eval
def evaluate_belief(entries, belief: BeliefDoc | None, clients, sampling, problem, log, seed, samples, parallel):
    def one(i_e):
        i, e = i_e
        sol = Solution(Path(resolve_path(e["snapshot_path"])).read_text(), e["f_x"])
        preds = []
        for s in range(samples):
            rr = RoleRunner(clients, sampling, log, problem, run_seed=seed)
            rr.begin(step=i, b_version=belief.version if belief else -1, iter_uid=f"probe_eval_{i}_{s}")
            est = roles.estimate(rr, sol, {"id": "p1", "plan": e["plan"]}, belief, idx=s)
            if est is not None:
                preds.append(est["delta_hat"])
        return {"snapshot": e["snapshot"], "plan_idx": e["plan_idx"], "delta": e["delta"],
                "delta_hat": sum(preds) / len(preds) if preds else None, "samples": preds}
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        return list(ex.map(one, enumerate(entries)))


def eval_cmd(args, cfg) -> None:
    entries = [json.loads(l) for l in resolve_path(args.probe).read_text().splitlines() if l]
    entries = [e for e in entries if e["impl_status"] == "ok"]
    if args.check_snapshots:
        for e in entries:
            assert code_hash(resolve_path(e["snapshot_path"]).read_text()) == e["snapshot_hash"], e["snapshot_path"]
    clients = build_clients(cfg)
    sampling = {r: sampling_for(cfg, r) for r in ROLES}
    problem = roles.render("problem", score_scale=cfg["problem"]["score_scale"])
    cap = int(cfg["loop"]["belief_cap"])
    targets: list[tuple[str, BeliefDoc | None]] = []
    if args.run_dir:
        bdir = resolve_path(args.run_dir) / "B"
        versions = sorted(int(m.group(1)) for p in bdir.glob("B_v*.json")
                          if (m := re.match(r"B_v(\d+)\.json", p.name)))
        keep = [v for v in versions if v % args.every == 0 or v == versions[-1]]
        targets = [(f"v{v}", BeliefDoc.load(bdir / f"B_v{v}.json", cap)) for v in keep]
    elif args.belief == "empty":
        targets = [("empty", BeliefDoc(cap=cap))]
    elif args.belief == "none":
        targets = [("none", None)]
    else:
        targets = [(Path(args.belief).stem, BeliefDoc.load(resolve_path(args.belief), cap))]
    out = resolve_path(args.out)
    log = RunLogger(out.parent / f"calls_{out.stem}")
    results = {"probe": args.probe, "model": cfg["llm"]["model"], "n_entries": len(entries), "targets": {},
               "baselines": {}}
    zero = [{"snapshot": e["snapshot"], "delta": e["delta"], "pred": 0.0} for e in entries]
    results["baselines"]["zero"] = metrics(zero, "pred")
    if args.hist_mean is not None:
        hm = [{"snapshot": e["snapshot"], "delta": e["delta"], "pred": args.hist_mean} for e in entries]
        results["baselines"]["hist_mean"] = metrics(hm, "pred") | {"value": args.hist_mean}
    for name, b in targets:
        t0 = time.time()
        rows = evaluate_belief(entries, b, clients, sampling, problem, log, args.seed, args.samples, args.parallel)
        results["targets"][name] = {"metrics": metrics(rows, eps=args.eps), "rows": rows,
                                    "B_version": b.version if b else None, "n_beliefs": len(b) if b else 0,
                                    "wall_s": round(time.time() - t0, 1)}
        m = results["targets"][name]["metrics"]
        print(f"[probe-eval] {name}: n={m.get('n')} mae={m.get('mae')} bias={m.get('bias')} "
              f"dir={m.get('direction_acc')} rho={m.get('spearman_within_snapshot_mean')}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--snapshot", action="append", required=True, help="name=path/to/solution.cpp")
    b.add_argument("--out", default="probe/probe_set.jsonl")
    b.add_argument("--plans", type=int, default=20)
    b.add_argument("--keep", type=int, default=15)
    b.add_argument("--parallel", type=int, default=8)
    e = sub.add_parser("eval")
    e.add_argument("--probe", default="probe/probe_set.jsonl")
    e.add_argument("--belief", default="empty", help="path to B json | empty | none")
    e.add_argument("--run-dir", default=None, help="evaluate every --every-th B version of this run")
    e.add_argument("--every", type=int, default=5)
    e.add_argument("--samples", type=int, default=1)
    e.add_argument("--parallel", type=int, default=16)
    e.add_argument("--eps", type=float, default=0.0, help="|x|<=eps counts as sign 0")
    e.add_argument("--hist-mean", type=float, default=None, help="baseline: mean delta outside the probe set")
    e.add_argument("--check-snapshots", action="store_true")
    e.add_argument("--out", required=True)
    for p in (b, e):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--set", action="append", default=[])
    args = ap.parse_args(argv)
    # probe generation: proposer without B at temperature 1.0 (base.yaml default for proposer)
    cfg = load_config("C2", args.set)
    if args.cmd == "build":
        build(args, cfg)
    else:
        eval_cmd(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
