"""Score noise under the concurrency the main experiment will actually use.

  python orchestrator/jobs/noise_concurrent.py --url $PE_JUDGE_URL --judge-type native \
      --concurrency 5 --reps 4 --out DIR/noise.json [--tau-floor 1.0]

Production layout: one job = one vLLM server + one shared judge + K concurrent runs. Each run submits
one candidate at a time, so the worst case is K simultaneous submissions. Here K threads submit the
same solution simultaneously (barrier per round), R rounds per solution, while the caller keeps vLLM
busy with background generation. An isolated pass (1 submitter, R rounds) is measured first as control.
sigma = std of the scores of the VALID concurrent evaluations (e is only computed for valid candidates);
tau = max(tau_floor, 3 * max sigma) over the solutions that obey the Implementer's rules (fixed seed or
<= 1.5 s wall clock). How often a valid program flips to a failure (e.g. one TLE case) is reported
separately as fail_rate, because that shows up as implementation failures, not as score noise.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from orchestrator.evaluator import JudgeEvaluator  # noqa: E402

PROBLEM = ROOT / "third_party/Frontier-CS/algorithmic/problems/0"
SOLUTIONS = {  # name -> (path, obeys implementer rules?)
    "baseline": (ROOT / "solutions/baseline_naive.cpp", True),             # deterministic, no clock
    "reference_tl1500": (ROOT / "solutions/noise/reference_tl1500.cpp", True),  # 1.5 s wall clock, clock seed
    "reference": (PROBLEM / "examples/reference.cpp", False),              # 1.98 s wall clock (> rule)
}


def run_round(ev: JudgeEvaluator, code: str, k: int, tag: str) -> list[dict]:
    barrier = threading.Barrier(k)
    out: list[dict] = [None] * k  # type: ignore[list-item]

    def worker(i):
        barrier.wait()
        out[i] = ev.evaluate(code, tag=f"{tag}_{i}")
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out


def stats(results: list[dict]) -> dict:
    scores = [r["official_score"] for r in results if r.get("official_score") is not None]
    ok = [r["score"] for r in results if r["status"] == "ok"]
    return {"n": len(results), "n_ok": len(ok), "statuses": [r["status"] for r in results],
            "mean": st.mean(scores) if scores else None, "std": st.stdev(scores) if len(scores) > 1 else 0.0,
            "mean_ok": st.mean(ok) if ok else None, "std_ok": st.stdev(ok) if len(ok) > 1 else 0.0,
            "min": min(scores) if scores else None, "max": max(scores) if scores else None,
            "max_case_cpu_s": max((c["time_s"] for r in results for c in r.get("per_case", [])), default=None),
            "wall_s_mean": st.mean(r.get("eval_wall_s") or 0 for r in results),
            "fail_reasons": [f"{r['status']}: {(r.get('stderr_tail') or r.get('compile_log') or '')[:200]}"
                             for r in results if r["status"] != "ok"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--judge-type", required=True)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--tau-floor", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ev = JudgeEvaluator(args.url, args.judge_type)
    report = {"judge_type": args.judge_type, "concurrency": args.concurrency, "reps": args.reps, "solutions": {}}
    for name, (path, obeys) in SOLUTIONS.items():
        code = path.read_text()
        iso = [ev.evaluate(code, tag=f"{name}_iso{r}") for r in range(args.reps)]
        conc = []
        t0 = time.time()
        for r in range(args.reps):
            conc += run_round(ev, code, args.concurrency, f"{name}_c{r}")
        report["solutions"][name] = {"obeys_implementer_rules": obeys, "isolated": stats(iso),
                                     "concurrent": stats(conc), "concurrent_wall_s": round(time.time() - t0, 1)}
        s = report["solutions"][name]
        print(f"[noise] {name}: iso mean={s['isolated']['mean']} std={s['isolated']['std']:.4f} | "
              f"conc(K={args.concurrency}) mean_ok={s['concurrent']['mean_ok']} std_ok={s['concurrent']['std_ok']:.4f} "
              f"ok={s['concurrent']['n_ok']}/{s['concurrent']['n']}", flush=True)
    for s in report["solutions"].values():
        c = s["concurrent"]
        c["fail_rate"] = 1 - c["n_ok"] / c["n"]
    sig = max(s["concurrent"]["std_ok"] for s in report["solutions"].values() if s["obeys_implementer_rules"])
    report["sigma_for_tau"] = sig
    report["tau"] = max(args.tau_floor, 3 * sig)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[noise] sigma={sig:.4f} -> tau={report['tau']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
