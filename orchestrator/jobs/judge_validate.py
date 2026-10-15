"""Validate the job's shared judge on the example solutions and write per-case tables.

  python orchestrator/jobs/judge_validate.py --url $PE_JUDGE_URL --judge-type $PE_JUDGE_TYPE --out DIR

Submits (after one warm-up) examples/reference.cpp, examples/gpt5.cpp, solutions/baseline_naive.cpp
(+ solutions/noise/gpt5_pre120.cpp as evidence) to the judge and the same files to local_eval on this
node. Writes DIR/{name}.judge.json (raw server result), DIR/{name}.local.json, DIR/per_case.md and
DIR/validation.json. Exit 0 iff: reference ok and within 0.3 of 89.08, baseline ok and == 45.99985793,
and gpt5 fails every case with a checker verdict (its 0 is then the example's bug, see eval/GPT5.md).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from orchestrator.evaluator import JudgeEvaluator, LocalEvaluator  # noqa: E402

PROBLEM = ROOT / "third_party/Frontier-CS/algorithmic/problems/0"
FILES = {
    "reference": PROBLEM / "examples/reference.cpp",
    "gpt5": PROBLEM / "examples/gpt5.cpp",
    "baseline": ROOT / "solutions/baseline_naive.cpp",
    "gpt5_pre120": ROOT / "solutions/noise/gpt5_pre120.cpp",
}
COMPILE_FLAGS = "/usr/bin/g++ main.cpp -O2 -pipe -std=gnu++17 (algorithmic/judge/src/gojudge.js:93)"


def cpu_model() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        return next(l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name"))
    except Exception:  # noqa: BLE001
        return platform.processor()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--judge-type", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-workers", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    judge = JudgeEvaluator(args.url, args.judge_type)
    local = LocalEvaluator(args.local_workers)
    judge.evaluate(FILES["baseline"].read_text(), tag="warmup")  # first submission after start can flake

    res = {}
    for name, path in FILES.items():
        code = path.read_text()
        j = judge.evaluate(code, tag=name)
        loc = local.evaluate(code, tag=name)
        (out / f"{name}.judge.json").write_text(json.dumps(j, indent=1))
        (out / f"{name}.local.json").write_text(json.dumps(loc, indent=1))
        res[name] = (j, loc)
        print(f"[validate] {name:12s} judge={j['status']}/{j.get('official_score')} "
              f"local={loc['status']}/{loc.get('official_score')} counts={j.get('status_counts')}", flush=True)

    host = platform.node()
    md = [f"# Per-case results ({args.judge_type} judge vs local_eval)", "",
          f"- node: {host}, CPU: {cpu_model()}", f"- judge compile: {COMPILE_FLAGS}",
          "- local_eval compile: g++ -O2 -pipe -std=gnu++17 (same flags), limits 2 s CPU / 256 MB", "",
          "| solution | judge status | judge score (failed cases = 0) | local status | local score |",
          "|---|---|---|---|---|"]
    for name, (j, loc) in res.items():
        md.append(f"| {name} | {j['status']} {j.get('status_counts')} | {j.get('official_score')} | "
                  f"{loc['status']} {loc.get('status_counts')} | {loc.get('official_score')} |")
    for name, (j, loc) in res.items():
        md += ["", f"## {name}", "", "| case | judge status | judge ratio | judge cpu s | judge MB | local status | "
               "local ratio | local cpu s | checker / runtime message |", "|---|---|---|---|---|---|---|---|---|"]
        lc = {c["case"]: c for c in loc.get("per_case", [])}
        for c in j.get("per_case", []):
            l = lc.get(c["case"], {})
            msg = (c.get("msg") or l.get("msg") or "").replace("|", "/").replace("\n", " ")[:90]
            md.append(f"| {c['case']} | {c['status']} | {c.get('ratio')} | {c['time_s']:.3f} | {c['mem_mb']:.1f} | "
                      f"{l.get('status')} | {l.get('ratio')} | {l.get('time_s')} | {msg} |")
        if j["status"] == "compile_error":
            md.append(f"\ncompile log: {j.get('compile_log', '')[:500]}")
    (out / "per_case.md").write_text("\n".join(md) + "\n")

    ref, gpt5, base = res["reference"][0], res["gpt5"][0], res["baseline"][0]
    checks = {
        "reference_ok": ref["status"] == "ok" and abs((ref.get("score") or 0) - 89.08) < 0.3,
        "baseline_ok": base["status"] == "ok" and abs((base.get("score") or 0) - 45.99985793142856) < 1e-6,
        "gpt5_all_cases_checker_rejected": gpt5.get("status_counts") == {"invalid": 70},
        "local_matches_judge_reference": abs((res["reference"][1].get("score") or 0) - (ref.get("score") or 0)) < 0.1,
    }
    summary = {"judge_type": args.judge_type, "judge_url": args.url, "host": host, "cpu_model": cpu_model(),
               "compile_flags": COMPILE_FLAGS, "checks": checks, "passed": all(checks.values()),
               "scores": {n: {"judge_status": j["status"], "judge_score": j.get("official_score"),
                              "local_status": l["status"], "local_score": l.get("official_score"),
                              "judge_counts": j.get("status_counts")} for n, (j, l) in res.items()}}
    (out / "validation.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
