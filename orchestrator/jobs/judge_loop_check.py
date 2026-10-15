"""Loop integration through the job's shared judge with a scripted LLM (no GPU needed).

  python orchestrator/jobs/judge_loop_check.py --url $PE_JUDGE_URL --judge-type native --runs-dir DIR

Iteration 1: the Implementer first returns a program that does not compile, then the baseline in the repair
call (compile_error -> implementer_fix -> ok). Iteration 2: it returns a program whose output is invalid
(everything at the origin), then the baseline in the runtime repair (invalid -> implementer_repair -> ok);
the repair prompt must carry the checker's messages. Expect: both iterations ok, delta = 0, outputs
identical to the parent, e defined, judge tag on every evaluation, infra ids logged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from orchestrator.config import load_config  # noqa: E402
from orchestrator.evaluator import JudgeEvaluator  # noqa: E402
from orchestrator.llm import MockLLM  # noqa: E402
from orchestrator.loop import ResearchLoop  # noqa: E402
from orchestrator.mock import scripted_llm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--judge-type", required=True)
    ap.add_argument("--runs-dir", required=True)
    args = ap.parse_args()
    base = (ROOT / "solutions/baseline_naive.cpp").read_text()
    invalid = ("#include <bits/stdc++.h>\nint main(){int n;std::cin>>n;for(int i=0;i<n;i++){int k;std::cin>>k;"
               "for(int j=0;j<k;j++){int x,y;std::cin>>x>>y;}}std::printf(\"1 1\\n\");"
               "for(int i=0;i<n;i++)std::printf(\"0 0 0 0\\n\");}\n")
    inner = scripted_llm(0, p_bad_json=0.0)
    n = {"impl": 0}
    seen_repair = []

    def respond(tag, messages):
        if tag == "implementer":
            n["impl"] += 1
            code = base.replace("int main", "int main_broken_", 1) if n["impl"] == 1 else invalid
            return f"<what_changed>original change {n['impl']}</what_changed>\n```cpp\n{code}```"
        if tag == "implementer_fix":
            assert "error:" in messages[-1]["content"], "compile diagnostics not passed to the fix prompt"
            return f"<what_changed>fixed compile</what_changed>\n```cpp\n{base}```"
        if tag == "implementer_repair":
            seen_repair.append(messages[-1]["content"])
            return f"<what_changed>fixed placement</what_changed>\n```cpp\n{base}```"
        return inner.responder(tag, messages)

    cfg = load_config("C3", ["run.budget_T=2", f"run.runs_dir={args.runs_dir}", "eval.backend=judge",
                             f"eval.judge_url={args.url}", f"eval.judge_type={args.judge_type}"])
    cfg["run"]["run_id"] = "judge_loop_check"
    loop = ResearchLoop(cfg, {"default": MockLLM(respond)}, JudgeEvaluator(args.url, args.judge_type),
                        Path(args.runs_dir) / "judge_loop_check")
    summary = loop.run()
    iters = loop.log.read_iters()
    checks = {
        "two_iterations": len(iters) == 2,
        "all_ok": all(r["impl_status"] == "ok" for r in iters),
        "first_needed_one_fix": iters[0]["n_compile_fixes"] == 1
        and iters[0]["eval_trace"][0]["status"] == "compile_error",
        "second_runtime_repair": iters[1]["impl_status_first"] == "invalid" and iters[1]["n_runtime_fixes"] == 1
        and iters[1]["repairs"][0]["status_after"] == "ok",
        "repair_prompt_has_checker_msgs": bool(seen_repair) and any(
            m in seen_repair[0] for m in ("Out of bounds", "Overlap")) and "invalid=" in seen_repair[0],
        "what_changed_preserved": iters[1]["what_changed"] == "original change 2"
        and iters[1]["repair_notes"] == ["fixed placement"],
        "output_identical_flag": all(r["output_identical"] is True for r in iters),
        "delta_zero": all(abs(r["delta"]) < 1e-9 for r in iters),
        "e_defined": all(r["e"] is not None for r in iters),
        "judge_tagged": all(t.get("judge") == args.judge_type for r in iters for t in r["eval_trace"]),
        "infra_logged": all(r["infra"].get("judge_id") for r in iters),
    }
    print(json.dumps({"summary": summary, "checks": checks,
                      "repair_prompt": seen_repair[0][-1500:] if seen_repair else None,
                      "iters": [{k: r[k] for k in ("iter", "impl_status", "impl_status_first", "n_compile_fixes",
                                                   "n_runtime_fixes", "repairs", "eval_trace", "delta", "e",
                                                   "output_identical", "case_cmp", "infra")} for r in iters]}, indent=1, default=str))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
