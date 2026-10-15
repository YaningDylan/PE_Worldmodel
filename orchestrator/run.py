"""CLI: run one (condition, seed) research loop.

  python -m orchestrator.run --condition C3 --seed 0 [--run-id NAME] [--set loop.tau=1.5 ...]
  python -m orchestrator.run --condition C3 --mock      # dry run with scripted LLM + mock evaluator

Re-running with an existing --run-id resumes from runs/{run_id}/state.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import PE_ROOT, ROLES, load_config, resolve_path
from .evaluator import JudgeEvaluator, LocalEvaluator, MockEvaluator
from .llm import OpenAIChat
from .loop import ResearchLoop


def build_clients(cfg: dict) -> dict:
    llm = cfg["llm"]
    default = OpenAIChat(llm["base_url"], llm["model"], llm["request_timeout_s"], llm["http_retries"])
    clients = {"default": default}
    for role, over in (llm.get("roles") or {}).items():
        if role not in ROLES:
            raise ValueError(f"unknown role in llm.roles: {role}")
        clients[role] = OpenAIChat(over.get("base_url", llm["base_url"]), over.get("model", llm["model"]),
                                   llm["request_timeout_s"], llm["http_retries"])
    return clients


def build_evaluator(cfg: dict):
    ev = cfg["eval"]
    if ev["backend"] == "judge":
        return JudgeEvaluator(ev["judge_url"], ev["judge_type"])
    if ev["backend"] == "local":
        return LocalEvaluator(ev["workers"], ev["time_limit_s"], ev["mem_mb"])
    raise ValueError(f"unknown eval.backend {ev['backend']!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None, help="extra YAML merged over base.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. loop.tau=1.5")
    ap.add_argument("--mock", action="store_true", help="scripted mock LLM + mock evaluator")
    args = ap.parse_args(argv)

    overrides = list(args.set)
    if args.seed is not None:
        overrides.append(f"run.seed={args.seed}")
    cfg = load_config(args.condition, overrides, args.config)
    run_id = args.run_id or f"{cfg['run']['condition']}_s{cfg['run']['seed']}_{time.strftime('%m%d_%H%M%S')}"
    cfg["run"]["run_id"] = run_id
    run_dir = resolve_path(cfg["run"]["runs_dir"]) / run_id

    prev_cfg = run_dir / "config.yaml"
    if prev_cfg.exists():  # resuming: never mix models, conditions or seeds inside one run
        import yaml
        prev = yaml.safe_load(prev_cfg.read_text())
        for key in (("llm", "model"), ("run", "condition"), ("run", "seed")):
            old_v, new_v = prev[key[0]][key[1]], cfg[key[0]][key[1]]
            if old_v != new_v:
                print(f"[orchestrator] REFUSING to resume {run_id}: {'.'.join(key)} was {old_v!r}, now {new_v!r}",
                      flush=True)
                return 3

    if args.mock:
        from .mock import scripted_llm
        clients = {"default": scripted_llm(seed=cfg["run"]["seed"])}
        evaluator = MockEvaluator()
    else:
        clients = build_clients(cfg)
        evaluator = build_evaluator(cfg)

    print(f"[orchestrator] run_id={run_id} dir={run_dir} condition={cfg['run']['condition']} "
          f"T={cfg['run']['budget_T']} tau={cfg['loop']['tau']}", flush=True)
    loop = ResearchLoop(cfg, clients, evaluator, run_dir)
    loop.setup()
    st = loop.state
    print(f"[orchestrator] start step={st['step']} best={st['best_score']} B_v{st['b_version']}", flush=True)

    def on_iter(rec: dict, state: dict) -> None:
        print(f"[iter {rec['iter']:02d}] chosen={rec['chosen_id']} argmax={rec['argmax_id']} "
              f"dhat={rec['delta_hat']} status={rec['impl_status']} delta={rec['delta']} e={rec['e']} "
              f"acc={rec['accepted']} B_v{rec['B_version_after']} best={state['best_score']} "
              f"tok_out={rec['tokens_out']} t={rec['timing']}", flush=True)

    summary = loop.run(on_iter)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
