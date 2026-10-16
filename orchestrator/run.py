"""CLI: run one (condition, seed) search.

  python -m orchestrator.run --condition vanilla --seed 0 --run-id NAME [--config overlay.yaml] [--set run.max_iters=20]
  python -m orchestrator.run --condition pe_labeled --mock-llm      # scripted LLM, real (local) evaluation

Re-running with an existing --run-id resumes from runs/{run_id}/state.json.
Exit codes: 0 done, 3 refused to resume (model / condition / seed changed), 4 infrastructure failure (the
current iteration was not committed; re-run to resume), 5 stopped at the deadline (env PE_DEADLINE_TS, epoch
seconds: no iteration is started after it; re-run to resume), 6 the run is owned by another Slurm job that is
still running (runs/{run_id}/OWNER), so this one does nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import ROLES, load_config, resolve_path
from .evaluator import JudgeEvaluator, LocalEvaluator
from .llm import OpenAIChat
from .loop import EvolveLoop, InfraError
from .roles import LLMError


def build_clients(cfg: dict) -> dict:
    llm = cfg["llm"]
    clients = {"default": OpenAIChat(llm["base_url"], llm["model"], llm["request_timeout_s"], llm["http_retries"])}
    for role, over in (llm.get("roles") or {}).items():
        if role not in ROLES:
            raise ValueError(f"unknown role in llm.roles: {role}")
        clients[role] = OpenAIChat(over.get("base_url", llm["base_url"]), over.get("model", llm["model"]),
                                   llm["request_timeout_s"], llm["http_retries"])
    return clients


def build_evaluators(cfg: dict):
    ev = cfg["eval"]
    cases = [int(c) for c in ev["stage1_cases"]]
    stage1 = LocalEvaluator(len(cases), ev["time_limit_s"], ev["mem_mb"], cases=cases)
    if ev["backend"] == "judge":
        return stage1, JudgeEvaluator(ev["judge_url"], ev["judge_type"])
    if ev["backend"] == "local":
        return stage1, LocalEvaluator(ev["workers"], ev["time_limit_s"], ev["mem_mb"])
    raise ValueError(f"unknown eval.backend {ev['backend']!r}")


LIVE_STATES = ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "RESIZING", "STAGE_OUT")


def claim_run(run_dir: Path) -> str | None:
    """Mark this Slurm job as the owner of the run; return the other owner's job id if a different job that is
    still alive owns it (several queued jobs may host the same run, only one may write it)."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    f = run_dir / "OWNER"
    if f.exists():
        other = (f.read_text().split() or [""])[0]
        if other and other != job:
            try:
                state = subprocess.run(["squeue", "-j", other, "-h", "-o", "%T"], capture_output=True, text=True,
                                       timeout=60).stdout.strip()
            except Exception:  # noqa: BLE001 - cannot tell: assume it is alive
                return other
            if state in LIVE_STATES:
                return other
    run_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(f"{job} {socket.gethostname()} {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None, help="extra YAML merged over base.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. run.max_iters=20")
    ap.add_argument("--mock-llm", action="store_true", help="scripted mock LLM (evaluation stays real)")
    args = ap.parse_args(argv)

    overrides = list(args.set) + ([f"run.seed={args.seed}"] if args.seed is not None else [])
    cfg = load_config(args.condition, overrides, args.config)
    if args.mock_llm:
        cfg["llm"]["model"] = "mock-scripted"
    run_id = args.run_id or f"{cfg['run']['condition']}_s{cfg['run']['seed']}_{time.strftime('%m%d_%H%M%S')}"
    cfg["run"]["run_id"] = run_id
    run_dir = resolve_path(cfg["run"]["runs_dir"]) / run_id

    other = claim_run(run_dir)
    if other:
        print(f"[orchestrator] {run_id} is owned by running job {other}; nothing to do here", flush=True)
        return 6
    prev_cfg = run_dir / "config.yaml"
    if prev_cfg.exists():  # resuming: never mix models, conditions or seeds inside one run
        import yaml
        prev = yaml.safe_load(prev_cfg.read_text())
        for a, b in (("llm", "model"), ("run", "condition"), ("run", "seed")):
            if prev[a][b] != cfg[a][b]:
                print(f"[orchestrator] REFUSING to resume {run_id}: {a}.{b} was {prev[a][b]!r}, now {cfg[a][b]!r}",
                      flush=True)
                return 3

    if args.mock_llm:
        from .mock import scripted_llm
        clients = {"default": scripted_llm(seed=cfg["run"]["seed"])}
    else:
        clients = build_clients(cfg)
    stage1, full = build_evaluators(cfg)
    lp = cfg["loop"]
    print(f"[orchestrator] run_id={run_id} dir={run_dir} condition={cfg['run']['condition']} N={lp['N']} "
          f"select={lp['select']} eval_all={lp['eval_all']} budget={cfg['run']['full_evals']} "
          f"max_iters={cfg['run']['max_iters']} backend={cfg['eval']['backend']}", flush=True)
    loop = EvolveLoop(cfg, clients, stage1, full, run_dir)

    def on_iter(rec: dict, st: dict) -> None:
        cands = " ".join(f"{c['idx']}:{c['gen_status'] if not c['hash'] else c['stage1']['status']}"
                         + (f"/dh={c['est']['delta_hat']:+.2f}" if c.get("est") else "")
                         + (f"/d={c['delta']:+.2f}" if c.get("delta") is not None else "") for c in rec["candidates"])
        ch = rec["child"]
        child = "-" if ch is None else f"{ch['id']} {ch['status']} {ch['score']:.2f}"
        print(f"[it {rec['iteration']:03d}] isl={rec['island']} mode={rec['sampling_mode']} "
              f"parent={rec['parent']['id']}({rec['parent']['score']:.2f}) cands=[{cands}] "
              f"pick={rec['selected_idx']}({rec['selection_rule']}) child={child} "
              f"delta={rec['delta']} dhat={rec['delta_hat']} lib={rec['librarian']['fired']} "
              f"B_v{rec['B_version_after']} mig={rec['migrated']} evals={st['full_evals']} best={st['best_score']:.4f} "
              f"tok_out={rec['tokens']['tokens_out']} t={rec['timing']}", flush=True)

    try:
        loop.setup()
        print(f"[orchestrator] start t={loop.state['t']} evals={loop.state['full_evals']} "
              f"best={loop.state['best_score']} B_v{loop.state['b_version']} resumes={loop.state['resumes']}",
              flush=True)
        deadline = float(os.environ["PE_DEADLINE_TS"]) if os.environ.get("PE_DEADLINE_TS") else None
        summary = loop.run(on_iter, deadline)
    except (InfraError, LLMError) as e:
        print(f"[orchestrator] INFRA FAILURE, iteration not committed (re-run to resume): {e}", flush=True)
        return 4
    if summary is None:
        print(f"[orchestrator] deadline reached at t={loop.state['t']} evals={loop.state['full_evals']}; "
              "re-run to resume", flush=True)
        return 5
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
