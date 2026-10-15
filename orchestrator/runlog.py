"""Run directory layout and append-only logs (plan section 5.5).

runs/{run_id}/
  config.yaml      resolved config + environment metadata
  calls.jsonl      one line per LLM call (prompt, raw output incl. reasoning, parsed, tokens, latency)
  iters.jsonl      one line per completed iteration
  solutions/       every evaluated program ({hash}.cpp) and its evaluation ({hash}.json)
  B/               B_v{version}.json for every version + B_changelog.jsonl
  state.json       checkpoint after each completed iteration (for resume)
  summary.json     written at the end
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import yaml


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return repr(o)


class RunLogger:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.sol_dir = self.dir / "solutions"
        self.b_dir = self.dir / "B"
        for d in (self.dir, self.sol_dir, self.b_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, name: str, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False, default=_json_default)
        with self._lock, open(self.dir / name, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log_call(self, rec: dict) -> None:
        rec.setdefault("ts", time.time())
        self._append("calls.jsonl", rec)

    def log_iter(self, rec: dict) -> None:
        rec.setdefault("ts", time.time())
        self._append("iters.jsonl", rec)

    def log_belief_change(self, rec: dict) -> None:
        rec.setdefault("ts", time.time())
        line = json.dumps(rec, ensure_ascii=False, default=_json_default)
        with self._lock, open(self.b_dir / "B_changelog.jsonl", "a") as f:
            f.write(line + "\n")

    def save_solution(self, code: str, result: dict | None) -> str:
        h = code_hash(code)
        (self.sol_dir / f"{h}.cpp").write_text(code)
        if result is not None:
            (self.sol_dir / f"{h}.json").write_text(
                json.dumps(result, indent=1, ensure_ascii=False, default=_json_default))
        return h

    def write_json(self, name: str, obj) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))
        os.replace(tmp, self.dir / name)

    def read_json(self, name: str):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else None

    def write_config(self, cfg: dict) -> None:
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    def read_iters(self) -> list[dict]:
        p = self.dir / "iters.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
