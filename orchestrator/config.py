"""Config loading: base.yaml <- conditions.yaml[condition] <- --set overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

from .llm import Sampling

PKG_DIR = Path(__file__).resolve().parent
PE_ROOT = Path(os.environ.get("PE_ROOT", PKG_DIR.parent)).resolve()
CONFIG_DIR = PKG_DIR / "configs"

ROLES = ("generator", "estimator", "librarian")
SELECT_RULES = ("single", "argmax", "random")
UPDATE_RULES = ("never", "error")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_override(cfg: dict, assignment: str) -> None:
    """Apply 'a.b.c=value' (value parsed as YAML) in place."""
    if "=" not in assignment:
        raise ValueError(f"override must look like key.path=value: {assignment!r}")
    key, raw = assignment.split("=", 1)
    parts = key.strip().split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(condition: str | None = None, overrides: list[str] | None = None,
                extra_file: str | None = None) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    if extra_file:
        cfg = deep_merge(cfg, yaml.safe_load(Path(extra_file).read_text()) or {})
    cond = condition or cfg["run"]["condition"]
    conds = yaml.safe_load((CONFIG_DIR / "conditions.yaml").read_text())
    if cond not in conds:
        raise ValueError(f"unknown condition {cond!r}; known: {sorted(conds)}")
    cfg = deep_merge(cfg, conds[cond])
    cfg["run"]["condition"] = cond
    for o in overrides or []:
        apply_override(cfg, o)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    lp = cfg["loop"]
    if lp["select"] not in SELECT_RULES:
        raise ValueError(f"loop.select must be one of {SELECT_RULES}")
    if lp["update_rule"] not in UPDATE_RULES:
        raise ValueError(f"loop.update_rule must be one of {UPDATE_RULES}")
    if int(lp["N"]) < 1 or (lp["select"] == "single") != (int(lp["N"]) == 1):
        raise ValueError("loop.N must be >= 1, and select=single iff N=1")
    if (lp["select"] == "argmax" or lp["update_rule"] != "never" or lp["eval_all"]) and not lp["use_estimator"]:
        raise ValueError("select=argmax, update_rule=error and eval_all need use_estimator")
    if int(cfg["run"]["full_evals"]) < 1 or int(cfg["run"]["max_iters"]) < 1:
        raise ValueError("run.full_evals and run.max_iters must be >= 1")
    if float(lp["tau"]) < 0:
        raise ValueError("tau must be >= 0")


def sampling_for(cfg: dict, role: str) -> Sampling:
    s = deep_merge(cfg["sampling"]["default"], cfg["sampling"].get(role, {}) or {})
    return Sampling(**s)


def resolve_path(p: str | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else PE_ROOT / path
