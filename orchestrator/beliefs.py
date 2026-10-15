"""Belief doc B: a structured, versioned list of falsifiable beliefs.

The Librarian never rewrites B directly; it emits a list of operations
(ADD / UPDATE / MERGE / DELETE) that this module validates and applies.
Every successful update bumps the version by one.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

EDITABLE_FIELDS = ("statement", "conditions", "predicted_effect", "confidence")
OPS = ("ADD", "UPDATE", "MERGE", "DELETE")
EVIDENCE_KINDS = ("for", "against")


def _clamp_conf(x: Any) -> float:
    v = float(x)
    if v != v:  # NaN
        raise ValueError("confidence is NaN")
    return min(1.0, max(0.0, v))


def _req_str(op: dict, key: str) -> str:
    v = op.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"missing or empty '{key}'")
    return v.strip()


def _opt_str(op: dict, key: str, default: str = "") -> str:
    v = op.get(key, default)
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return str(v)
    if not isinstance(v, str):
        raise ValueError(f"'{key}' must be a string")
    return v.strip()


class BeliefDoc:
    def __init__(self, beliefs: list[dict] | None = None, version: int = 0,
                 next_id: int | None = None, cap: int = 30):
        self.beliefs: list[dict] = [dict(b) for b in (beliefs or [])]
        self.version = int(version)
        self.cap = int(cap)
        if next_id is None:
            nums = [int(b["id"][1:]) for b in self.beliefs
                    if isinstance(b.get("id"), str) and b["id"][1:].isdigit()]
            next_id = max(nums, default=0) + 1
        self.next_id = int(next_id)

    # ---------- io ----------
    @classmethod
    def from_dict(cls, d: dict, cap: int = 30) -> "BeliefDoc":
        return cls(d.get("beliefs", []), d.get("version", 0), d.get("next_id"), cap)

    @classmethod
    def load(cls, path: str | Path, cap: int = 30) -> "BeliefDoc":
        return cls.from_dict(json.loads(Path(path).read_text()), cap)

    def to_dict(self) -> dict:
        return {"version": self.version, "next_id": self.next_id,
                "beliefs": copy.deepcopy(self.beliefs)}

    def save(self, belief_dir: str | Path) -> Path:
        p = Path(belief_dir) / f"B_v{self.version}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return p

    # ---------- queries ----------
    def __len__(self) -> int:
        return len(self.beliefs)

    def get(self, bid: str) -> dict | None:
        for b in self.beliefs:
            if b["id"] == bid:
                return b
        return None

    def subset(self, ids: list[str]) -> list[dict]:
        return [b for b in (self.get(i) for i in ids) if b is not None]

    def render(self) -> str:
        """Compact text form used inside prompts (one JSON object per belief)."""
        if not self.beliefs:
            return f"(version {self.version}; empty - no beliefs recorded yet)"
        lines = [f"(version {self.version}; {len(self.beliefs)}/{self.cap} beliefs)"]
        for b in self.beliefs:
            lines.append(json.dumps({
                "id": b["id"],
                "statement": b["statement"],
                "conditions": b.get("conditions", ""),
                "predicted_effect": b.get("predicted_effect", ""),
                "confidence": round(float(b.get("confidence", 0.5)), 2),
                "n_evidence_for": len(b.get("evidence_for", [])),
                "n_evidence_against": len(b.get("evidence_against", [])),
            }, ensure_ascii=False))
        return "\n".join(lines)

    # ---------- updates ----------
    def _new_id(self) -> str:
        bid = f"b{self.next_id:03d}"
        self.next_id += 1
        return bid

    def apply_ops(self, ops: list[dict], *, step: int, evidence_ref: str
                  ) -> tuple["BeliefDoc", list[dict]]:
        """Apply ops in order on a copy. Invalid ops are skipped and reported.

        Returns (new_doc, results). new_doc.version = version + 1 iff at least
        one op was applied; otherwise new_doc is an unchanged copy.
        """
        new = BeliefDoc(copy.deepcopy(self.beliefs), self.version, self.next_id, self.cap)
        results = []
        for i, op in enumerate(ops if isinstance(ops, list) else []):
            res = {"index": i, "op": op.get("op") if isinstance(op, dict) else None,
                   "ok": False, "error": None, "id": None, "warning": None}
            try:
                if not isinstance(op, dict):
                    raise ValueError("op is not an object")
                kind = str(op.get("op", "")).upper().strip()
                res["op"] = kind
                if kind not in OPS:
                    raise ValueError(f"unknown op '{op.get('op')}'")
                if not _opt_str(op, "reason"):
                    res["warning"] = "missing reason"
                res["id"] = getattr(new, f"_op_{kind.lower()}")(op, step, evidence_ref)
                res["ok"] = True
            except (ValueError, TypeError, KeyError) as e:
                res["error"] = str(e)
            results.append(res)
        if any(r["ok"] for r in results):
            new.version = self.version + 1
        return new, results

    def _evidence(self, op: dict) -> str | None:
        ev = op.get("evidence")
        if ev is None or (isinstance(ev, str) and ev.lower() in ("", "none", "null")):
            return None
        ev = str(ev).lower().strip()
        if ev not in EVIDENCE_KINDS:
            raise ValueError(f"evidence must be one of {EVIDENCE_KINDS} or none")
        return ev

    def _op_add(self, op: dict, step: int, ref: str) -> str:
        if len(self.beliefs) >= self.cap:
            raise ValueError(f"belief cap {self.cap} reached; MERGE or DELETE first")
        ev = self._evidence(op) or "for"
        b = {
            "id": self._new_id(),
            "statement": _req_str(op, "statement"),
            "conditions": _opt_str(op, "conditions"),
            "predicted_effect": _req_str(op, "predicted_effect"),
            "confidence": _clamp_conf(op.get("confidence", 0.5)),
            "evidence_for": [ref] if ev == "for" else [],
            "evidence_against": [ref] if ev == "against" else [],
            "created_at": step,
            "updated_at": step,
        }
        self.beliefs.append(b)
        return b["id"]

    def _op_update(self, op: dict, step: int, ref: str) -> str:
        bid = _req_str(op, "id")
        b = self.get(bid)
        if b is None:
            raise ValueError(f"unknown belief id '{bid}'")
        changes = {}
        for k in EDITABLE_FIELDS:
            if k in op and op[k] is not None:
                changes[k] = _clamp_conf(op[k]) if k == "confidence" else _opt_str(op, k)
        if "statement" in changes and not changes["statement"]:
            raise ValueError("statement cannot be empty")
        ev = self._evidence(op)
        if not changes and ev is None:
            raise ValueError("UPDATE changes nothing")
        b.update(changes)
        if ev is not None and ref not in b[f"evidence_{ev}"]:
            b[f"evidence_{ev}"].append(ref)
        b["updated_at"] = step
        return bid

    def _op_merge(self, op: dict, step: int, ref: str) -> str:
        ids = op.get("ids")
        if not isinstance(ids, list) or len(set(ids)) < 2:
            raise ValueError("MERGE needs 'ids' with >= 2 distinct ids")
        olds = []
        for bid in dict.fromkeys(ids):
            b = self.get(bid)
            if b is None:
                raise ValueError(f"unknown belief id '{bid}'")
            olds.append(b)
        ev = self._evidence(op)
        ev_for = list(dict.fromkeys(e for b in olds for e in b.get("evidence_for", [])))
        ev_against = list(dict.fromkeys(e for b in olds for e in b.get("evidence_against", [])))
        if ev == "for" and ref not in ev_for:
            ev_for.append(ref)
        if ev == "against" and ref not in ev_against:
            ev_against.append(ref)
        merged = {
            "id": None,
            "statement": _req_str(op, "statement"),
            "conditions": _opt_str(op, "conditions"),
            "predicted_effect": _req_str(op, "predicted_effect"),
            "confidence": _clamp_conf(op.get("confidence",
                                             sum(b.get("confidence", 0.5) for b in olds) / len(olds))),
            "evidence_for": ev_for,
            "evidence_against": ev_against,
            "created_at": min(b.get("created_at", step) for b in olds),
            "updated_at": step,
            "merged_from": [b["id"] for b in olds],
        }
        drop = {b["id"] for b in olds}
        self.beliefs = [b for b in self.beliefs if b["id"] not in drop]
        merged["id"] = self._new_id()
        self.beliefs.append(merged)
        return merged["id"]

    def _op_delete(self, op: dict, step: int, ref: str) -> str:
        bid = _req_str(op, "id")
        if self.get(bid) is None:
            raise ValueError(f"unknown belief id '{bid}'")
        self.beliefs = [b for b in self.beliefs if b["id"] != bid]
        return bid
