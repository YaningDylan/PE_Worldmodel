#!/usr/bin/env python3
"""Minimal client for the OFFICIAL Frontier-CS judge HTTP API (server.js), returning the full result.

The frontier CLI's --json output drops the per-case details, so for calibration we talk to the
same endpoints the official runner uses (POST /submit, GET /result/<sid>) and keep everything.

  python eval/official_judge_client.py --url http://127.0.0.1:8081 file.cpp [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests


def submit_and_wait(url: str, code: str, pid: str = "0", lang: str = "cpp",
                    timeout_s: float = 1000.0, poll_s: float = 0.5) -> dict:
    url = url.rstrip("/")
    t0 = time.time()
    r = requests.post(f"{url}/submit", files={"code": ("solution.cpp", code)},
                      data={"pid": pid, "lang": lang}, timeout=30)
    r.raise_for_status()
    sid = r.json()["sid"]
    while time.time() - t0 < timeout_s:
        rr = requests.get(f"{url}/result/{sid}", timeout=10)
        if rr.status_code == 200:
            res = rr.json()
            if res.get("status") in ("done", "error"):
                res["_sid"] = sid
                res["_wall_s"] = time.time() - t0
                return res
        time.sleep(poll_s)
    return {"status": "client_timeout", "_sid": sid, "_wall_s": time.time() - t0}


def summarize(res: dict) -> dict:
    cases = res.get("cases") or []
    statuses: dict[str, int] = {}
    for c in cases:
        statuses[c.get("status")] = statuses.get(c.get("status"), 0) + 1
    return {
        "status": res.get("status"),
        "score": res.get("score"),
        "scoreUnbounded": res.get("scoreUnbounded"),
        "n_cases": len(cases),
        "case_statuses": statuses,
        "max_time_ms": max([c.get("time", 0) for c in cases] or [0]) / 1e6,
        "max_mem_mb": max([c.get("memory", 0) for c in cases] or [0]) / 2**20,
        "wall_s": round(res.get("_wall_s", 0.0), 2),
        # known flake of the natively hosted judge: a case that never ran (time 0, memory 0, no msg)
        "suspect_infra_cases": [i + 1 for i, c in enumerate(cases)
                                if c.get("time", 0) == 0 and c.get("memory", 0) == 0 and not c.get("msg")],
        "error": res.get("error"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--pid", default="0")
    ap.add_argument("--json", default=None, help="write the full judge result here")
    a = ap.parse_args()
    code = open(a.file, encoding="utf-8").read()
    res = submit_and_wait(a.url, code, pid=a.pid)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps(summarize(res)))
    return 0 if res.get("status") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
