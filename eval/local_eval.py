#!/usr/bin/env python3
"""Fast local evaluator for Frontier-CS algorithmic problem 0 (Polyomino Packing).

Reproduces the official judge (Frontier-CS e6ef46eb, algorithmic/judge/src/{gojudge,judge_engine}.js
running on go-judge v1.11.1) without HTTP/Docker:

  compile   /usr/bin/g++ main.cpp -O2 -pipe -std=gnu++17 -o a      (gojudge.js _prepareCpp; NOTE the
            -static -s line in judge/config/langs.yaml is not read by any judge code)
  checker   /usr/bin/g++ chk.cc -O2 -pipe -std=gnu++17 -I <testlib> -o chk   (compiled once, cached by hash)
  run       CPU limit = time_limit_s (go-judge cpuLimit), wall limit = 2 x time_limit_s (clockLimit),
            RLIMIT_AS = mem + 16 KiB (memoryLimit + go-judge extra, addressSpaceLimit: true),
            RLIMIT_STACK = mem, RLIMIT_CPU = floor(t)+1 (go-judge backstop), RLIMIT_NOFILE = 256,
            stdout <= 128 MiB, env PATH=/usr/bin:/bin, no network (unshare -rn), own process group.
  case score   ratio parsed from the checker message `Ratio: <x>` iff the checker finished with exit
               code 0 or 7 (testlib _ok / quitp); otherwise 0.
  final score  (sum of per-case ratios in case order / n_cases) * 100      (judge_engine.js)

Public API (imported by the orchestrator - keep signatures):
  evaluate_code(code, *, workers=None, time_limit_s=2.0, mem_mb=256, cases=None, tag=None) -> dict
  evaluate_file(path, **kw) -> dict
Result keys: status in {"ok","compile_error","runtime_error","timeout","invalid"} (plus
"internal_error" for evaluator/infrastructure failures that are not the candidate's fault),
score / score_unbounded (official scale, None unless status == "ok"), official_score (what the
official judge would report: failed cases count 0; None on compile error), per_case, stderr_tail,
compile_log, n_cases, eval_wall_s, and some bookkeeping keys.

CLI: python eval/local_eval.py file.cpp [--workers N] [--json out.json] [--cases 1,2,5-9] [-v]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVALUATOR_VERSION = "pe-local-eval-1"

_HERE = Path(__file__).resolve().parent
PE_ROOT = Path(os.environ.get("PE_ROOT", str(_HERE.parent)))
PROBLEM_DIR = Path(os.environ.get(
    "PROBLEM_DIR", str(PE_ROOT / "third_party/Frontier-CS/algorithmic/problems/0")))
TESTLIB_DIR = Path(os.environ.get(
    "PE_TESTLIB_DIR", str(PE_ROOT / "third_party/Frontier-CS/algorithmic/judge/include")))

GXX = "/usr/bin/g++"
COMPILE_FLAGS = ["-O2", "-pipe", "-std=gnu++17"]            # gojudge.js _prepareCpp / prepareChecker
RUN_ENV = {"PATH": "/usr/bin:/bin"}                         # judge_engine.js judgeCase env

GOJUDGE_EXTRA_MEM = 16 << 10        # go-judge -extra-memory-limit default (16 KiB)
STDOUT_MAX = 128 << 20              # judgeCase files[1].max
STDERR_TAIL = 2000
COMPILE_LOG_TAIL = 4000
COMPILE_CPU_S = 10                  # gojudge.js cpuLimit: 10e9
COMPILE_MEM = 512 << 20             # gojudge.js memoryLimit: 512 << 20
COMPILE_WALL_S = 60
CHECKER_CPU_S = 10                  # judge_engine.js checker cpuLimit 10e9
CHECKER_WALL_S = 20                 # clockLimit 20e9
CHECKER_MEM = 256 << 20
TESTLIB_SCORING_EXIT_CODES = (0, 7)  # judge_engine.js isScoringVerdict
RATIO_RE = re.compile(r"Ratio: ([\d.]+)")
RATIO_UNB_RE = re.compile(r"RatioUnbounded: ([\d.]+)")

STATUS_PRIORITY = ["runtime_error", "timeout", "invalid"]   # overall status when several kinds fail

_lock = threading.Lock()
_build_lock = threading.Lock()
_sandbox_mode: str | None = None
_gxx_version: str | None = None


# --------------------------------------------------------------------------------------- helpers

def _local_tmp_root() -> Path:
    root = os.environ.get("PE_LOCAL_TMP") or os.path.join(
        os.environ.get("SLURM_TMPDIR") or "/tmp", f"pe_{os.environ.get('USER', 'user')}")
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tail(b: bytes | str, n: int) -> str:
    if isinstance(b, bytes):
        b = b.decode("utf-8", errors="replace")
    return b[-n:]


def _read_tail(path: Path, n: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return _tail(f.read(), n)
    except OSError:
        return ""


def _gxx_version_str() -> str:
    global _gxx_version
    if _gxx_version is None:
        out = subprocess.run([GXX, "--version"], capture_output=True, text=True, timeout=30)
        _gxx_version = out.stdout.splitlines()[0] if out.stdout else "unknown"
    return _gxx_version


def _cached_build(name: str, sources: list[Path], argv_fn, extra_key: str = "") -> Path:
    """Build a trusted helper binary once per node; cache keyed by content hash; atomic install."""
    h = hashlib.sha256()
    for src in sources:
        h.update(hashlib.sha256(src.read_bytes()).digest())
    h.update(extra_key.encode())
    h.update(_gxx_version_str().encode())
    cache_dir = _local_tmp_root() / "bin_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{name}_{h.hexdigest()[:24]}"
    if target.exists() and os.access(target, os.X_OK):
        return target
    with _build_lock:
        if target.exists():
            return target
        work = Path(tempfile.mkdtemp(prefix=f"build_{name}_", dir=str(cache_dir)))
        try:
            r = subprocess.run(argv_fn(work / "out"), cwd=work, capture_output=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"building {name} failed: " + _tail(r.stderr, 2000))
            os.replace(work / "out", target)  # atomic; concurrent builders just overwrite
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return target


def _launcher() -> Path:
    src = _HERE / "sandbox_run.c"
    return _cached_build("sandbox_run", [src],
                         lambda out: ["/usr/bin/gcc", "-O2", "-std=gnu11", "-Wall", str(src), "-o", str(out)])


def _sandbox_caps() -> tuple[bool, bool]:
    """Probe once: (no-network userns works, read-only-FS mount namespace works)."""
    global _sandbox_mode
    with _lock:
        if _sandbox_mode is None:
            mode = "none"
            for flags in ((True, True), (True, False)):
                try:
                    r = _run_sandboxed(["/bin/true"], cwd=_local_tmp_root(), stdin_path=None,
                                       stdout_path=None, stderr_path=None, netns=flags[0],
                                       rofs=flags[1], wall_s=10)
                    if r.get("exit_code") == 0 and not r.get("launcher_error"):
                        mode = "userns+netns+rofs" if flags[1] else "userns+netns"
                        break
                except Exception:
                    pass
            _sandbox_mode = mode
            if mode == "none":
                print("[local_eval] WARNING: unprivileged user/net namespaces unavailable; candidates "
                      "run WITHOUT network/filesystem isolation", file=sys.stderr)
    return _sandbox_mode != "none", _sandbox_mode.endswith("+rofs")


def _netns_available() -> bool:
    return _sandbox_caps()[0]


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_sandboxed(argv: list[str], *, cwd: Path, stdin_path: Path | None, stdout_path: Path | None,
                   stderr_path: Path | None, wall_s: float, netns: bool | None = None,
                   rofs: bool | None = None,
                   cpu_s: int | None = None, as_bytes: int | None = None, data_bytes: int | None = None,
                   stack_bytes: int | None = None, fsize_bytes: int | None = None,
                   nofile: int | None = None, env: dict | None = None) -> dict:
    """Run argv via the sandbox_run launcher (rlimits, optional no-network namespace, wall kill).

    The launcher runs in its own session; after it returns, the whole process group is SIGKILLed
    (stragglers forked by the candidate). Python adds a backstop kill at wall_s + 10 s.
    Returns {exit_code, signal, cpu_s, wall_s, maxrss_mb, wall_killed, launcher_error}.
    """
    if netns is None or rofs is None:
        caps = _sandbox_caps()
        netns = caps[0] if netns is None else netns
        rofs = (caps[1] and netns) if rofs is None else rofs
    la = [str(_launcher())]
    if netns:
        la.append("--netns")
        if rofs:
            la.append("--rofs")
    for flag, v in (("--cpu", cpu_s), ("--as", as_bytes), ("--data", data_bytes),
                    ("--stack", stack_bytes), ("--fsize", fsize_bytes), ("--nofile", nofile)):
        if v is not None:
            la += [flag, str(int(v))]
    la += ["--wall", f"{wall_s:.3f}"]
    rfd, wfd = os.pipe()
    la += ["--report-fd", str(wfd), "--"] + argv
    fin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    fout = open(stdout_path, "wb") if stdout_path else subprocess.DEVNULL
    ferr = open(stderr_path, "wb") if stderr_path else subprocess.DEVNULL
    p = None
    backstop = False
    t0 = time.monotonic()
    try:
        p = subprocess.Popen(la, cwd=str(cwd), stdin=fin, stdout=fout, stderr=ferr,
                             env=env if env is not None else RUN_ENV, start_new_session=True,
                             close_fds=True, pass_fds=(wfd,))
        os.close(wfd)
        wfd = -1
        try:
            p.wait(timeout=wall_s + 10.0)
        except subprocess.TimeoutExpired:
            backstop = True
            _kill_group(p.pid)
            p.wait()
        _kill_group(p.pid)
        data = b""
        while True:
            chunk = os.read(rfd, 4096)
            if not chunk:
                break
            data += chunk
        wall = time.monotonic() - t0
        if p.returncode == 0 and data:
            rep = json.loads(data.decode())
            return {"exit_code": rep["exit_code"] if rep["exit_code"] >= 0 else None,
                    "signal": rep["signal"] or None, "cpu_s": rep["cpu_s"], "wall_s": rep["wall_s"],
                    "maxrss_mb": rep["maxrss_kb"] / 1024.0, "wall_killed": bool(rep["wall_killed"]),
                    "rofs_failures": rep.get("rofs_failures"), "launcher_error": None}
        # launcher itself died (e.g. the candidate killed its parent) or could not set up
        err = "" if stderr_path is None else _read_tail(stderr_path, 300)
        return {"exit_code": None, "signal": signal.SIGKILL if backstop else None, "cpu_s": 0.0,
                "wall_s": wall, "maxrss_mb": 0.0, "wall_killed": backstop,
                "launcher_error": f"launcher rc={p.returncode} {err.strip()[-200:]}"}
    finally:
        if wfd >= 0:
            os.close(wfd)
        os.close(rfd)
        if p is not None and p.returncode is None:
            _kill_group(p.pid)
            try:
                p.wait(timeout=5)
            except Exception:
                pass
        for f in (fin, fout, ferr):
            if f is not subprocess.DEVNULL:
                try:
                    f.close()
                except Exception:
                    pass


# --------------------------------------------------------------------------------------- problem

def _parse_size_bytes(s) -> int:
    if isinstance(s, (int, float)):
        return int(s)
    m = re.match(r"^([\d.]+)\s*([kmg]?)$", str(s).strip(), re.I)
    v, u = float(m.group(1)), m.group(2).lower()
    return int(round(v * {"g": 1 << 30, "m": 1 << 20, "k": 1 << 10, "": 1}[u]))


def _parse_time_s(s) -> float:
    if isinstance(s, (int, float)):
        return float(s) / 1e9  # judge treats bare numbers as ns
    m = re.match(r"^([\d.]+)\s*(ms|s)?$", str(s).strip(), re.I)
    v, u = float(m.group(1)), (m.group(2) or "s").lower()
    return v / 1000.0 if u == "ms" else v


def problem_cases(problem_dir: Path = PROBLEM_DIR) -> list[dict]:
    """Case list exactly as problem_manager.js loadSubtasks() builds it (config cases, then extra
    complete .in/.ans pairs found in testdata/, lexicographic)."""
    import yaml  # local import: only needed here
    cfg = yaml.safe_load((problem_dir / "config.yaml").read_text())
    gt = cfg.get("time_limit") or cfg.get("time") or "1s"
    gm = cfg.get("memory_limit") or cfg.get("memory") or "256m"
    cases, cur = [], 1
    for st in cfg["subtasks"]:
        t = st.get("time_limit") or st.get("time") or gt
        m = st.get("memory_limit") or st.get("memory") or gm
        if "n_cases" in st:
            for i in range(st["n_cases"]):
                cases.append({"id": cur + i, "input": f"{cur + i}.in", "ans": f"{cur + i}.ans",
                              "time_s": _parse_time_s(t), "mem_bytes": _parse_size_bytes(m)})
            cur += st["n_cases"]
        else:
            for c in st["cases"]:
                cases.append({"id": c["input"], "input": c["input"], "ans": c["output"],
                              "time_s": _parse_time_s(c.get("time") or t),
                              "mem_bytes": _parse_size_bytes(c.get("memory") or m)})
    have = {c["input"] for c in cases}
    td = problem_dir / "testdata"
    extra = []
    for f in sorted(os.listdir(td)):
        if f.endswith(".in") and f not in have:
            base = f[:-3]
            ans = base + ".ans" if (td / (base + ".ans")).exists() else (
                base + ".out" if (td / (base + ".out")).exists() else None)
            if ans:
                extra.append({"id": int(base) if base.isdigit() else base, "input": f, "ans": ans,
                              "time_s": cases[-1]["time_s"], "mem_bytes": cases[-1]["mem_bytes"]})
    extra.sort(key=lambda c: str(c["id"]))  # localeCompare on base names
    return cases + extra


# --------------------------------------------------------------------------------------- checker

def _checker_binary() -> Path:
    """Compile the official chk.cc + testlib.h exactly like gojudge.js prepareChecker, once per node
    (cached by content hash of chk.cc, testlib.h, flags and g++ version)."""
    chk = PROBLEM_DIR / "chk.cc"
    return _cached_build("chk", [chk, TESTLIB_DIR / "testlib.h"],
                         lambda out: [GXX, str(chk), *COMPILE_FLAGS, "-I", str(TESTLIB_DIR), "-o", str(out)],
                         extra_key=" ".join(COMPILE_FLAGS))


# --------------------------------------------------------------------------------------- eval

def _compile(code: str, work: Path) -> tuple[bool, str, float]:
    (work / "main.cpp").write_text(code, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(work)}
    t0 = time.monotonic()
    # compiler runs in the sandbox too (no network); per-process CPU rlimit, data-segment memory limit
    # (go-judge's rlimit-mode equivalent of the 512 MiB compile memoryLimit).
    r = _run_sandboxed([GXX, "main.cpp", *COMPILE_FLAGS, "-o", "a"], cwd=work, stdin_path=None,
                       stdout_path=work / "compile.out", stderr_path=work / "compile.err",
                       wall_s=COMPILE_WALL_S, env=env, cpu_s=COMPILE_CPU_S + 1,
                       data_bytes=COMPILE_MEM + GOJUDGE_EXTRA_MEM, fsize_bytes=256 << 20)
    dt = time.monotonic() - t0
    log = _read_tail(work / "compile.err", COMPILE_LOG_TAIL)
    ok = r["exit_code"] == 0 and (work / "a").exists()
    if not ok:
        why = []
        if r.get("launcher_error"):
            why.append(r["launcher_error"])
        if r["wall_killed"]:
            why.append(f"compile wall-clock limit {COMPILE_WALL_S}s exceeded")
        if r["signal"]:
            why.append(f"compiler killed by signal {r['signal']}")
        if r["cpu_s"] > COMPILE_CPU_S:
            why.append(f"compile CPU limit {COMPILE_CPU_S}s exceeded")
        if why:
            log = (log + "\n[local_eval] " + "; ".join(why)).strip()
    return ok, log, dt


def _run_case(c: dict, exe: Path, chk: Path, work: Path, time_limit_s: float, mem_mb: int) -> dict:
    cdir = work / f"case_{c['id']}"
    cdir.mkdir()
    inp = PROBLEM_DIR / "testdata" / c["input"]
    ans = PROBLEM_DIR / "testdata" / c["ans"]
    out = cdir / "out.txt"
    err = cdir / "err.txt"
    mem_bytes = int(mem_mb) << 20
    cpu_rlimit = int(math.floor(time_limit_s)) + 1         # go-judge: Truncate(time,1s)+1
    r = _run_sandboxed([str(exe)], cwd=cdir, stdin_path=inp, stdout_path=out, stderr_path=err,
                       wall_s=2.0 * time_limit_s, cpu_s=cpu_rlimit,
                       as_bytes=mem_bytes + GOJUDGE_EXTRA_MEM, stack_bytes=mem_bytes,
                       fsize_bytes=STDOUT_MAX, nofile=256)
    res = {"case": c["id"], "status": None, "ratio": 0.0, "ratio_unbounded": 0.0,
           "time_s": round(r["cpu_s"], 4), "wall_s": round(r["wall_s"], 4),
           "mem_mb": round(r["maxrss_mb"], 2), "msg": ""}
    stderr_tail = _read_tail(err, STDERR_TAIL)
    # verdict order mirrors go-judge: wall/cpu kill or Time > limit => TLE; Memory > limit => MLE;
    # signals / nonzero exit => RE; output beyond the collector max => OLE.
    out_size = out.stat().st_size if out.exists() else 0
    if r["wall_killed"] or r["cpu_s"] > time_limit_s:
        res["status"] = "timeout"
        res["msg"] = (f"time limit exceeded (cpu {r['cpu_s']:.3f}s > {time_limit_s}s)" if not r["wall_killed"]
                      else f"wall-clock limit {2 * time_limit_s:.1f}s exceeded (cpu {r['cpu_s']:.3f}s)")
    elif r["maxrss_mb"] * (1 << 20) > mem_bytes:
        res["status"] = "runtime_error"
        res["msg"] = f"memory limit exceeded ({r['maxrss_mb']:.1f} MiB > {mem_mb} MiB)"
    elif r.get("launcher_error"):
        res["status"] = "runtime_error"
        res["msg"] = "sandbox launcher died (" + r["launcher_error"] + ")"
    elif r["signal"] is not None:
        res["status"] = "runtime_error"
        sig = r["signal"]
        name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else str(sig)
        oom = sig in (signal.SIGABRT, signal.SIGSEGV, signal.SIGKILL) and (
            r["maxrss_mb"] > 0.8 * mem_mb or "bad_alloc" in stderr_tail)
        extra = " (output limit exceeded)" if sig == signal.SIGXFSZ else (
            f" (likely out of memory: {mem_mb} MiB address-space limit)" if oom else "")
        res["msg"] = f"killed by signal {name}{extra}"
    elif r["exit_code"] != 0:
        res["status"] = "runtime_error"
        res["msg"] = f"nonzero exit code {r['exit_code']}"
    elif out_size > STDOUT_MAX:
        res["status"] = "runtime_error"
        res["msg"] = "output limit exceeded"
    if res["status"] is not None:
        res["stderr_tail"] = stderr_tail
        return res

    # ---- official checker: chk in.txt out.txt ans.txt
    cout, cerr = cdir / "chk.out", cdir / "chk.err"
    cr = _run_sandboxed([str(chk), str(inp), str(out), str(ans)], cwd=cdir, stdin_path=None,
                        stdout_path=cout, stderr_path=cerr, wall_s=CHECKER_WALL_S, netns=False, rofs=False,
                        cpu_s=CHECKER_CPU_S + 1, data_bytes=CHECKER_MEM + GOJUDGE_EXTRA_MEM,
                        stack_bytes=CHECKER_MEM, nofile=256)
    if cr.get("launcher_error"):
        raise RuntimeError("checker launcher failed: " + cr["launcher_error"])
    co = cout.read_bytes()[: 1 << 20].decode("utf-8", errors="replace") if cout.exists() else ""
    ce = cerr.read_bytes()[: 1 << 20].decode("utf-8", errors="replace") if cerr.exists() else ""
    chk_msg = co or ce                                       # judge_engine: stdout || stderr
    finished = (not cr["wall_killed"] and cr["signal"] is None and cr["cpu_s"] <= CHECKER_CPU_S)
    scoring = finished and cr["exit_code"] in TESTLIB_SCORING_EXIT_CODES
    ok = finished and cr["exit_code"] == 0
    bounded = unbounded = None
    if scoring:
        bounded = _js_parse_float(RATIO_RE.search(chk_msg))
        unbounded = _js_parse_float(RATIO_UNB_RE.search(chk_msg))
    ratio = bounded if bounded is not None else (1.0 if ok else 0.0)
    ratio_unb = unbounded if unbounded is not None else ratio
    res["msg"] = chk_msg.strip()[:500]
    res["ratio"], res["ratio_unbounded"] = ratio, ratio_unb
    if scoring and bounded is not None:
        res["status"] = "ok"
    elif scoring:  # _ok without Ratio (not produced by this checker); official gives ratio 1.0
        res["status"] = "ok"
    else:
        res["status"] = "invalid"
        res["ratio"] = res["ratio_unbounded"] = 0.0
        if not finished:
            res["msg"] = f"checker did not finish (exit={cr['exit_code']} sig={cr['signal']}): " + res["msg"]
    res["checker_exit"] = cr["exit_code"]
    if stderr_tail:
        res["stderr_tail"] = stderr_tail
    return res


def _js_parse_float(m) -> float | None:
    """judge_engine.js parseRatio: parseFloat(match[1]) (longest numeric prefix), finite or null."""
    if not m:
        return None
    pm = re.match(r"\d*\.?\d*", m.group(1))
    txt = pm.group(0) if pm else ""
    try:
        v = float(txt)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _parse_cases_arg(cases, all_cases: list[dict]) -> list[dict]:
    if cases is None:
        return all_cases
    want = set(cases)
    sel = [c for c in all_cases if c["id"] in want]
    missing = want - {c["id"] for c in sel}
    if missing:
        raise ValueError(f"unknown case ids: {sorted(missing, key=str)}")
    return sel


def evaluate_code(code: str, *, workers: int | None = None, time_limit_s: float = 2.0,
                  mem_mb: int = 256, cases: list[int] | None = None, tag: str | None = None,
                  fail_fast: bool = False, keep_dir: bool = False) -> dict:
    """Compile + run + check `code` on the official test cases. Never raises; never hangs."""
    t_start = time.monotonic()
    result: dict = {
        "status": "internal_error", "score": None, "score_unbounded": None, "official_score": None,
        "per_case": [], "stderr_tail": "", "compile_log": "", "n_cases": 0, "eval_wall_s": 0.0,
        "tag": tag, "code_sha256": hashlib.sha256((code or "").encode("utf-8", "replace")).hexdigest(),
        "evaluator": EVALUATOR_VERSION, "workers": None, "status_counts": {}, "compile_s": None,
        "limits": {"time_limit_s": time_limit_s, "wall_limit_s": 2 * time_limit_s, "mem_mb": mem_mb},
    }
    work: Path | None = None
    try:
        if workers is None:
            workers = min(len(os.sched_getaffinity(0)), 70)
        workers = max(1, int(workers))
        result["workers"] = workers
        _sandbox_caps()
        result["sandbox"] = _sandbox_mode
        if not isinstance(code, str) or not code.strip():
            result.update(status="compile_error", compile_log="empty source")
            return result
        sel = _parse_cases_arg(cases, problem_cases())
        result["n_cases"] = len(sel)
        chk = _checker_binary()
        work = Path(tempfile.mkdtemp(prefix=f"eval_{os.getpid()}_", dir=str(_local_tmp_root())))
        ok, log, dt = _compile(code, work)
        result["compile_log"] = log
        result["compile_s"] = round(dt, 3)
        if not ok:
            result["status"] = "compile_error"
            return result
        exe = work / "a"
        stop = threading.Event()

        def job(c):
            if stop.is_set():
                return {"case": c["id"], "status": "skipped", "ratio": 0.0, "ratio_unbounded": 0.0,
                        "time_s": None, "wall_s": None, "mem_mb": None, "msg": "skipped (fail_fast)"}
            try:
                r = _run_case(c, exe, chk, work, time_limit_s, mem_mb)
            except Exception as e:  # infrastructure problem in this case
                r = {"case": c["id"], "status": "internal_error", "ratio": 0.0, "ratio_unbounded": 0.0,
                     "time_s": None, "wall_s": None, "mem_mb": None,
                     "msg": f"{type(e).__name__}: {e}"}
            if fail_fast and r["status"] != "ok":
                stop.set()
            return r

        with ThreadPoolExecutor(max_workers=min(workers, len(sel))) as ex:
            per_case = list(ex.map(job, sel))   # preserves case order (like Promise.all)
        result["per_case"] = per_case

        counts: dict[str, int] = {}
        for r in per_case:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        result["status_counts"] = counts
        # official aggregation: sequential sum in case order, / n, * 100 (failed cases contribute 0)
        tot = tot_u = 0.0
        for r in per_case:
            tot += r["ratio"]
            tot_u += r["ratio_unbounded"]
        n = len(per_case)
        result["official_score"] = (tot / n) * 100 if n else 0.0
        official_unb = (tot_u / n) * 100 if n else 0.0
        result["official_score_unbounded"] = official_unb

        if "internal_error" in counts:
            result["status"] = "internal_error"
        elif all(r["status"] == "ok" for r in per_case):
            result["status"] = "ok"
            result["score"] = result["official_score"]
            result["score_unbounded"] = official_unb
        else:
            failed = [r for r in per_case if r["status"] not in ("ok", "skipped")]
            kinds = {r["status"] for r in failed}
            result["status"] = next(s for s in STATUS_PRIORITY if s in kinds)
        bad = [r for r in per_case if r["status"] not in ("ok", "skipped")]
        first = next((r for r in bad if r["status"] == result["status"]), bad[0] if bad else None)
        if first is not None:
            result["stderr_tail"] = (f"[case {first['case']}: {first['status']}] {first['msg']}\n"
                                     + first.get("stderr_tail", ""))[-STDERR_TAIL:]
        else:
            result["stderr_tail"] = next((r.get("stderr_tail", "") for r in per_case
                                          if r.get("stderr_tail")), "")[-STDERR_TAIL:]
        for r in per_case:
            r.pop("stderr_tail", None) if r["status"] == "ok" else None
        return result
    except BaseException as e:  # noqa: BLE001 - must never propagate into the orchestrator
        result["status"] = "internal_error"
        result["stderr_tail"] = _tail(f"{type(e).__name__}: {e}\n{traceback.format_exc()}", STDERR_TAIL)
        if isinstance(e, KeyboardInterrupt):
            raise
        return result
    finally:
        result["eval_wall_s"] = round(time.monotonic() - t_start, 3)
        if work is not None and not keep_dir:
            _rmtree_force(work)
        elif work is not None:
            result["work_dir"] = str(work)


def _rmtree_force(path: Path) -> None:
    def onerr(func, p, exc):
        try:
            os.chmod(os.path.dirname(p), 0o700)
            os.chmod(p, 0o700)
            func(p)
        except Exception:
            pass
    shutil.rmtree(path, onerror=onerr)


def evaluate_file(path: str, **kw) -> dict:
    try:
        code = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return {"status": "internal_error", "score": None, "score_unbounded": None,
                "official_score": None, "per_case": [], "stderr_tail": f"cannot read {path}: {e}",
                "compile_log": "", "n_cases": 0, "eval_wall_s": 0.0, "tag": kw.get("tag")}
    kw.setdefault("tag", os.path.basename(path))
    return evaluate_code(code, **kw)


# --------------------------------------------------------------------------------------- CLI

def _parse_case_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--json", default=None, help="write the full result dict here")
    ap.add_argument("--cases", default=None, help="e.g. 1,2,10-20 (default: all)")
    ap.add_argument("--time-limit", type=float, default=2.0)
    ap.add_argument("--mem-mb", type=int, default=256)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="print per-case lines")
    a = ap.parse_args(argv)
    res = evaluate_file(a.file, workers=a.workers, time_limit_s=a.time_limit, mem_mb=a.mem_mb,
                        cases=_parse_case_list(a.cases) if a.cases else None, tag=a.tag,
                        fail_fast=a.fail_fast)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    if a.verbose:
        for r in res.get("per_case", []):
            print(f"case {r['case']:>3} {r['status']:<13} ratio={r['ratio']:.6f} cpu={r['time_s']} "
                  f"wall={r['wall_s']} mem={r['mem_mb']}MB {r['msg'][:100]}")
    summary = {k: res.get(k) for k in ("status", "score", "score_unbounded", "official_score",
                                       "n_cases", "status_counts", "eval_wall_s", "compile_s",
                                       "workers", "sandbox", "tag")}
    if res.get("status") != "ok":
        summary["stderr_tail"] = res.get("stderr_tail", "")[-500:]
        if res.get("status") == "compile_error":
            summary["compile_log"] = res.get("compile_log", "")[-1000:]
    print(json.dumps(summary))
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
