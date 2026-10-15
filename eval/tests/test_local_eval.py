"""pytest suite for eval/local_eval.py.  Run:  $PE_PY -m pytest -q eval/tests
(cheap: only a few test cases per program; safe on the login node)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))
import local_eval as le  # noqa: E402

PE_ROOT = EVAL_DIR.parent
BASELINE = PE_ROOT / "solutions" / "baseline_naive.cpp"
W = 2  # workers

TRIVIAL_INVALID = r"""
#include <bits/stdc++.h>
int main(){ int n; scanf("%d",&n); printf("1 1\n"); for(int i=0;i<n;i++) printf("0 0 0 0\n"); }
"""
TIMEOUT = r"""
#include <bits/stdc++.h>
int main(){ volatile unsigned long long x=0; for(;;) x++; }
"""
SLEEPER = r"""
#include <unistd.h>
int main(){ sleep(100); return 0; }
"""
SEGV = r"""
#include <bits/stdc++.h>
int main(){ volatile int *p = nullptr; *p = 1; return 0; }
"""
EXIT3 = r"""
int main(){ return 3; }
"""
MEMHOG = r"""
#include <bits/stdc++.h>
int main(){ std::vector<char> v(400u<<20, 1); printf("%d\n", (int)v[12345]); return 0; }
"""
BIGOUT = r"""
#include <bits/stdc++.h>
int main(){ std::string s(1<<20, 'x'); for(int i=0;i<200;i++) fwrite(s.data(),1,s.size(),stdout); return 0; }
"""
FORK_ORPHAN = r"""
#include <unistd.h>
#include <cstdio>
int main(){
  if (fork()==0){ execl("/bin/sleep","sleep","4242",(char*)0); _exit(0); }
  int n; scanf("%d",&n); printf("1 1\n"); for(int i=0;i<n;i++) printf("0 0 0 0\n");
  return 0;
}
"""
NET = r"""
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
int main(){
  int s = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(443);
  inet_pton(AF_INET, "140.82.112.3", &a.sin_addr);   // github.com
  int r = connect(s, (sockaddr*)&a, sizeof a);
  return r == 0 ? 42 : 7;   // 7 = no network (expected)
}
"""


def test_compile_error():
    r = le.evaluate_code("int main( { return 0; }", workers=W, cases=[1])
    assert r["status"] == "compile_error"
    assert r["score"] is None and r["official_score"] is None
    assert "error" in r["compile_log"]


def test_empty_source():
    assert le.evaluate_code("   ", workers=W)["status"] == "compile_error"


def test_invalid_output():
    r = le.evaluate_code(TRIVIAL_INVALID, workers=W, cases=[1, 2])
    assert r["status"] == "invalid"
    assert r["score"] is None
    assert r["official_score"] == 0.0
    assert all(c["status"] == "invalid" for c in r["per_case"])
    assert "wrong answer" in r["per_case"][0]["msg"]


def test_timeout_cpu():
    t0 = time.time()
    r = le.evaluate_code(TIMEOUT, workers=W, cases=[1])
    assert r["status"] == "timeout"
    assert r["per_case"][0]["time_s"] > 2.0
    assert time.time() - t0 < 30


def test_timeout_wall_clock():
    r = le.evaluate_code(SLEEPER, workers=W, cases=[1])
    assert r["status"] == "timeout"
    assert r["per_case"][0]["wall_s"] < 6.0           # killed at 2 x 2 s


def test_runtime_errors():
    r = le.evaluate_code(SEGV, workers=W, cases=[1])
    assert r["status"] == "runtime_error" and "SIGSEGV" in r["per_case"][0]["msg"]
    r = le.evaluate_code(EXIT3, workers=W, cases=[1])
    assert r["status"] == "runtime_error" and "exit code 3" in r["per_case"][0]["msg"]


def test_memory_limit():
    r = le.evaluate_code(MEMHOG, workers=W, cases=[1])
    assert r["status"] == "runtime_error"


def test_output_limit():
    r = le.evaluate_code(BIGOUT, workers=W, cases=[1])
    assert r["status"] == "runtime_error"
    assert "SIGXFSZ" in r["per_case"][0]["msg"]


def test_no_network():
    r = le.evaluate_code(NET, workers=W, cases=[1])
    assert r["sandbox"].startswith("userns+netns")
    assert r["status"] == "runtime_error"
    assert "exit code 7" in r["per_case"][0]["msg"], r["per_case"][0]["msg"]


def test_children_killed_and_tmp_cleaned():
    before = set(os.listdir(le._local_tmp_root()))
    r = le.evaluate_code(FORK_ORPHAN, workers=W, cases=[1, 2])
    assert r["status"] == "invalid"
    time.sleep(0.5)
    out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "sleep 4242"], capture_output=True, text=True)
    assert out.stdout.strip() == "", "orphaned child survived"
    after = set(os.listdir(le._local_tmp_root()))
    assert not [d for d in after - before if d.startswith("eval_")]


def test_baseline_ok_and_deterministic():
    r1 = le.evaluate_file(str(BASELINE), workers=W, cases=[1, 2, 3])
    r2 = le.evaluate_file(str(BASELINE), workers=W, cases=[1, 2, 3])
    assert r1["status"] == "ok", r1["stderr_tail"]
    assert r1["score"] == r2["score"]
    assert 40.0 < r1["score"] < 60.0
    assert r1["score_unbounded"] == r1["score"]
    assert r1["n_cases"] == 3 and len(r1["per_case"]) == 3
    for k in ("status", "score", "score_unbounded", "per_case", "stderr_tail", "compile_log",
              "n_cases", "eval_wall_s"):
        assert k in r1


def test_score_aggregation_matches_judge_engine():
    # judge_engine.js: (sum of ratios in case order / n) * 100, ratios parsed with parseFloat
    ratios = [0.943346906, 0.935101680, 0.942288557]
    tot = 0.0
    for x in ratios:
        tot += x
    assert le._js_parse_float(le.RATIO_RE.search("points 0.94 Ratio: 0.943346906 (cells=1)")) == 0.943346906
    assert le._js_parse_float(le.RATIO_RE.search("Ratio: 1.5.3")) == 1.5
    assert le._js_parse_float(le.RATIO_RE.search("no ratio here")) is None
    assert (tot / 3) * 100 == pytest.approx(94.02, abs=0.01)


def test_case_list_matches_config():
    cases = le.problem_cases()
    assert len(cases) == 70
    assert [c["id"] for c in cases] == list(range(1, 71))
    assert all(c["time_s"] == 2.0 and c["mem_bytes"] == 256 << 20 for c in cases)


ROFS = r"""
#include <cstdio>
#include <cstdlib>
int main(){
  const char* home = getenv("HOME");
  FILE* f = fopen("/work/mit/ppliang_mit/yaning/workspace/pe_scratch/pe_rofs_probe.txt", "w");
  FILE* g = fopen("local_ok.txt", "w");          // cwd stays writable
  if (f) { fclose(f); return 42; }               // must not be able to write outside the case dir
  if (!g) return 43;
  fclose(g);
  return 7;
}
"""


def test_readonly_filesystem_outside_case_dir():
    r = le.evaluate_code(ROFS, workers=W, cases=[1])
    if not r["sandbox"].endswith("+rofs"):
        pytest.skip("read-only FS sandbox unavailable on this node")
    assert "exit code 7" in r["per_case"][0]["msg"], r["per_case"][0]["msg"]
    assert not os.path.exists("/work/mit/ppliang_mit/yaning/workspace/pe_scratch/pe_rofs_probe.txt")
