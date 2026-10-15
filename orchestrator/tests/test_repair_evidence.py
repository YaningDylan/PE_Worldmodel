import json

from orchestrator import evidence, roles
from orchestrator.config import load_config
from orchestrator.evaluator import MockEvaluator, submission_status
from orchestrator.llm import MockLLM
from orchestrator.loop import ResearchLoop
from orchestrator.mock import scripted_llm

PROG = "#include <bits/stdc++.h>\n// MOCK_SCORE={s}\n// {st}\nint main() {{ return 0; }}\n"


def test_submission_status_majority():
    assert submission_status({"ok": 70}, 70) == "ok"
    assert submission_status({"ok": 60, "invalid": 6, "timeout": 4}, 70) == "invalid"
    assert submission_status({"ok": 56, "invalid": 4, "timeout": 10}, 70) == "timeout"
    assert submission_status({"ok": 68, "invalid": 1, "infra": 1}, 70) == "internal_error"


def run_with(tmp_path, implementer_outputs, T=1, extra=()):
    """Scripted run where the implementer/repair calls return the given programs in order."""
    init = tmp_path / "init.cpp"
    init.write_text(PROG.format(s=50.0, st="MOCK_STATUS=ok"))
    cfg = load_config("C3", [f"run.budget_T={T}", f"run.init_solution={init}", f"run.runs_dir={tmp_path}",
                             "loop.estimator_parallel=false", *extra])
    cfg["run"]["run_id"] = "r"
    inner = scripted_llm(0, p_bad_json=0.0)
    seen = []
    outs = iter(implementer_outputs)

    def respond(tag, messages):
        if tag in ("implementer", "implementer_fix", "implementer_repair"):
            seen.append((tag, messages[-1]["content"]))
            note, code = next(outs)
            return f"<what_changed>{note}</what_changed>\n```cpp\n{code}```"
        return inner.responder(tag, messages)

    loop = ResearchLoop(cfg, {"default": MockLLM(respond)}, MockEvaluator(), tmp_path / "r")
    loop.run()
    iters = [json.loads(l) for l in (tmp_path / "r" / "iters.jsonl").read_text().splitlines()]
    calls = [json.loads(l) for l in (tmp_path / "r" / "calls.jsonl").read_text().splitlines()]
    return iters, calls, seen


def test_runtime_repair_then_ok_keeps_summary(tmp_path):
    iters, calls, seen = run_with(tmp_path, [
        ("original change", PROG.format(s=55.0, st="MOCK_STATUS=invalid")),
        ("repair 1", PROG.format(s=55.0, st="MOCK_STATUS=timeout")),
        ("repair 2", PROG.format(s=55.0, st="MOCK_STATUS=ok")),
    ])
    r = iters[0]
    assert r["impl_status_first"] == "invalid" and r["impl_status"] == "ok" and r["n_runtime_fixes"] == 2
    assert [x["trigger_status"] for x in r["repairs"]] == ["invalid", "timeout"]
    assert [x["status_after"] for x in r["repairs"]] == ["timeout", "ok"]
    assert r["what_changed"] == "original change" and r["repair_notes"] == ["repair 1", "repair 2"]
    assert [t for t, _ in seen] == ["implementer", "implementer_repair", "implementer_repair"]
    report = seen[1][1]
    assert "Overlap at cell (3,4)" in report and "invalid=" in report
    assert "CPU 2.50 s" in seen[2][1]
    assert r["delta"] == 5.0 and r["output_identical"] is False
    assert r["case_cmp"]["improved"] == 70


def test_repair_budget_exhausted_is_failure(tmp_path):
    bad = PROG.format(s=55.0, st="MOCK_STATUS=invalid")
    iters, _, _ = run_with(tmp_path, [("x", bad), ("r1", bad), ("r2", bad)])
    r = iters[0]
    assert r["impl_status"] == "invalid" and r["n_runtime_fixes"] == 2 and r["e"] is None
    assert not r["librarian_called"] and r["output_identical"] is None


def test_output_identical_flag_and_librarian_evidence(tmp_path):
    iters, calls, _ = run_with(tmp_path, [("noop", PROG.format(s=50.0, st="MOCK_STATUS=ok"))])
    r = iters[0]
    assert r["delta"] == 0.0 and r["output_identical"] is True and r["case_cmp"]["unchanged"] == 70
    lib = [c for c in calls if c["role"] == "librarian"]
    if r["librarian_called"]:
        u = lib[0]["messages"][1]["content"]
        assert "Output identical to the previous solution on all 70 cases" in u and ": YES" in u
        assert "Diff size:" in u and "Changed functions:" in u


def test_code_diff_changed_functions():
    old = "#include <x>\nint helper(int a) {\n  return a + 1;\n}\nint main() {\n  return helper(1);\n}\n"
    new = "#include <x>\nint helper(int a) {\n  return a + 2;\n}\nint main() {\n  return helper(1);\n}\n"
    d = evidence.code_diff(old, new)
    assert d["added"] == 1 and d["removed"] == 1 and d["changed_functions"] == ["helper"]
    lam = "int main() {\n  auto place = [&](int x) {\n    return x;\n  };\n}\n"
    d2 = evidence.code_diff(lam, lam.replace("return x;", "return x + 1;"))
    assert d2["changed_functions"] == ["place"]


def test_case_evidence_text():
    cmp = {"n": 70, "identical": True, "improved": 0, "worsened": 0, "unchanged": 70,
           "mean_delta": 0.0, "min_delta": 0.0, "max_delta": 0.0}
    t = roles.case_evidence_text(cmp)
    assert "YES" in t and "0 / 0 / 70" in t
    assert "unavailable" in roles.case_evidence_text(None)
