import json

import pytest

from orchestrator import roles
from orchestrator.config import load_config
from orchestrator.evaluator import MockEvaluator
from orchestrator.llm import MockLLM
from orchestrator.loop import ResearchLoop
from orchestrator.mock import scripted_llm

REQUIRED_ITER_KEYS = {"iter", "x_t_hash", "f_x_t", "plans", "delta_hats", "chosen_id", "argmax_id",
                      "impl_status", "f_x_new", "delta", "e", "B_updated", "B_version_after"}
REQUIRED_CALL_KEYS = {"role", "iter", "model", "B_version", "messages", "content", "reasoning", "parsed",
                      "tokens_in", "tokens_out", "latency_s"}


def make_cfg(tmp_path, cond, T=12, extra=()):
    init = tmp_path / "init.cpp"
    init.write_text("#include <bits/stdc++.h>\n// MOCK_SCORE=50.0\nint main(){}\n")
    cfg = load_config(cond, [f"run.budget_T={T}", f"run.init_solution={init}", f"run.runs_dir={tmp_path}",
                             *extra])
    cfg["run"]["run_id"] = f"{cond}_test"
    return cfg


def run_mock(tmp_path, cond, T=12, seed=0, extra=()):
    cfg = make_cfg(tmp_path, cond, T, extra)
    loop = ResearchLoop(cfg, {"default": scripted_llm(seed=seed)}, MockEvaluator(), tmp_path / cfg["run"]["run_id"])
    summary = loop.run()
    rd = tmp_path / cfg["run"]["run_id"]
    iters = [json.loads(l) for l in (rd / "iters.jsonl").read_text().splitlines()]
    calls = [json.loads(l) for l in (rd / "calls.jsonl").read_text().splitlines()]
    return summary, iters, calls, rd


@pytest.mark.parametrize("cond", ["C0", "C1", "C2", "C3", "C4"])
def test_conditions_run_and_log(tmp_path, cond):
    summary, iters, calls, rd = run_mock(tmp_path, cond)
    assert len(iters) == 12 and summary["step"] == 12
    assert [r["iter"] for r in iters] == list(range(1, 13))
    for r in iters:
        assert REQUIRED_ITER_KEYS <= set(r)
        if r["impl_status"] != "ok":
            assert r["e"] is None and not r["B_updated"] and not r["librarian_called"]
        if r["accepted"]:
            assert r["delta"] > 0
    for c in calls:
        assert REQUIRED_CALL_KEYS <= set(c)
    roles_used = {c["role"] for c in calls}
    assert (rd / "config.yaml").exists() and (rd / "summary.json").exists() and (rd / "B" / "B_v0.json").exists()
    # best score is monotone and equals the max accepted score
    best = 50.0
    for r in iters:
        assert r["f_x_t"] == pytest.approx(best)
        if r["accepted"]:
            best = r["f_x_new"]
    assert summary["best_score"] == pytest.approx(best)

    if cond == "C0":
        assert roles_used <= {"proposer", "implementer", "implementer_fix", "implementer_repair"}
        assert all(len(r["plans"]) == 1 for r in iters if r["plans"])
    if cond == "C1":
        assert "estimator" not in roles_used and "selector" in roles_used
        assert all(r["delta_hats"] is None for r in iters)
    if cond in ("C0", "C1", "C2"):
        assert "librarian" not in roles_used
        assert summary["b_version"] == 0 and not any(r["B_updated"] for r in iters)
    if cond == "C2":
        assert all(c["B_version"] == 0 for c in calls)
    if cond == "C4":
        # every iteration with a defined error calls the librarian
        assert all(r["librarian_called"] == (r["e"] is not None) for r in iters)
    if cond == "C3":
        tau = 1.0
        assert all(r["librarian_called"] == (r["e"] is not None and abs(r["e"]) > tau) for r in iters)
        # B versions in iters and files agree
        for r in iters:
            assert (rd / "B" / f"B_v{r['B_version_after']}.json").exists()
        # estimator calls of an iteration read the B version that was current at that iteration
        by_iter = {r["iter"]: r["B_version_before"] for r in iters}
        for c in calls:
            if c["role"] == "estimator":
                assert c["B_version"] == by_iter[c["iter"]]


def test_c3_with_huge_tau_never_updates(tmp_path):
    summary, iters, calls, _ = run_mock(tmp_path, "C3", extra=["loop.tau=1e9"])
    assert summary["b_version"] == 0 and "librarian" not in {c["role"] for c in calls}


def test_resume_continues_from_checkpoint(tmp_path):
    cfg = make_cfg(tmp_path, "C3", T=4)
    rd = tmp_path / cfg["run"]["run_id"]
    ResearchLoop(cfg, {"default": scripted_llm(seed=1)}, MockEvaluator(), rd).run()
    cfg2 = make_cfg(tmp_path, "C3", T=7)
    loop = ResearchLoop(cfg2, {"default": scripted_llm(seed=2)}, MockEvaluator(), rd)
    loop.setup()
    assert loop.state["step"] == 4 and loop.state["resumes"] == 1
    loop.run()
    iters = [json.loads(l) for l in (rd / "iters.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in iters] == list(range(1, 8))


def test_init_belief_and_frozen_mode(tmp_path):
    b = tmp_path / "B.json"
    b.write_text(json.dumps({"version": 9, "beliefs": [
        {"id": "b007", "statement": "seed belief", "conditions": "", "predicted_effect": "+1",
         "confidence": 0.5, "evidence_for": [], "evidence_against": [], "created_at": 0, "updated_at": 0}]}))
    summary, iters, calls, rd = run_mock(tmp_path, "C2", T=3, extra=[f"loop.init_belief={b}"])
    est_prompts = [c["messages"][1]["content"] for c in calls if c["role"] == "estimator"]
    assert est_prompts and all("seed belief" in p for p in est_prompts)
    assert summary["b_version"] == 0


def test_c2_and_c3_prompts_identical_before_first_update(tmp_path):
    """C2 differs from C3 only by B updates: the first iteration's prompts must match exactly."""
    out = {}
    for cond in ("C2", "C3"):
        (tmp_path / cond).mkdir()
        # sequential estimates: parallel threads would consume the mock's RNG in arbitrary order
        _, _, calls, _ = run_mock(tmp_path / cond, cond, T=1, seed=5, extra=["loop.estimator_parallel=false"])
        out[cond] = [(c["role"], c["messages"]) for c in calls if c["role"] != "librarian"]
    assert out["C2"] == out["C3"]


def test_templates_render_all_roles():
    for name in ("problem", "system", "proposer", "estimator", "selector", "implementer", "implementer_fix",
                 "librarian"):
        tpl = (roles.PROMPT_DIR / f"{name}.md").read_text()
        keys = set(roles._TPL_RE.findall(tpl))
        txt = roles.render(name, **{k: "{{x}}" for k in keys})  # values are not re-scanned
        assert "{{x}}" in txt or not keys
    with pytest.raises(KeyError):
        roles.render("estimator", problem="p")


def test_all_failures_never_crash(tmp_path):
    """LLM that never produces parseable output: iterations fail gracefully, budget is consumed."""
    llm = MockLLM(lambda tag, m: "garbage")
    cfg = make_cfg(tmp_path, "C3", T=3)
    rd = tmp_path / "garbage"
    s = ResearchLoop(cfg, {"default": llm}, MockEvaluator(), rd).run()
    iters = [json.loads(l) for l in (rd / "iters.jsonl").read_text().splitlines()]
    assert s["step"] == 3 and all(r["impl_status"] == "no_plans" for r in iters)
    assert llm.n_calls == 3 * 2  # proposer + 1 parse retry per iteration
