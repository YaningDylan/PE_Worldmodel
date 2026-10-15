import pytest

from orchestrator.beliefs import BeliefDoc
from orchestrator.config import ROLES, load_config, sampling_for
from orchestrator.mock import scripted_llm
from orchestrator.probe import dedupe, evaluate_belief, metrics, spearman
from orchestrator.runlog import RunLogger


def test_spearman_and_metrics():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    rows = [{"snapshot": "a", "delta": d, "delta_hat": h} for d, h in [(1, 2), (-1, 0), (2, 3), (0.5, -1)]]
    m = metrics(rows)
    assert m["n"] == 4 and m["mae"] == pytest.approx((1 + 1 + 1 + 1.5) / 4)
    assert m["bias"] == pytest.approx((1 + 1 + 1 - 1.5) / 4)
    assert m["direction_acc"] == pytest.approx(2 / 4)  # (1,2) yes, (-1,0) no, (2,3) yes, (0.5,-1) no
    assert metrics([{"snapshot": "a", "delta": 1, "delta_hat": None}])["n"] == 0


def test_dedupe():
    plans = ["use skyline packing with best fit", "Use skyline packing with best-fit!", "try simulated annealing"]
    assert dedupe(plans) == [plans[0], plans[2]]


def test_evaluate_belief_with_mock(tmp_path):
    snap = tmp_path / "s.cpp"
    snap.write_text("int main(){}\n")
    entries = [{"snapshot": "s", "snapshot_path": str(snap), "f_x": 50.0, "plan": f"plan {i}", "plan_idx": i,
                "delta": float(i - 2)} for i in range(5)]
    cfg = load_config("C2")
    sampling = {r: sampling_for(cfg, r) for r in ROLES}
    rows = evaluate_belief(entries, BeliefDoc(), {"default": scripted_llm(0, p_bad_json=0.0)}, sampling,
                           "PROBLEM", RunLogger(tmp_path / "log"), seed=0, samples=2, parallel=3)
    assert len(rows) == 5 and all(len(r["samples"]) == 2 for r in rows)
    assert metrics(rows)["n"] == 5
