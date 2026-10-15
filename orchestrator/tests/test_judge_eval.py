from orchestrator.evaluator import classify_judge_case, judge_result_to_eval


def case(ratio=0.0, msg="", time=10**8, mem=2**24):
    return {"scoreRatio": ratio, "msg": msg, "time": time, "memory": mem, "status": "Wrong Answer"}


def test_classify_cases():
    assert classify_judge_case(case(0.48, "points 0.48 Ratio: 0.48")) == "ok"
    assert classify_judge_case(case(0, "wrong answer Out of bounds")) == "invalid"
    assert classify_judge_case(case(0, "", time=2_000_500_000)) == "timeout"
    assert classify_judge_case(case(0, "", mem=300 * 2**20)) == "memory_limit"
    assert classify_judge_case(case(0, "(signal=11)")) == "runtime_error"
    assert classify_judge_case(case(0, "", time=0, mem=0)) == "infra"


def test_submission_status():
    ok = judge_result_to_eval({"status": "done", "score": 50.0, "scoreUnbounded": 50.0,
                               "cases": [case(0.5, "Ratio: 0.5")] * 70}, "native")
    assert ok["status"] == "ok" and ok["score"] == 50.0 and ok["judge"] == "native"
    mixed = judge_result_to_eval({"status": "done", "score": 30.0, "cases": [case(0.5, "Ratio: 0.5")] * 68
                                  + [case(0, "wrong answer x"), case(0, "", time=3 * 10**9)]}, "native")
    # 1 invalid vs 1 timeout: tie -> invalid; per-type counts kept
    assert mixed["status"] == "invalid" and mixed["score"] is None and mixed["official_score"] == 30.0
    assert mixed["status_counts"] == {"ok": 68, "invalid": 1, "timeout": 1}
    most_tle = judge_result_to_eval({"status": "done", "cases": [case(0.5, "Ratio: 0.5")] * 60
                                     + [case(0, "wrong answer x")] * 3 + [case(0, "", time=3 * 10**9)] * 7}, "native")
    assert most_tle["status"] == "timeout"
    flake = judge_result_to_eval({"status": "done", "cases": [case(0.5, "Ratio: 0.5")] * 69
                                  + [case(0, "", time=0, mem=0)]}, "native")
    assert flake["status"] == "internal_error"
    ce = judge_result_to_eval({"status": "error", "error": "Error: compile failed: main.cpp:2:21: error: expected ';'"},
                              "native")
    assert ce["status"] == "compile_error" and "expected" in ce["compile_log"]
    load = judge_result_to_eval({"status": "error", "error": "Error: compile failed: Time Limit Exceeded"}, "native")
    assert load["status"] == "internal_error"
