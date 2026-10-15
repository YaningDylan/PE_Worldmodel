import json

from orchestrator.beliefs import BeliefDoc


def add(stmt="s", **kw):
    return {"op": "ADD", "statement": stmt, "conditions": "c", "predicted_effect": "+1", "confidence": 0.6,
            "reason": "r", **kw}


def test_add_update_merge_delete_versioning(tmp_path):
    b0 = BeliefDoc(cap=5)
    b1, res = b0.apply_ops([add("a"), add("b", evidence="against")], step=1, evidence_ref="r/iter01")
    assert [r["ok"] for r in res] == [True, True]
    assert b0.version == 0 and len(b0) == 0  # original untouched
    assert b1.version == 1 and [b["id"] for b in b1.beliefs] == ["b001", "b002"]
    assert b1.get("b001")["evidence_for"] == ["r/iter01"]
    assert b1.get("b002")["evidence_against"] == ["r/iter01"]

    b2, res = b1.apply_ops([{"op": "update", "id": "b001", "confidence": 1.7, "evidence": "against",
                             "reason": "x"}], step=2, evidence_ref="r/iter02")
    assert res[0]["ok"] and b2.version == 2
    assert b2.get("b001")["confidence"] == 1.0 and b2.get("b001")["evidence_against"] == ["r/iter02"]
    assert b2.get("b001")["updated_at"] == 2 and b2.get("b001")["created_at"] == 1

    b3, res = b2.apply_ops([{"op": "MERGE", "ids": ["b001", "b002"], "statement": "m", "predicted_effect": "0",
                             "reason": "x"}], step=3, evidence_ref="r/iter03")
    assert res[0]["ok"] and res[0]["id"] == "b003"
    assert [b["id"] for b in b3.beliefs] == ["b003"]
    m = b3.get("b003")
    assert m["merged_from"] == ["b001", "b002"] and m["created_at"] == 1
    assert set(m["evidence_against"]) == {"r/iter01", "r/iter02"}

    b4, res = b3.apply_ops([{"op": "DELETE", "id": "b003", "reason": "x"}, add("c")], step=4, evidence_ref="z")
    assert all(r["ok"] for r in res) and [b["id"] for b in b4.beliefs] == ["b004"]  # ids never reused

    p = b4.save(tmp_path)
    assert p.name == "B_v4.json"
    again = BeliefDoc.load(p, cap=5)
    assert again.to_dict() == b4.to_dict()
    assert json.loads(p.read_text())["version"] == 4


def test_invalid_ops_are_rejected_without_version_bump():
    b0 = BeliefDoc(cap=2)
    b1, res = b0.apply_ops([
        {"op": "UPDATE", "id": "b404", "confidence": 0.1},
        {"op": "RENAME", "id": "b001"},
        {"op": "ADD", "statement": "", "predicted_effect": "+1"},
        {"op": "MERGE", "ids": ["b001"]},
        "not an op",
        {"op": "ADD", "statement": "s", "predicted_effect": "+1", "confidence": "high"},
    ], step=1, evidence_ref="x")
    assert not any(r["ok"] for r in res)
    assert all(r["error"] for r in res)
    assert b1.version == 0 and len(b1) == 0


def test_cap_enforced_but_delete_then_add_works():
    b, _ = BeliefDoc(cap=2).apply_ops([add("a"), add("b")], step=1, evidence_ref="x")
    b2, res = b.apply_ops([add("c")], step=2, evidence_ref="y")
    assert not res[0]["ok"] and "cap" in res[0]["error"] and b2.version == b.version
    b3, res = b.apply_ops([{"op": "DELETE", "id": "b001", "reason": "r"}, add("c")], step=2, evidence_ref="y")
    assert all(r["ok"] for r in res) and len(b3) == 2


def test_missing_reason_is_warning_only():
    b, res = BeliefDoc().apply_ops([{"op": "ADD", "statement": "s", "predicted_effect": "+1"}], step=1,
                                   evidence_ref="x")
    assert res[0]["ok"] and res[0]["warning"] == "missing reason"


def test_render():
    assert "empty" in BeliefDoc().render()
    b, _ = BeliefDoc().apply_ops([add("hello")], step=1, evidence_ref="x")
    r = b.render()
    assert '"id": "b001"' in r and "hello" in r and "1/30" in r
