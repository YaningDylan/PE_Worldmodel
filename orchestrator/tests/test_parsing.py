import pytest

from orchestrator.parsing import ParseError, extract_cpp, extract_json, extract_tag, to_float
from orchestrator.roles import parse_estimate, parse_generation


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('<think>{"x": 0}</think>\n{"a": 2}') == {"a": 2}
    assert extract_json('Here:\n```json\n{"a": 3}\n```\nthanks') == {"a": 3}
    assert extract_json('blah {"a": {"b": [1, 2]}} trailing } {') == {"a": {"b": [1, 2]}}
    assert extract_json('reasoning leaked</think>{"a": 4}') == {"a": 4}
    with pytest.raises(ParseError):
        extract_json("no json here")
    with pytest.raises(ParseError):
        extract_json("")


def test_extract_cpp():
    txt = "<what_changed>x</what_changed>\n```cpp\n#include <cstdio>\nint main(){}\n```"
    assert extract_cpp(txt).startswith("#include <cstdio>")
    assert extract_tag(txt, "what_changed") == "x"
    two = "```cpp\nint helper();\n```\n```cpp\n#include <a>\nint main(){return 0;}\n```"
    assert "return 0" in extract_cpp(two)
    assert extract_cpp("#include <x>\nint main(){}") is not None
    assert extract_cpp("no code") is None
    # braces like {{ in code survive
    assert "{{1,2}}" in extract_cpp("```c++\n#include <v>\nint main(){int a[1][2]={{1,2}};}\n```")


def test_to_float():
    assert to_float("+1.5 points") == 1.5
    assert to_float("−0.25") == -0.25
    assert to_float(3) == 3.0
    for bad in ("none", True, None, float("nan")):
        with pytest.raises(ParseError):
            to_float(bad)


def test_role_parsers():
    est = parse_estimate('{"cited_beliefs": "b001", "p_valid": 90, "delta_hat": "-0.5", "reason": "r"}')
    assert est == {"cited_beliefs": ["b001"], "p_valid": 0.9, "delta_hat": -0.5, "reason": "r"}
    code = "int a = 1;\nint b = 2;\n"
    ok = parse_generation(code, "x\n<<<<<<< SEARCH\nint b = 2;  \n=======\nint b = 3;\n>>>>>>> REPLACE\n", 100)
    assert ok["gen_status"] == "ok" and ok["code"] == "int a = 1;\nint b = 3;\n" and ok["n_applied"] == 1
    assert parse_generation(code, "no diff", 100)["gen_status"] == "no_valid_diff"
    assert parse_generation(code, "<<<<<<< SEARCH\nint c;\n=======\nint d;\n>>>>>>> REPLACE", 100)["gen_status"] \
        == "search_not_matching"
    assert parse_generation(code, "=======\n<<<<<<< SEARCH\nint a = 1;\n=======\nx\n>>>>>>> REPLACE", 100)[
        "gen_status"] == "no_valid_diff"
    assert parse_generation(code, "<<<<<<< SEARCH\nint a = 1;\n=======\n" + "x" * 200 + "\n>>>>>>> REPLACE",
                            100)["gen_status"] == "code_too_long"
