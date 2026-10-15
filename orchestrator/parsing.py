"""Robust extraction of JSON objects / C++ code / tagged fields from model output."""
from __future__ import annotations

import json
import re

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*\n?([\s\S]*?)```")
_CPP_FENCE_RE = re.compile(r"```(?:cpp|c\+\+|cc|cxx|C\+\+)\s*\n([\s\S]*?)```")


class ParseError(ValueError):
    pass


def strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    # An unterminated <think> (e.g. truncated) leaves nothing usable before </think>.
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def extract_json(text: str) -> dict:
    """Return the first JSON object found in text (whole text, fenced block, or embedded)."""
    t = strip_think(text)
    if not t:
        raise ParseError("empty output")
    candidates = [t] + [m.group(1).strip() for m in _FENCE_RE.finditer(t)]
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            obj, _ = dec.raw_decode(t[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ParseError("no JSON object found")


def extract_cpp(text: str) -> str | None:
    t = strip_think(text)
    blocks = [b for b in _CPP_FENCE_RE.findall(t) if "main" in b]
    if not blocks:
        blocks = [b for b in _FENCE_RE.findall(t) if "#include" in b and "main" in b]
    if blocks:
        return blocks[-1].strip() + "\n"
    i = t.find("#include")
    if i >= 0 and "main" in t[i:]:
        return t[i:].strip() + "\n"
    return None


def extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>\s*([\s\S]*?)\s*</{tag}>", strip_think(text))
    return m.group(1).strip() if m else None


def to_float(x) -> float:
    if isinstance(x, bool):
        raise ParseError("boolean is not a number")
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x.replace("−", "-"))
        if not m:
            raise ParseError(f"not a number: {x!r}")
        v = float(m.group(0))
    else:
        raise ParseError(f"not a number: {x!r}")
    if v != v or v in (float("inf"), float("-inf")):
        raise ParseError("non-finite number")
    return v
