"""LLM clients: OpenAI-compatible (vLLM) and a scripted mock for tests."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Callable

from .parsing import _THINK_RE


@dataclass
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    max_tokens: int = 32768
    enable_thinking: bool = True
    reasoning_effort: str | None = None  # Qwen3.8 chat template: xhigh (template default) | medium | low

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMResponse:
    content: str
    reasoning: str
    finish_reason: str | None
    tokens_in: int
    tokens_out: int
    latency_s: float
    model: str
    extra: dict = field(default_factory=dict)


def _split_inline_think(content: str, reasoning: str) -> tuple[str, str]:
    """If the server did not split reasoning, move <think>...</think> out of content."""
    if reasoning or ("<think>" not in content and "</think>" not in content):
        return content, reasoning
    thinks = [m.group(0) for m in _THINK_RE.finditer(content)]
    rest = _THINK_RE.sub("", content)
    if "</think>" in rest:  # opening tag was in the prompt template
        head, rest = rest.split("</think>", 1)
        thinks.insert(0, head)
    return rest.strip(), "\n".join(thinks)


class OpenAIChat:
    """Chat client for a vLLM OpenAI-compatible server (thinking via chat_template_kwargs)."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 3600.0, http_retries: int = 3):
        from openai import OpenAI
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = OpenAI(base_url=self.base_url,
                             api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                             timeout=timeout_s, max_retries=http_retries)

    def server_info(self) -> dict:
        info = {"base_url": self.base_url, "model": self.model}
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        try:
            with urllib.request.urlopen(root + "/version", timeout=10) as r:
                info["server_version"] = json.loads(r.read().decode()).get("version")
        except Exception as e:  # noqa: BLE001 - informational only
            info["server_version_error"] = repr(e)
        return info

    def chat(self, messages: list[dict], sampling: Sampling, seed: int | None = None,
             tag: str | None = None) -> LLMResponse:
        t0 = time.time()
        template_kwargs = {"enable_thinking": sampling.enable_thinking}
        if sampling.reasoning_effort:
            template_kwargs["reasoning_effort"] = sampling.reasoning_effort
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            presence_penalty=sampling.presence_penalty,
            max_tokens=sampling.max_tokens,
            seed=seed,
            extra_body={"top_k": sampling.top_k, "chat_template_kwargs": template_kwargs},
        )
        dt = time.time() - t0
        ch = r.choices[0]
        msg = ch.message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                     or extra.get("reasoning_content") or extra.get("reasoning") or "")
        content, reasoning = _split_inline_think(msg.content or "", reasoning)
        u = r.usage
        return LLMResponse(content=content, reasoning=reasoning, finish_reason=ch.finish_reason,
                           tokens_in=getattr(u, "prompt_tokens", 0) or 0,
                           tokens_out=getattr(u, "completion_tokens", 0) or 0,
                           latency_s=dt, model=r.model or self.model)


class MockLLM:
    """Scripted LLM for tests: responder(tag, messages) -> content string."""

    def __init__(self, responder: Callable[[str | None, list[dict]], str], model: str = "mock"):
        self.responder = responder
        self.model = model
        self.n_calls = 0

    def server_info(self) -> dict:
        return {"model": self.model, "server_version": "mock"}

    def chat(self, messages, sampling, seed=None, tag=None) -> LLMResponse:
        self.n_calls += 1
        out = self.responder(tag, messages)
        n_in = sum(len(m["content"]) for m in messages) // 4
        return LLMResponse(content=out, reasoning="(mock reasoning)", finish_reason="stop",
                           tokens_in=n_in, tokens_out=len(out) // 4, latency_s=0.0, model=self.model)
