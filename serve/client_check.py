"""One-shot OpenAI-client check against a vLLM server started by serve/launch_vllm.sh.

Sends one chat request with the plan's thinking-mode coding sampling params and reports where the
reasoning text lands in the response (message.reasoning vs message.reasoning_content vs inside
message.content) plus usage token counts.

  $PE_PY serve/client_check.py --base-url http://127.0.0.1:8000/v1 [--model Qwen/Qwen3.5-9B]
         [--max-tokens 4096] [--no-thinking] [--json]
Defaults come from $PE_VLLM_BASE_URL / $PE_VLLM_MODEL_NAME (see the run dir's server.env).
Exit code 0 iff the request succeeded and the reasoning/content split matches the requested mode.
"""
import argparse
import json
import os
import sys
import time

from openai import OpenAI

PROMPT = (
    "Write a C++ function `int count_tilings(int n)` that returns the number of ways to tile a "
    "2 x n board with 1 x 2 dominoes, modulo 1e9+7. Briefly justify the recurrence."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("PE_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("PE_VLLM_MODEL_NAME"))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-thinking", action="store_true", help="send chat_template_kwargs.enable_thinking=False")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--json", action="store_true", help="also print one JSON summary line")
    a = ap.parse_args()

    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=3600)
    model = a.model or client.models.list().data[0].id
    thinking = not a.no_thinking
    t0 = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": a.prompt}],
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        top_p=0.95,
        presence_penalty=0.0,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
    )
    dt = time.time() - t0
    msg = r.choices[0].message
    raw = msg.model_dump()
    fields = {k: v for k, v in raw.items() if k in ("reasoning", "reasoning_content") and v}
    content = msg.content or ""
    think_in_content = "<think>" in content or "</think>" in content
    u = r.usage
    print(f"model={model} thinking={'on' if thinking else 'off'} finish_reason={r.choices[0].finish_reason} wall={dt:.1f}s")
    print(f"message keys: {sorted(raw.keys())}")
    for k in ("reasoning", "reasoning_content"):
        v = raw.get(k)
        print(f"message.{k}: {'None' if v is None else f'{len(v)} chars'}")
    print(f"message.content: {len(content)} chars; contains <think>/</think> tags: {think_in_content}")
    reasoning_text = next(iter(fields.values()), "")
    print("--- reasoning (first 300 chars) ---\n" + reasoning_text[:300].strip())
    print("--- content (first 300 chars) ---\n" + content[:300].strip())
    details = getattr(u, "completion_tokens_details", None)
    print(f"usage: prompt_tokens={u.prompt_tokens} completion_tokens={u.completion_tokens} "
          f"total_tokens={u.total_tokens} completion_tokens_details={details}")
    ok = (bool(fields) and bool(content) and not think_in_content) if thinking else (not fields and bool(content))
    if r.choices[0].finish_reason == "length":
        print("NOTE: hit max_tokens; content may be empty because the model was still thinking")
    print(f"reasoning field(s) populated: {sorted(fields) or 'none'} -> {'OK' if ok else 'UNEXPECTED'}")
    if a.json:
        print(json.dumps({"model": model, "thinking": thinking, "ok": ok, "reasoning_fields": sorted(fields),
                          "finish_reason": r.choices[0].finish_reason, "prompt_tokens": u.prompt_tokens,
                          "completion_tokens": u.completion_tokens, "wall_s": round(dt, 2)}))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
