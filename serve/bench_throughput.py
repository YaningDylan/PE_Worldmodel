"""Throughput benchmark for a vLLM OpenAI-compatible server (thinking mode, streaming).

For each concurrency level c, fires `--waves` waves of c simultaneous chat requests. Each request
has a unique ~`--prompt-tokens` prompt (unique prefix => no prefix-cache hits) and generates exactly
`--gen-tokens` tokens (ignore_eos=True, so lengths are comparable across runs).
Per request: TTFT (first streamed reasoning/content delta) and decode tok/s after the first token.
Per wave: aggregate output tok/s = sum(completion_tokens) / wave wall time.

  $PE_PY serve/bench_throughput.py --base-url $PE_VLLM_BASE_URL --model Qwen/Qwen3.5-9B \
      --concurrency 1 4 16 --prompt-tokens 2048 --gen-tokens 4096 --out results.jsonl
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import time

import httpx
from openai import AsyncOpenAI

FILLER = (
    "Polyomino packing: given a multiset of polyominoes, place each piece (rotations and reflections "
    "allowed) on an integer grid without overlap so that the bounding rectangle is as small as possible. "
    "A good heuristic sorts pieces by area, tries bottom-left placements, and repairs with local search. "
)


def tokenize_len(base_url: str, model: str, text: str) -> int:
    root = base_url.rstrip("/").removesuffix("/v1")
    r = httpx.post(f"{root}/tokenize", json={"model": model, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["count"]


def build_prompt_body(base_url: str, model: str, target: int) -> str:
    """Repeat FILLER until the user text is ~target tokens (chat template adds a few more)."""
    n_fill = tokenize_len(base_url, model, FILLER)
    reps = max(1, target // n_fill)
    body = FILLER * reps
    while tokenize_len(base_url, model, body) < target - n_fill:
        body += FILLER
    return body


async def one_request(client, model, prompt, gen_tokens, temperature, thinking):
    t0 = time.perf_counter()
    t_first = None
    usage = None
    n_chunks = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=gen_tokens,
        temperature=temperature,
        top_p=0.95,
        presence_penalty=0.0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"top_k": 20, "min_p": 0.0, "ignore_eos": True,
                    "chat_template_kwargs": {"enable_thinking": thinking}},
    )
    async for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.choices:
            d = chunk.choices[0].delta.model_dump()
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                n_chunks += 1
                if t_first is None:
                    t_first = time.perf_counter()
    t_end = time.perf_counter()
    ct = usage.completion_tokens if usage else n_chunks
    return {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": ct,
        "ttft_s": (t_first - t0) if t_first else None,
        "e2e_s": t_end - t0,
        "decode_tok_s": (ct - 1) / (t_end - t_first) if t_first and t_end > t_first else None,
    }


async def run(a):
    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=7200, max_retries=0)
    model = a.model or (await client.models.list()).data[0].id
    body = build_prompt_body(a.base_url, model, a.prompt_tokens)
    rng = random.Random(a.seed)

    def fresh_prompt():
        tag = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(16))
        return f"[request {tag}] Read the notes below, then design a better packing heuristic.\n\n{body}"

    # warmup (also triggers any lazy compilation / CUDA graph capture paths)
    await one_request(client, model, fresh_prompt(), 64, a.temperature, a.thinking)

    results = []
    for c in a.concurrency:
        for w in range(a.waves):
            t0 = time.perf_counter()
            reqs = await asyncio.gather(*[
                one_request(client, model, fresh_prompt(), a.gen_tokens, a.temperature, a.thinking)
                for _ in range(c)])
            wall = time.perf_counter() - t0
            tot_out = sum(r["completion_tokens"] for r in reqs)
            rec = {
                "label": a.label, "model": model, "concurrency": c, "wave": w, "thinking": a.thinking,
                "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] or 0 for r in reqs),
                "gen_tokens": a.gen_tokens, "wall_s": round(wall, 2),
                "agg_out_tok_s": round(tot_out / wall, 1),
                "per_req_decode_tok_s_mean": round(statistics.mean(r["decode_tok_s"] for r in reqs), 1),
                "per_req_decode_tok_s_min": round(min(r["decode_tok_s"] for r in reqs), 1),
                "ttft_s_mean": round(statistics.mean(r["ttft_s"] for r in reqs), 3),
                "ttft_s_max": round(max(r["ttft_s"] for r in reqs), 3),
                "completion_tokens_total": tot_out,
            }
            results.append(rec)
            print(json.dumps(rec), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    print("\nlabel | conc | prompt tok | gen tok | wall s | agg out tok/s | per-req decode tok/s | TTFT s (mean/max)")
    for r in results:
        print(f"{r['label']} | {r['concurrency']} | {r['prompt_tokens_mean']:.0f} | {r['gen_tokens']} | {r['wall_s']} | "
              f"{r['agg_out_tok_s']} | {r['per_req_decode_tok_s_mean']} (min {r['per_req_decode_tok_s_min']}) | "
              f"{r['ttft_s_mean']}/{r['ttft_s_max']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("PE_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("PE_VLLM_MODEL_NAME"))
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--waves", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--gen-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--label", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="append JSON lines here")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
