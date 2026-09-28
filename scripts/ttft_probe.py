#!/usr/bin/env python3
"""Single-request TTFT/TPOT probe for an attention-backend (or any one-variable) A/B.

Runs INSIDE the serving pod against localhost, so both arms of an A/B are measured the same
way with no network or tunnel in the path. Sends the prompt profile's prompts one at a time
(concurrency 1), shaped exactly like the benchmark harness sends them (client.py: one user
message, temperature 0.7, stream with include_usage), and prints one JSON line of medians.

Standard library only, because the vLLM image is all it has:

    kubectl exec -i POD -c vllm -- python3 - --prompts /tmp/p.json < scripts/ttft_probe.py

Exits non-zero if any request fails or reports no tokens: a probe that quietly averages over
failures would certify a broken arm as fast.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

TEMPERATURE = 0.7  # config.BENCH_TEMPERATURE, so the probe matches the harness


def served_model(base: str) -> str:
    """Return the single model id the server reports, so no model name is guessed."""
    with urllib.request.urlopen(f"{base}/models", timeout=30) as response:
        models = [m["id"] for m in json.load(response)["data"]]
    if len(models) != 1:
        sys.exit(f"expected exactly one served model, got {models}")
    return models[0]


def stream_request(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Send one streaming chat request and time its first token and its completion."""
    body = {"model": model, "max_tokens": max_tokens, "temperature": TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first, usage = None, None
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if first is None and (delta.get("content") or delta.get("reasoning_content")):
                first = time.perf_counter()
    end = time.perf_counter()
    return {"start": start, "first": first, "end": end, "usage": usage}


def to_sample(result: dict) -> dict:
    """Turn a raw timing into TTFT/TPOT, failing loudly on a response with no tokens."""
    usage = result["usage"] or {}
    tokens = usage.get("completion_tokens", 0)
    if result["first"] is None or tokens < 2:
        raise RuntimeError(f"request produced no usable tokens (usage={usage})")
    return {"ttft_s": result["first"] - result["start"],
            "tpot_ms": 1000 * (result["end"] - result["first"]) / (tokens - 1),
            "prompt_tokens": usage.get("prompt_tokens", 0)}


def summarize(samples: list[dict], model: str) -> dict:
    """Medians plus the spread needed to judge whether two arms really differ."""
    ttft = sorted(s["ttft_s"] for s in samples)
    return {"model": model, "n": len(samples),
            "ttft_p50_s": round(statistics.median(ttft), 4),
            "ttft_p90_s": round(ttft[int(0.9 * (len(ttft) - 1))], 4),
            "ttft_min_s": round(ttft[0], 4),
            "tpot_p50_ms": round(statistics.median(s["tpot_ms"] for s in samples), 2),
            "prompt_tokens_p50": int(statistics.median(s["prompt_tokens"] for s in samples))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--prompts", required=True, help="prompt profile JSON (prompt_profiles/*.json)")
    parser.add_argument("-n", type=int, default=16, help="measured requests, cycling the profile")
    parser.add_argument("--warmup", type=int, default=2, help="unmeasured requests first (JIT, graphs)")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    prompts = json.load(open(args.prompts))["prompts"]
    model = served_model(args.url)
    for i in range(args.warmup):
        to_sample(stream_request(args.url, model, prompts[i % len(prompts)], args.max_tokens))
    samples = [to_sample(stream_request(args.url, model, prompts[i % len(prompts)], args.max_tokens))
               for i in range(args.n)]
    print(json.dumps(summarize(samples, model)))


if __name__ == "__main__":
    main()
