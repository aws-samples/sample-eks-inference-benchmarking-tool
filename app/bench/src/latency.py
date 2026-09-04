"""Closed-loop streaming latency harness: TTFT, TPOT and ITL at a representative in-flight depth.

TTFT is request start to first content chunk, so it reflects prefill and interactivity; ITL
is the gap between successive chunks, so it reflects decode smoothness and tails; TPOT is
steady-state decode cost as (end_to_end - ttft) / (tokens - 1). Streaming is required to see
any of them, which means the client parses per token and can itself become the bottleneck —
run this from a well-provisioned host and check the runner's client-CPU column before
trusting a high-concurrency row. Requests go out in waves of exactly `concurrency` so no
request waits in a client-side queue and has that wait charged to its TTFT.
"""
import asyncio
import time

import aiohttp

import config
import client
from stats import percentile
from workload import build_prompts, describe_workload

TITLE = "vLLM Latency Probe (streaming: TTFT / TPOT / ITL)"


# ---------------------------------------------------------------------------- #
#                                 REQUEST PATH                                 #
# ---------------------------------------------------------------------------- #

async def stream_once(session, prompt: str) -> dict | None:
    """Stream one request and return its per-request latency metrics, or None on failure."""
    url, payload = client.build_request(prompt, config.LATENCY_MAX_TOKENS, stream=True)
    start = time.perf_counter()
    first_token_at = None
    piece_times: list[float] = []
    counted_pieces = 0
    reported_output = None
    reported_input = None
    try:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            async for chunk in client.iter_sse_chunks(response):
                output_tokens, input_tokens = client.usage_counts(chunk)
                reported_output = output_tokens if output_tokens is not None else reported_output
                reported_input = input_tokens if input_tokens is not None else reported_input
                if not client.token_piece(chunk):
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                piece_times.append(now)
                counted_pieces += 1
            end_to_end = time.perf_counter() - start
    except Exception as exc:
        kind = client.classify_failure(exc)
        print(f"  Request failed ({kind}): {exc}")
        return {"failure": kind}

    if first_token_at is None:
        # Completed without ever emitting a token: the server answered, so this counts as a
        # server-side failure rather than a broken connection.
        return {"failure": client.SERVER}
    return derive_metrics(start, first_token_at, end_to_end, piece_times,
                          reported_output or counted_pieces, reported_input)


def derive_metrics(start: float, first_token_at: float, end_to_end: float,
                   piece_times: list[float], output_tokens: int,
                   input_tokens: int | None) -> dict:
    """Turn raw per-chunk timestamps into the TTFT / ITL / TPOT figures for one request."""
    inter_token = [piece_times[i] - piece_times[i - 1] for i in range(1, len(piece_times))]
    return {
        "e2e": end_to_end,
        "ttft": first_token_at - start,
        "output_tokens": output_tokens,
        "itls": inter_token,
        "tpot": (end_to_end - (first_token_at - start)) / (output_tokens - 1)
                if output_tokens > 1 else None,
        "prompt_tokens": input_tokens,
    }


# ---------------------------------------------------------------------------- #
#                                 MEASUREMENT                                  #
# ---------------------------------------------------------------------------- #

async def run_level(num_requests: int, concurrency: int) -> dict:
    """Issue requests in whole waves of `concurrency` and summarise the level.

    Gathering all requests at once would let the connector cap concurrent connections, so
    surplus requests queue on the CLIENT and that wait is charged to TTFT. That produced an
    obviously inverted result — TTFT p50 of 70.9s at concurrency 1 against 0.74s at
    concurrency 32 — which measured our own backlog rather than the server.
    """
    prompts = build_prompts(num_requests)
    results = []
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector,
                                     timeout=aiohttp.ClientTimeout(total=None)) as session:
        for offset in range(0, num_requests, concurrency):
            wave = prompts[offset:offset + concurrency]
            results.extend(await asyncio.gather(*[stream_once(session, p) for p in wave]))
    return summarize_level(results, num_requests, concurrency)


def summarize_level(results: list, num_requests: int, concurrency: int) -> dict:
    """Reduce one level's per-request metrics to the row reported in the summary table."""
    usable = [r for r in results if r is not None and "failure" not in r]
    kinds = [r["failure"] for r in results if r is not None and "failure" in r]
    failures = {
        "transport_failures": kinds.count(client.TRANSPORT),
        "server_failures": kinds.count(client.SERVER),
    }
    if not usable:
        return {"concurrency": concurrency, "error": "all requests failed",
                "total": num_requests, "successful": 0, **failures}

    time_to_first = [r["ttft"] for r in usable]
    per_output_token = [r["tpot"] for r in usable if r["tpot"] is not None]
    inter_token = [gap for r in usable for gap in r["itls"]]
    # TTFT is dominated by prefill once prompts are realistic, so a TTFT figure without its
    # input length is not interpretable.
    input_tokens = [r["prompt_tokens"] or 0 for r in usable]

    return {
        "concurrency": concurrency,
        "successful": len(usable),
        "total": num_requests,
        "ttft_p50": percentile(time_to_first, 0.50),
        "ttft_p99": percentile(time_to_first, 0.99),
        "tpot_p50_ms": percentile(per_output_token, 0.50) * 1000.0,
        "tpot_p99_ms": percentile(per_output_token, 0.99) * 1000.0,
        "itl_p50_ms": percentile(inter_token, 0.50) * 1000.0,
        "itl_p99_ms": percentile(inter_token, 0.99) * 1000.0,
        **failures,
        **client.summarise_input_tokens(input_tokens),
    }


# ---------------------------------------------------------------------------- #
#                                  REPORTING                                   #
# ---------------------------------------------------------------------------- #

def print_level(row: dict) -> None:
    """Print the one-line progress result for a finished level."""
    if "error" in row:
        print(f"  concurrency={row['concurrency']:>4}  ERROR: {row['error']}")
        return
    print(f"  concurrency={row['concurrency']:>4}  "
          f"TTFT p50={row['ttft_p50']:.3f}s/p99={row['ttft_p99']:.3f}s  "
          f"TPOT p50={row['tpot_p50_ms']:.1f}ms/p99={row['tpot_p99_ms']:.1f}ms  "
          f"ITL p50={row['itl_p50_ms']:.1f}ms/p99={row['itl_p99_ms']:.1f}ms  "
          f"({row['successful']}/{row['total']} ok)")


def print_summary_table(rows: list[dict]) -> None:
    """Print the aligned summary table the protocol runner parses."""
    print("\n" + "=" * 60)
    print("Summary — Latency (streaming)")
    print(f"{'Concurrency':>12} {'TTFT p50':>10} {'TTFT p99':>10} "
          f"{'TPOT p50':>10} {'TPOT p99':>10} {'ITL p50':>9} {'ITL p99':>9}")
    print("-" * 60)
    for row in rows:
        print(f"{row['concurrency']:>12} "
              f"{row['ttft_p50']:>9.3f}s "
              f"{row['ttft_p99']:>9.3f}s "
              f"{row['tpot_p50_ms']:>8.1f}ms "
              f"{row['tpot_p99_ms']:>8.1f}ms "
              f"{row['itl_p50_ms']:>7.1f}ms "
              f"{row['itl_p99_ms']:>7.1f}ms")
    print("=" * 60)


# ---------------------------------------------------------------------------- #
#                                 ENTRY POINT                                  #
# ---------------------------------------------------------------------------- #

async def run_sweep(num_requests: int, levels: list[int]) -> list[dict]:
    """Run every concurrency level in turn and report the aggregate."""
    client.print_header(TITLE, {
        "Endpoint": config.VLLM_API_URL,
        "Model": config.MODEL_ID,
        "Instance": config.INSTANCE_TYPE,
        "Quantization": config.QUANTIZATION,
        "Requests/run": num_requests,
        "Max tokens": config.LATENCY_MAX_TOKENS,
        "Prompts": describe_workload(),
    }, notes=("NOTE: run at representative concurrency, not peak; "
              "use a well-provisioned client.",))

    rows = []
    for concurrency in levels:
        print(f"\nRunning at concurrency={concurrency}...")
        row = await run_level(num_requests, concurrency)
        row["instance_type"] = config.INSTANCE_TYPE
        row["quantization"] = config.QUANTIZATION
        print_level(row)
        rows.append(row)
        # Same abort as the throughput sweep: once the client's connections are failing, the
        # remaining levels measure the broken connection. This sweep is the expensive one —
        # on 2026-08-08 it ran for two hours after its tunnel had already collapsed.
        total = row.get("total") or 0
        share = (row.get("transport_failures", 0) / total) if total else 0.0
        if share > config.TRANSPORT_ABORT_SHARE:
            print(f"\n  ABORTING SWEEP: {share:.0%} of requests at concurrency={concurrency} "
                  f"failed on transport, above the {config.TRANSPORT_ABORT_SHARE:.0%} limit.")
            print("  Remaining levels are skipped. Run in-cluster: `main.py launch`.")
            break

    usable = [r for r in rows if "error" not in r]
    print_summary_table(usable)
    client.print_token_shape(rows, config.LATENCY_MAX_TOKENS, with_output_per_request=False)
    for row in rows:
        print(client.request_success_line(
            row["concurrency"], row.get("successful", 0), row.get("total", 0),
            row.get("transport_failures", 0), row.get("server_failures", 0)))
    return rows


def main() -> None:
    """Run the full sweep, or a single level when CONCURRENCY is set."""
    levels = [config.CONCURRENCY] if config.CONCURRENCY else config.LATENCY_CONCURRENCY_SWEEP
    asyncio.run(run_sweep(config.LATENCY_NUM_REQUESTS, levels))


if __name__ == "__main__":
    main()
