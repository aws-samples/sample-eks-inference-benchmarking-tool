"""Closed-loop throughput harness: how much work does the server do at a fixed in-flight depth.

Holds exactly `concurrency` requests in flight for the whole measurement window and reports
tokens/sec, req/sec and end-to-end p50/p99 per level. Non-streaming on purpose, so the
client does one response parse per request and can never become the bottleneck the way a
per-token SSE parser can. Use latency.py for token-level timing and
arrival_rate.py for "can the server hold an SLA at X arrivals/sec"; the three answer
different questions and their numbers are not interchangeable.
"""
import asyncio
import time

import aiohttp

import config
import client
from stats import percentile
from workload import describe_workload, prompt_stream

TITLE = "vLLM Benchmark (throughput, non-streaming)"


# ---------------------------------------------------------------------------- #
#                                 REQUEST PATH                                 #
# ---------------------------------------------------------------------------- #

async def request_once(session, prompt: str):
    """Send one non-streaming request.

    Returns (latency_seconds, parsed_body, failure_kind). On success failure_kind is None; on
    failure the first two are None and the kind distinguishes a broken connection from a
    server rejection, because only the latter is a property of the server under test.
    """
    url, payload = client.build_request(prompt, config.THROUGHPUT_MAX_TOKENS)
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            return time.perf_counter() - start, await response.json(), None
    except Exception as exc:
        kind = client.classify_failure(exc)
        print(f"  Request failed ({kind}): {exc}")
        return None, None, kind


def open_session(concurrency: int) -> aiohttp.ClientSession:
    """Create a session capped at the target in-flight depth, with no request timeout.

    Long prompts on a very large model can legitimately take minutes, and aiohttp's default
    total timeout would cancel them and silently bias the sample toward fast requests.
    """
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=concurrency),
        timeout=aiohttp.ClientTimeout(total=None),
    )


# ---------------------------------------------------------------------------- #
#                                 MEASUREMENT                                  #
# ---------------------------------------------------------------------------- #

async def drain_warmup(session, prompts, concurrency: int, count: int) -> None:
    """Issue and discard `count` requests so ramp-up cost stays out of the measurement."""
    if count <= 0:
        return
    print(f"  warmup: {count} requests (discarded)")
    issued = {"n": 0}

    async def warm_worker():
        """Issue warmup requests one at a time until the shared counter is exhausted."""
        while issued["n"] < count:
            issued["n"] += 1
            await request_once(session, next(prompts))

    await asyncio.gather(*[warm_worker() for _ in range(min(concurrency, count))])


def requests_for_level(num_requests: int, concurrency: int) -> int:
    """Size a level's request count so it can actually reach its labelled in-flight depth.

    A request count below the concurrency level cannot fill it: workers that never receive
    work exit immediately, so in-flight peaks at `num_requests` and the row is labelled with a
    concurrency the run never applied. One whole wave fixes the peak but leaves a window that
    is a single round-trip, where ramp-up and drain dominate the average, so the floor is
    THROUGHPUT_MIN_WAVES whole waves.

    `num_requests` stays a floor: a protocol asking for more than the wave minimum still gets
    it. Rounding up to a whole multiple keeps every wave full, matching how the latency
    harness is sized in suite.requests_for_level().
    """
    if not config.THROUGHPUT_FILL_LEVELS or concurrency <= 0:
        return num_requests
    target = max(num_requests, config.THROUGHPUT_MIN_WAVES * concurrency)
    return ((target + concurrency - 1) // concurrency) * concurrency


def unfillable_level(num_requests: int, concurrency: int) -> bool:
    """True when this level cannot possibly reach its stated in-flight depth.

    With no duration floor the workers stop as soon as the request count is exhausted, so
    fewer requests than workers means the surplus workers exit without ever issuing one.
    A duration floor removes the problem: workers keep issuing until the clock is satisfied.
    """
    return config.THROUGHPUT_MIN_DURATION_S <= 0 and num_requests < concurrency


async def measure_window(session, prompts, num_requests: int, concurrency: int):
    """Keep `concurrency` requests in flight until both the request and duration floors are met.

    Firing all requests at once and letting the connection pool throttle them reports the
    drain rate of a burst rather than sustained throughput: at high concurrency a small
    request count empties before the server ever reaches steady state.
    """
    results = []
    issued = {"n": 0}
    inflight = {"now": 0, "peak": 0}
    started = time.perf_counter()
    deadline = started + config.THROUGHPUT_MIN_DURATION_S

    async def worker():
        """Hold one request in flight at a time until both floors are satisfied."""
        # Stopping on BOTH floors means a fast server does more requests rather than
        # reporting a three-second sample.
        while not (issued["n"] >= num_requests and time.perf_counter() >= deadline):
            issued["n"] += 1
            inflight["now"] += 1
            inflight["peak"] = max(inflight["peak"], inflight["now"])
            try:
                results.append(await request_once(session, next(prompts)))
            finally:
                inflight["now"] -= 1

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return results, time.perf_counter() - started, inflight["peak"]


def summarize_level(results: list, elapsed: float, concurrency: int, peak: int) -> dict:
    """Reduce one level's raw results to the row reported in the summary table."""
    latencies = [latency for latency, _, _ in results if latency is not None]
    bodies = [body for _, body, _ in results if body is not None]
    kinds = [kind for _, _, kind in results if kind is not None]
    failures = {
        "transport_failures": kinds.count(client.TRANSPORT),
        "server_failures": kinds.count(client.SERVER),
    }
    if not latencies:
        return {"concurrency": concurrency, "error": "all requests failed",
                "successful": 0, "total": len(results), **failures}

    # Output tokens drive the headline tokens/sec figure; input tokens are recorded
    # alongside because that figure is unreadable without knowing what produced it.
    output_tokens = sum(client.usage_counts(b)[0] or 0 for b in bodies)
    input_tokens = [client.usage_counts(b)[1] or 0 for b in bodies]

    # Time-averaged in-flight depth: total busy request-seconds over wall seconds. This is
    # the load the server actually saw, and it is what the concurrency column claims to be.
    # Reporting it is what makes an under-filled level visible instead of mislabelled.
    achieved = sum(latencies) / elapsed if elapsed > 0 else 0.0

    return {
        "concurrency": concurrency,
        "achieved_concurrency": round(achieved, 1),
        "peak_inflight": peak,
        "successful": len(latencies),
        "total": len(results),
        "total_time": elapsed,
        "throughput_rps": len(latencies) / elapsed,
        "tokens_per_sec": output_tokens / elapsed if elapsed > 0 else 0,
        "output_tokens_per_request": round(output_tokens / len(latencies), 1),
        "avg_latency": sum(latencies) / len(latencies),
        "p50_latency": percentile(latencies, 0.50),
        "p99_latency": percentile(latencies, 0.99),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        **failures,
        **client.summarise_input_tokens(input_tokens),
    }


async def run_level(num_requests: int, concurrency: int) -> dict:
    """Run warmup then the measured window at one concurrency level."""
    prompts = prompt_stream()
    async with open_session(concurrency) as session:
        await drain_warmup(session, prompts, concurrency, config.THROUGHPUT_WARMUP_REQUESTS)
        results, elapsed, peak = await measure_window(session, prompts, num_requests,
                                                     concurrency)
    return summarize_level(results, elapsed, concurrency, peak)


# ---------------------------------------------------------------------------- #
#                                  REPORTING                                   #
# ---------------------------------------------------------------------------- #

def underfilled(row: dict) -> bool:
    """True when the level never reached the in-flight depth its label claims.

    Keyed on PEAK in-flight, not the time-average. Peak below the requested level is
    unambiguous: there was never enough work outstanding to fill it, so the label is wrong.
    The time-average is reported alongside but deliberately not used as the verdict, because
    it sits below the requested level on any short window purely from ramp-up and drain — a
    level that genuinely filled would then be flagged for a reason that is not a defect.
    """
    if "error" in row:
        return False
    return row["peak_inflight"] < row["concurrency"]


def print_level(row: dict) -> None:
    """Print the one-line progress result for a finished level."""
    if "error" in row:
        print(f"  concurrency={row['concurrency']:>4}  ERROR: {row['error']}")
        return
    print(f"  concurrency={row['concurrency']:>4}  "
          f"achieved={row['achieved_concurrency']:>6.1f}  "
          f"tokens/sec={row['tokens_per_sec']:>8.1f}  "
          f"req/sec={row['throughput_rps']:>6.2f}  "
          f"p50={row['p50_latency']:.3f}s  "
          f"p99={row['p99_latency']:.3f}s  "
          f"({row['successful']}/{row['total']} ok)")
    if underfilled(row):
        print(f"    <-- UNDER-FILLED: in-flight averaged "
              f"{row['achieved_concurrency']:.1f}, not {row['concurrency']}. "
              f"This row does NOT measure concurrency={row['concurrency']}.")


def print_success_lines(rows: list[dict]) -> None:
    """Emit one machine-parsable REQUEST_SUCCESS line per level."""
    for row in rows:
        print(client.request_success_line(
            row["concurrency"], row.get("successful", 0), row.get("total", 0),
            row.get("transport_failures", 0), row.get("server_failures", 0)))


def transport_share(row: dict) -> float:
    """Share of a level's requests that failed on transport."""
    total = row.get("total") or 0
    return (row.get("transport_failures", 0) / total) if total else 0.0


def level_degraded(row: dict) -> bool:
    """True when a level lost enough requests that its figures cannot be trusted."""
    total = row.get("total") or 0
    return bool(total) and (row.get("successful", 0) / total) < config.MIN_LEVEL_SUCCESS


def print_success_warning(rows: list[dict]) -> None:
    """Report levels that lost requests, and name the cause rather than guessing at it."""
    degraded = [r for r in rows if level_degraded(r)]
    if not degraded:
        return
    transport_hit = [r for r in degraded if r.get("transport_failures")]
    server_hit = [r for r in degraded if r.get("server_failures")]
    tunnelled = any(host in config.VLLM_API_URL for host in config.LOOPBACK_HOSTS)

    print()
    print("=" * 72)
    if transport_hit:
        print("RESULT INVALID — this client lost connections. Not a server measurement.")
    else:
        print("SERVER REJECTED LOAD — this is a result, not a harness fault.")
    print()
    print(f"  {'level':>6} {'ok':>12} {'share':>8} {'transport':>10} {'server':>8}")
    for row in degraded:
        print(f"  {row['concurrency']:>6} "
              f"{str(row['successful']) + '/' + str(row['total']):>12} "
              f"{row['successful'] / row['total']:>7.1%} "
              f"{row.get('transport_failures', 0):>10} "
              f"{row.get('server_failures', 0):>8}")
    print()
    if transport_hit:
        print("  WHAT IT MEANS")
        print("    A transport failure is this client's connection dropping. The percentiles")
        print("    above are computed over whatever survived, which measures the connection,")
        print(f"    not the server. Threshold: below {config.MIN_LEVEL_SUCCESS:.0%} success a "
              f"level is unusable.")
        print()
        print("  WHY")
        if tunnelled:
            print(f"    {config.VLLM_API_URL} is a local tunnel. kubectl port-forward carries")
            print("    every connection over ONE multiplexed stream to the kubelet, which")
            print("    cannot sustain hundreds of concurrent requests. This is a hard limit of")
            print("    port-forward, not something the harness can retry around — a retried")
            print("    request has different latency and changes in-flight depth.")
        else:
            print(f"    {config.VLLM_API_URL} dropped connections. Check the server's own logs")
            print("    and anything between it and this client (proxy, load balancer, MTU).")
        print()
        print("  WHAT TO DO")
        print("    Run the client inside the cluster, beside the server, so there is no tunnel:")
        print("      python3 app/bench/src/main.py launch --context <ctx> --namespace <ns> \\")
        print("          --service <service> --protocol <protocol> --instance <instance> ...")
        print("    `launch` derives the endpoint from the Service over cluster DNS and brings")
        print("    the result files back. See README > Reference > `launch` flags.")
    if server_hit and not transport_hit:
        print("  The server returned errors rather than the connection failing, so this is a")
        print("  property of the server under test at this level. Check its logs for the cause")
        print("  (out of memory, queue limits, rejected request shape).")
    print("=" * 72)


def abort_for_transport(row: dict) -> bool:
    """True when transport failures are high enough that continuing wastes time.

    Once the client's connections are failing at this rate every remaining level and rep is
    unusable too. Stopping converts a wasted multi-hour run into a wasted minute.
    """
    return transport_share(row) > config.TRANSPORT_ABORT_SHARE


def print_fill_lines(rows: list[dict]) -> None:
    """Emit one machine-parsable fill line per level for the protocol runner.

    A separate line type rather than a summary-table column: the runner's table regexes are
    matched against published result files, and adding a column would silently stop parsing
    every one of them.
    """
    for row in rows:
        if "error" in row:
            continue
        print(f"CONCURRENCY_FILL requested={row['concurrency']} "
              f"achieved={row['achieved_concurrency']} "
              f"peak={row['peak_inflight']} "
              f"requests={row['total']} "
              f"underfilled={'yes' if underfilled(row) else 'no'}")


def print_fill_warning(rows: list[dict], num_requests: int) -> None:
    """Explain any under-filled levels, and why, once at the end of the run."""
    bad = [r for r in rows if underfilled(r)]
    if not bad:
        return
    print()
    print("!" * 72)
    print("UNDER-FILLED LEVELS — the concurrency column overstates the load applied.")
    for row in bad:
        print(f"  concurrency={row['concurrency']:<5} achieved "
              f"{row['achieved_concurrency']:.1f} in-flight (peak {row['peak_inflight']})")
    if not config.THROUGHPUT_FILL_LEVELS:
        print("  Cause: THROUGHPUT_FILL_LEVELS=0, so levels were not sized to whole waves and")
        print(f"         NUM_REQUESTS={num_requests} capped in-flight below the label.")
        print("  Fix:   leave THROUGHPUT_FILL_LEVELS at its default (on).")
    print("  Do not quote these rows as capacity at their labelled concurrency.")
    print("!" * 72)


def print_summary_table(rows: list[dict]) -> None:
    """Print the aligned summary table the protocol runner parses."""
    print("\n" + "=" * 60)
    print("Summary")
    print(f"{'Concurrency':>12} {'Tokens/sec':>12} {'Req/sec':>9} {'p50 lat':>9} {'p99 lat':>9}")
    print("-" * 60)
    for row in rows:
        if "error" not in row:
            print(f"{row['concurrency']:>12} "
                  f"{row['tokens_per_sec']:>12.1f} "
                  f"{row['throughput_rps']:>9.2f} "
                  f"{row['p50_latency']:>9.3f}s "
                  f"{row['p99_latency']:>9.3f}s")
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
        "Requests/run": f"{num_requests} "
                        + ("(floor; runs until "
                           f">= {config.THROUGHPUT_MIN_DURATION_S:.0f}s)"
                           if config.THROUGHPUT_MIN_DURATION_S > 0
                           else "(CAP: no duration floor, so levels above this "
                                "cannot fill)"),
        "Max tokens": config.THROUGHPUT_MAX_TOKENS,
        "Warmup": f"{config.THROUGHPUT_WARMUP_REQUESTS} requests discarded per level",
        "Level fill": (f"on, >= {config.THROUGHPUT_MIN_WAVES} whole waves per level"
                       if config.THROUGHPUT_FILL_LEVELS
                       else "OFF - levels above the request count cannot fill"),
        # A tokens/sec number is uninterpretable without the prompt regime that produced it.
        "Prompts": describe_workload(),
    })

    # Levels that cannot fill are knowable before any request is sent, so say so up front
    # rather than leaving it to be inferred from the numbers afterwards.
    unfillable = [c for c in levels if unfillable_level(requests_for_level(num_requests, c), c)]
    if unfillable:
        print(f"  WARNING: levels {unfillable} cannot fill — request count is below the level "
              f"and MIN_DURATION_S=0.")
        print("           They will cap in-flight below their label. Set "
              "THROUGHPUT_FILL_LEVELS=1.")

    rows = []
    for concurrency in levels:
        count = requests_for_level(num_requests, concurrency)
        note = f" ({count} requests, {count // concurrency} whole waves)" if count != num_requests \
            else ""
        print(f"\nRunning at concurrency={concurrency}{note}...")
        row = await run_level(count, concurrency)
        row["instance_type"] = config.INSTANCE_TYPE
        row["quantization"] = config.QUANTIZATION
        print_level(row)
        rows.append(row)
        if abort_for_transport(row):
            print(f"\n  ABORTING SWEEP: {transport_share(row):.0%} of requests at "
                  f"concurrency={concurrency} failed on transport, above the "
                  f"{config.TRANSPORT_ABORT_SHARE:.0%} limit.")
            print("  Remaining levels are skipped: with the connection failing they would "
                  "produce")
            print("  equally unusable data. Diagnosis below.")
            break

    print_summary_table(rows)
    client.print_token_shape(rows, config.THROUGHPUT_MAX_TOKENS, with_output_per_request=True)
    print_fill_lines(rows)
    print_success_lines(rows)
    print_fill_warning(rows, num_requests)
    print_success_warning(rows)
    return rows


def main() -> None:
    """Run the full sweep, or a single level when CONCURRENCY is set."""
    levels = [config.CONCURRENCY] if config.CONCURRENCY else config.THROUGHPUT_CONCURRENCY_SWEEP
    asyncio.run(run_sweep(config.THROUGHPUT_NUM_REQUESTS, levels))


if __name__ == "__main__":
    main()
