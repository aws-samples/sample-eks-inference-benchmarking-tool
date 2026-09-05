"""Closed-loop throughput harness backed by NVIDIA AIPerf, sitting behind the suite's harness seam. It
reads the same environment contract as throughput.py, translates it into one aiperf profile
invocation per concurrency level, parses the typed JSON export rather than stdout, and re-emits
this repo's stdout contract so the suite's medians, spread flags and success-floor exclusion work
unchanged. Token counts come from the server because the run passes --use-server-token-count, and
AIPerf does not separate transport from server failures, so all its errors are reported as server
failures and transport_failures stays 0 without affecting the ok_fraction the exclusion rule keys
on. A pinned prompt profile is required, since generated prompt modes are unsupported here. Setting
AIPERF_MODE=trace replays a committed timestamped trace once under aiperf's fixed schedule, taking
arrivals from the trace rather than a concurrency knob and emitting a single TRACE_ROW line, whose
streaming metrics are AIPerf's own definitions and are not interchangeable with latency.py's
columns."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import aiperf_profiles
import client
import config
from throughput import requests_for_level

# A tokenizer AIPerf insists on loading; it never influences a count, all of which are server usage.
AIPERF_TOKENIZER = os.environ.get("AIPERF_TOKENIZER", "gpt2")
AIPERF_BIN = os.environ.get("AIPERF_BIN", "aiperf")
# Pinned rather than sized from host CPU count, so the recorded argv does not depend on where it ran.
AIPERF_WORKERS_MAX = int(os.environ.get("AIPERF_WORKERS_MAX", "16"))


def build_argv(level: int, requests: int, dataset: Path, artifacts: Path) -> list[str]:
    """Translate the harness environment into one aiperf invocation for one level."""
    endpoint_type = {"chat": "chat", "completions": "completions"}[
        os.environ.get("BENCH_ENDPOINT", "chat")]
    return [
        AIPERF_BIN, "profile",
        "--model-names", config.MODEL_ID,
        "--tokenizer", AIPERF_TOKENIZER,
        "--use-server-token-count",
        "--url", config.VLLM_API_URL.removesuffix("/v1"),
        "--endpoint-type", endpoint_type,
        "--input-file", str(dataset),
        "--custom-dataset-type", "single_turn",
        "--concurrency", str(level),
        # Scaled by throughput.py's imported rule, since a flat count below the level cannot fill it.
        "--request-count", str(requests),
        "--workers-max", str(AIPERF_WORKERS_MAX),
        # aiperf 0.12.0 validates warmup-request-count > 0, so zero means omit the flag.
        *(["--warmup-request-count", str(config.THROUGHPUT_WARMUP_REQUESTS)]
          if config.THROUGHPUT_WARMUP_REQUESTS > 0 else []),
        "--random-seed", str(config.PROMPT_SEED),
        "--output-artifact-dir", str(artifacts),
        "--ui", "simple",
    ]


def run_level(level: int, profile_name: str, output_cap: int, scratch: Path) -> dict:
    """Run one concurrency level and reduce its export to a summary-row dict.

    The dataset is rendered per level, sized to that level's scaled request count, so
    every request maps 1:1 to a dataset line — aiperf is never left to decide how to
    stretch or sample a fixed pool, which could diverge from workload.build_prompts.
    """
    requests = requests_for_level(config.THROUGHPUT_NUM_REQUESTS, level)
    artifacts = scratch / f"level-{level}"
    dataset = aiperf_profiles.profile_to_jsonl(
        profile_name, requests, output_cap, scratch / f"dataset-{level}.jsonl")
    argv = build_argv(level, requests, dataset, artifacts)
    print(f"AIPERF_ARGV level={level} {' '.join(argv)}")
    done = subprocess.run(argv, capture_output=True, text=True)
    export = artifacts / "profile_export_aiperf.json"
    if done.returncode != 0 or not export.exists():
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-8:]
        return {"concurrency": level, "error": "aiperf failed", "detail": "\n".join(tail)}
    return reduce_export(level, json.loads(export.read_text()))


def reduce_export(level: int, data: dict) -> dict:
    """Map one typed export onto the fields the stdout contract needs."""
    def stat(name: str, key: str, default: float = 0.0) -> float:
        entry = data.get(name) or {}
        value = entry.get(key)
        return float(value) if value is not None else default

    completed = int(stat("request_count", "avg"))
    failures = sum(int(e.get("count", 0)) for e in data.get("error_summary", []))
    fill = data.get("effective_concurrency") or {}
    return {
        "concurrency": level,
        "tokens_per_sec": stat("output_token_throughput", "avg"),
        "throughput_rps": stat("request_throughput", "avg"),
        "p50_latency": stat("request_latency", "p50") / 1000.0,
        "p99_latency": stat("request_latency", "p99") / 1000.0,
        "successful": completed,
        "total": completed + failures,
        "transport_failures": 0,          # aiperf does not split; see module docstring
        "server_failures": failures,
        "achieved_concurrency": float(fill.get("avg", 0.0)),
        "peak_inflight": int(fill.get("max", 0)),
        "shape": {
            "input_tokens_median": int(stat("usage_prompt_tokens", "p50")),
            "input_tokens_min": int(stat("usage_prompt_tokens", "min")),
            "input_tokens_max": int(stat("usage_prompt_tokens", "max")),
        },
        "output_per_request": stat("usage_completion_tokens", "avg"),
        "aiperf_version": data.get("aiperf_version", "unknown"),
    }


def print_contract(rows: list[dict], output_cap: int) -> None:
    """Emit the summary table and provenance lines suite.py parses."""
    print("\n" + "=" * 60)
    print("Summary (harness: aiperf)")
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
    for row in rows:
        if "error" in row:
            continue
        print(client.token_shape_line(row["concurrency"], row["shape"], output_cap,
                                      row["output_per_request"]))
        print(client.request_success_line(
            row["concurrency"], row["successful"], row["total"],
            row["transport_failures"], row["server_failures"]))
        underfilled = row["peak_inflight"] < row["concurrency"]
        print(f"CONCURRENCY_FILL requested={row['concurrency']} "
              f"achieved={row['achieved_concurrency']} "
              f"peak={row['peak_inflight']} "
              f"requests={row['total']} "
              f"underfilled={'yes' if underfilled else 'no'}")


def build_trace_argv(trace: Path, trace_type: str, artifacts: Path) -> list[str]:
    """One streaming fixed-schedule invocation replaying a committed trace."""
    endpoint_type = {"chat": "chat", "completions": "completions"}[
        os.environ.get("BENCH_ENDPOINT", "chat")]
    return [
        AIPERF_BIN, "profile",
        "--model-names", config.MODEL_ID,
        "--tokenizer", AIPERF_TOKENIZER,
        "--use-server-token-count",
        "--url", config.VLLM_API_URL.removesuffix("/v1"),
        "--endpoint-type", endpoint_type,
        "--input-file", str(trace),
        "--custom-dataset-type", trace_type,
        "--streaming",
        # Trace timestamps drive arrivals, so the flag is explicit rather than left to aiperf's auto-switch.
        "--fixed-schedule",
        "--workers-max", str(AIPERF_WORKERS_MAX),
        "--random-seed", str(config.PROMPT_SEED),
        "--output-artifact-dir", str(artifacts),
        "--ui", "simple",
    ]


def run_trace(trace: Path, trace_type: str, scratch: Path) -> dict:
    """Replay the trace once and reduce the streaming export to one summary row."""
    artifacts = scratch / "trace"
    argv = build_trace_argv(trace, trace_type, artifacts)
    print(f"AIPERF_ARGV level=1 {' '.join(argv)}")
    done = subprocess.run(argv, capture_output=True, text=True)
    export = artifacts / "profile_export_aiperf.json"
    if done.returncode != 0 or not export.exists():
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-8:]
        return {"error": "aiperf failed", "detail": "\n".join(tail)}
    data = json.loads(export.read_text())

    def stat(name: str, key: str, default: float = 0.0) -> float:
        entry = data.get(name) or {}
        value = entry.get(key)
        return float(value) if value is not None else default

    completed = int(stat("request_count", "avg"))
    failures = sum(int(e.get("count", 0)) for e in data.get("error_summary", []))
    return {
        "tokens_per_sec": stat("output_token_throughput", "avg"),
        "req_per_sec": stat("request_throughput", "avg"),
        "ttft_p50_s": stat("time_to_first_token", "p50") / 1000.0,
        "ttft_p99_s": stat("time_to_first_token", "p99") / 1000.0,
        "itl_p50_ms": stat("inter_token_latency", "p50"),
        "itl_p99_ms": stat("inter_token_latency", "p99"),
        "successful": completed,
        "total": completed + failures,
        "server_failures": failures,
        "aiperf_version": data.get("aiperf_version", "unknown"),
    }


def trace_main() -> int:
    """Replay a committed trace and emit the TRACE_ROW stdout contract."""
    trace_env = os.environ.get("TRACE_FILE", "")
    if not trace_env:
        sys.exit("trace mode requires TRACE_FILE (repo-relative path, e.g. "
                 "app/bench/data/traces/agentx_32k_seed42.jsonl). The agentx protocol "
                 "pins one; trace_replay takes it from --trace.")
    trace = (config.PROJECT_ROOT / trace_env).resolve()
    if not trace.exists():
        sys.exit(f"trace file not found: {trace}. Traces must live inside the repo "
                 f"(app/bench/data/traces/) so the launcher payload carries them.")
    trace_type = os.environ.get("TRACE_TYPE", "mooncake_trace")
    if shutil.which(AIPERF_BIN) is None:
        sys.exit(f"aiperf binary not found ({AIPERF_BIN!r}). "
                 f"pip install -r app/bench/requirements-aiperf.txt, or set AIPERF_BIN.")
    scratch = Path(tempfile.mkdtemp(prefix="aiperf-trace-"))
    print(f"aiperf trace replay: trace={trace_env} type={trace_type} "
          f"seed={config.PROMPT_SEED} tokenizer={AIPERF_TOKENIZER} (counting: server)")
    row = run_trace(trace, trace_type, scratch)
    if "error" in row:
        sys.exit(f"trace replay FAILED — {row['detail']}")
    print(f"AIPERF_VERSION {row['aiperf_version']}")
    print("TRACE_ROW level=1 "
          f"tokens_per_sec={row['tokens_per_sec']:.3f} "
          f"req_per_sec={row['req_per_sec']:.3f} "
          f"ttft_p50_s={row['ttft_p50_s']:.3f} "
          f"ttft_p99_s={row['ttft_p99_s']:.3f} "
          f"itl_p50_ms={row['itl_p50_ms']:.3f} "
          f"itl_p99_ms={row['itl_p99_ms']:.3f}")
    print(client.request_success_line(1, row["successful"], row["total"],
                                      0, row["server_failures"]))
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


def main() -> int:
    """Sweep the configured levels through aiperf and emit the stdout contract."""
    if os.environ.get("AIPERF_MODE") == "trace":
        return trace_main()
    profile_name = os.environ.get("PROMPT_PROFILE")
    if not profile_name:
        sys.exit("aiperf harness requires PROMPT_PROFILE: generated prompt modes are "
                 "not supported, so both clients always measure the same pinned workload.")
    if shutil.which(AIPERF_BIN) is None:
        sys.exit(f"aiperf binary not found ({AIPERF_BIN!r}). "
                 f"pip install -r app/bench/requirements-aiperf.txt, or set AIPERF_BIN.")
    levels = [int(x) for x in os.environ["CONCURRENCY_SWEEP"].split(",") if x.strip()]
    output_cap = config.THROUGHPUT_MAX_TOKENS
    scratch = Path(tempfile.mkdtemp(prefix="aiperf-"))
    per_level = {lvl: requests_for_level(config.THROUGHPUT_NUM_REQUESTS, lvl)
                 for lvl in levels}
    print(f"aiperf harness: profile={profile_name} levels={levels} "
          f"requests/level={per_level} output_cap={output_cap} "
          f"seed={config.PROMPT_SEED} tokenizer={AIPERF_TOKENIZER} (counting: server)")
    rows = []
    for level in levels:
        rows.append(run_level(level, profile_name, output_cap, scratch))
        row = rows[-1]
        if "error" in row:
            print(f"  level {level}: FAILED — {row['detail']}")
        else:
            print(f"  level {level}: {row['tokens_per_sec']:.1f} tok/s, "
                  f"{row['successful']}/{row['total']} ok, "
                  f"fill avg {row['achieved_concurrency']:.1f} peak {row['peak_inflight']}")
    if all("error" in row for row in rows):
        sys.exit("every aiperf level failed; nothing to report")
    versions = {row.get("aiperf_version") for row in rows if "error" not in row}
    print(f"AIPERF_VERSION {sorted(versions)[0]}")
    print_contract(rows, output_cap)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
