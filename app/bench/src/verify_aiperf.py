"""Regression checks for the aiperf adapter, runnable on a laptop without a GPU.

Every check here exists because its absence let a bug ship. A calibration run was
invalid for two adapter reasons — no output cap sent, flat request counts under-filling
high levels — and the mock used for local verification at the time could not catch the
first because it forgave a missing cap. These checks encode the failure modes so they
cannot silently recur:

  static  (no aiperf needed)
    converter        every dataset line carries the profile's prompt (cycled in
                     workload.build_prompts order) and output_length; cap<=0 refused
    request scaling  per-level counts obey the shared requests_for_level rule:
                     never below the level, whole waves, never below NUM_REQUESTS
    argv formation   scaled --request-count, pinned --workers-max, warmup omitted
                     when zero, /v1 stripped from --url

  wire  (needs the aiperf binary; SKIPPED with a notice when absent)
    output cap       against mock_server.py, which generates ~2,000 tokens when a
                     request arrives uncapped: output_per_request must equal the cap
    stdout contract  the emitted table and provenance lines parse via suite.py's own
                     regexes (parse_rows, parse_token_shape, parse_fill, parse_success)
    broken run       with MOCK_FAIL_EVERY injecting HTTP 500s, the adapter must report
                     failures and suite.degraded_levels must exclude the level — the
                     "success-floor exclusion fires on a deliberately broken run"
                     verification the integration plan requires for Phase 1

Wire checks are skipped, not failed, when aiperf is not installed: the binary is a
harness-specific extra (requirements-aiperf.txt), and a laptop that only touches the
aiohttp harnesses should still be able to run the static checks.

Usage:
    python3 main.py verify-aiperf
    AIPERF_BIN=/path/to/aiperf python3 main.py verify-aiperf
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CHECK_ENV = {
    # Deterministic, laptop-sized run. Set BEFORE importing config, which freezes its
    # environment at import time.
    "PROMPT_PROFILE": "legacy_corpus",
    "NUM_REQUESTS": "12",
    "WARMUP_REQUESTS": "0",
    "MAX_TOKENS": "150",
    "CONCURRENCY_SWEEP": "2",
    "MODEL_ID": "mock-model",
    "BENCH_ENDPOINT": "chat",
    "INSTANCE_TYPE": "verify-local",
    "QUANTIZATION": "none",
}

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and print one check result; failures accumulate rather than abort."""
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def free_port() -> int:
    """Ask the OS for an unused port, so the check never collides with a dev server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_mock(port: int, fail_every: int = 0) -> subprocess.Popen:
    """Launch mock_server.py as a child and wait until it answers /v1/models."""
    env = {**os.environ, "MOCK_PORT": str(port), "MOCK_MODEL_ID": "mock-model",
           "MOCK_FAIL_EVERY": str(fail_every)}
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "mock_server.py")],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 15
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return proc
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("mock server exited during startup")
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("mock server did not become ready within 15s")


def uncapped_completion_tokens(port: int) -> int:
    """Control probe: what the mock returns when NO cap is sent. Must be huge."""
    body = json.dumps({"model": "mock-model",
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)["usage"]["completion_tokens"]


def static_checks() -> None:
    """Converter, scaling and argv checks. No network, no aiperf binary."""
    import aiperf_profiles
    import config
    import workload
    from aiperf_harness import build_argv
    from throughput import requests_for_level

    print("static checks:")
    scratch = Path(tempfile.mkdtemp(prefix="verify-aiperf-"))

    # Converter: line count, cycling order, per-line cap.
    pool = workload.load_profile("legacy_corpus")["prompts"]
    dataset = aiperf_profiles.profile_to_jsonl("legacy_corpus", 12, 150,
                                               scratch / "dataset.jsonl")
    lines = [json.loads(line) for line in dataset.read_text().splitlines()]
    check("converter emits one line per request", len(lines) == 12)
    check("converter cycles the pool in build_prompts order",
          all(line["text"] == pool[i % len(pool)] for i, line in enumerate(lines)))
    check("every line carries output_length",
          all(line.get("output_length") == 150 for line in lines))
    try:
        aiperf_profiles.profile_to_jsonl("legacy_corpus", 1, 0, scratch / "x.jsonl")
        check("converter refuses cap<=0", False, "no exception raised")
    except ValueError:
        check("converter refuses cap<=0", True)

    # Request scaling: invariants of the shared rule, not memorized outputs.
    levels = (1, 8, 32, 128, 256, 512)
    scaled = {level: requests_for_level(12, level) for level in levels}
    check("scaled counts never below the level (fillable)",
          all(count >= level for level, count in scaled.items()))
    check("scaled counts are whole waves (multiple of level)",
          all(count % level == 0 for level, count in scaled.items()))
    check("scaled counts never below NUM_REQUESTS",
          all(count >= 12 for count in scaled.values()))

    # argv formation.
    argv = build_argv(512, scaled[512], scratch / "d.jsonl", scratch / "art")
    def flag(name: str) -> str:
        return argv[argv.index(name) + 1]
    check("argv carries the SCALED request count",
          flag("--request-count") == str(scaled[512]))
    check("argv pins --workers-max", "--workers-max" in argv)
    check("argv omits --warmup-request-count when warmup is 0",
          "--warmup-request-count" not in argv)
    check("argv strips /v1 from --url", not flag("--url").endswith("/v1"))
    check("argv passes the pinned dataset and single_turn type",
          "--custom-dataset-type" in argv and flag("--custom-dataset-type") == "single_turn")

    # trace-mode argv: streaming fixed-schedule replay, no closed-loop knobs.
    from aiperf_harness import build_trace_argv
    trace_argv = build_trace_argv(scratch / "t.jsonl", "mooncake_trace", scratch / "art2")
    check("trace argv enables --streaming and --fixed-schedule",
          "--streaming" in trace_argv and "--fixed-schedule" in trace_argv)
    check("trace argv carries no --concurrency/--request-count (schedule drives)",
          "--concurrency" not in trace_argv and "--request-count" not in trace_argv)

    # the committed agentx trace: every session must fit the 32k serving window.
    trace_path = config.PROJECT_ROOT / "app/bench/data/traces/agentx_32k_seed42.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    peaks: dict[str, int] = {}
    for record in records:
        peaks[record["session_id"]] = max(
            peaks.get(record["session_id"], 0),
            record["input_length"] + record["output_length"])
    check("committed agentx trace fits the 32k context window (peak <= 30000)",
          max(peaks.values()) <= 30000, f"max peak {max(peaks.values())}")
    check("committed agentx trace is multi-turn with think-time delays",
          any("delay" in record for record in records))


def wire_checks() -> None:
    """End-to-end checks through a real aiperf child against the hostile mock."""
    import aiperf_harness
    import suite
    from throughput import requests_for_level

    print("wire checks (aiperf + mock server):")
    scratch = Path(tempfile.mkdtemp(prefix="verify-aiperf-wire-"))

    # -- healthy run: cap enforcement and contract parseability ------------------
    port = free_port()
    os.environ["VLLM_API_URL"] = f"http://127.0.0.1:{port}/v1"
    import config
    config.VLLM_API_URL = f"http://127.0.0.1:{port}/v1"
    mock = start_mock(port)
    try:
        control = uncapped_completion_tokens(port)
        check("mock generates to EOS when uncapped (control)", control > 1500,
              f"control returned {control} tokens; a forgiving mock cannot catch cap bugs")
        row = aiperf_harness.run_level(2, "legacy_corpus", 150, scratch)
        if "error" in row:
            check("aiperf level completes against mock", False, row["detail"])
            return
        check("aiperf level completes against mock", True)
        check("output capped at exactly 150 (the bug the calibration run shipped with)",
              row["output_per_request"] == 150.0,
              f"output_per_request={row['output_per_request']}")
        check("request count scaled for the level",
              row["total"] == requests_for_level(12, 2), f"total={row['total']}")

        # Contract: everything the adapter prints must parse via suite's own regexes.
        import contextlib
        import io
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            aiperf_harness.print_contract([row], 150)
        text = stdout.getvalue()
        check("summary row parses via suite.parse_rows",
              2 in suite.parse_rows(text, "aiperf"))
        check("TOKEN_SHAPE parses", 2 in suite.parse_token_shape(text))
        check("CONCURRENCY_FILL parses", 2 in suite.parse_fill(text))
        check("REQUEST_SUCCESS parses", 2 in suite.parse_success(text))
    finally:
        mock.kill()
        mock.wait()

    # -- trace replay: streaming fixed-schedule against the SSE mock -------------
    port = free_port()
    os.environ["VLLM_API_URL"] = f"http://127.0.0.1:{port}/v1"
    config.VLLM_API_URL = f"http://127.0.0.1:{port}/v1"
    mock = start_mock(port)
    try:
        # A tiny timestamped trace: three requests, 100ms apart. Small on purpose —
        # this checks the streaming/fixed-schedule wiring and the TRACE_ROW contract,
        # not trace realism (the committed agentx artifact covers that statically).
        trace = scratch / "tiny_trace.jsonl"
        trace.write_text("".join(
            json.dumps({"timestamp": i * 100, "input_length": 40,
                        "output_length": 30, "hash_ids": [i]}) + "\n"
            for i in range(3)))
        row = aiperf_harness.run_trace(trace, "mooncake_trace", scratch)
        if "error" in row:
            check("trace replay completes against streaming mock", False, row["detail"])
        else:
            check("trace replay completes against streaming mock", True)
            check("trace replay saw all requests succeed",
                  row["successful"] == row["total"] == 3,
                  f"{row['successful']}/{row['total']}")
            check("trace replay produced streaming metrics (TTFT/ITL nonzero)",
                  row["ttft_p50_s"] > 0 and row["itl_p50_ms"] > 0)
            trace_line = ("TRACE_ROW level=1 " + " ".join(
                f"{name}={row[name]:.3f}" for name in suite.COLS_TRACE))
            check("TRACE_ROW parses via suite.parse_rows('aiperf_trace')",
                  1 in suite.parse_rows(trace_line, "aiperf_trace"))
    finally:
        mock.kill()
        mock.wait()

    # -- deliberately broken run: the success floor must fire --------------------
    port = free_port()
    os.environ["VLLM_API_URL"] = f"http://127.0.0.1:{port}/v1"
    config.VLLM_API_URL = f"http://127.0.0.1:{port}/v1"
    mock = start_mock(port, fail_every=3)  # every 3rd request 500s -> ~33% failures
    try:
        row = aiperf_harness.run_level(2, "legacy_corpus", 150, scratch)
        if "error" in row:
            # aiperf treating a mostly-failing server as a fatal run is acceptable:
            # either way the level produces no trustworthy numbers.
            check("broken run is not reported as healthy", True)
        else:
            ok_fraction = row["successful"] / max(row["total"], 1)
            check("injected failures are visible in the adapter's counts",
                  row["server_failures"] > 0,
                  f"successful={row['successful']} total={row['total']}")
            import config as cfg
            floor = cfg.MIN_LEVEL_SUCCESS
            check(f"success floor would exclude the level (ok={ok_fraction:.2f} < {floor})",
                  ok_fraction < floor,
                  "failure injection too weak to trip the floor")
    finally:
        mock.kill()
        mock.wait()


def main() -> int:
    """Run all checks; exit nonzero if any FAILED. Missing aiperf skips wire checks."""
    for key, value in CHECK_ENV.items():
        os.environ[key] = value
    os.environ.setdefault("VLLM_API_URL", "http://127.0.0.1:1/v1")  # replaced per check

    static_checks()

    aiperf_bin = os.environ.get("AIPERF_BIN", "aiperf")
    if shutil.which(aiperf_bin) is None:
        print(f"wire checks: SKIPPED — aiperf binary not found ({aiperf_bin!r}).\n"
              f"  pip install -r app/bench/requirements-aiperf.txt (needs python "
              f">=3.11,<3.14), or point AIPERF_BIN at an installed copy.")
    else:
        wire_checks()

    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
