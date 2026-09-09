"""Regression checks for the open-loop MLPerf LoadGen harness, runnable without a GPU, each encoding a
defect that reached a real run and produced figures that could not be relied on. LoadGen certifies
a p99 only once the query count clears its early-stopping arithmetic, which is 459 queries with no
violations, so a floor set below that returned UNDERSAMPLED no matter how well the server behaved.
The harness had no warmup, so first-request costs landed in the certified percentiles and flipped a
level's SLA verdict. Most importantly, failed requests are completed to LoadGen, which has no
failed-query concept, so they enter its latency percentiles as instant responses and flatter them:
against a killed server with most requests failing, LoadGen still reported the SLA met and the
result valid, and with no success line emitted a dead rep was aggregated as successful. A circuit
breaker now opens after consecutive failures and the runner probes liveness per rep and per level.
Wire checks need mlcommons-loadgen and are skipped with a notice when it is absent, and they talk
only to the local mock."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from math import comb
from pathlib import Path

# Deterministic and laptop-sized, set before importing config, which freezes its environment.
CHECK_ENV = {
    "PROMPT_PROFILE": "coding_completion",
    "MAX_TOKENS": "64",
    "MODEL_ID": "mock-model",
    "INSTANCE_TYPE": "verify-local",
    "QUANTIZATION": "none",
    "BENCH_ENDPOINT": "chat",
    "TTFT_SLA_MS": "500",
    "TPOT_SLA_MS": "50",
    "MIN_DURATION_S": "1",
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


def queries_needed(over_sla_samples: int) -> int:
    """LoadGen's Server-scenario early-stopping requirement, reimplemented.

    Mirrors early_stopping.cc: the minimum h such that the probability of seeing at most
    t over-latency queries in h + t trials is under 1 - confidence, with p = 0.99 and
    confidence = 0.99. Reimplemented rather than imported because the C++ helper is not
    exposed through the Python bindings, and because a check that asks LoadGen to confirm
    its own rule would prove nothing about our configuration.
    """
    p, confidence = 0.99, 0.99
    h = 1
    while sum(comb(h + over_sla_samples, k) * (1 - p) ** k * p ** (h + over_sla_samples - k)
              for k in range(over_sla_samples + 1)) > 1 - confidence:
        h += 1
    return h + over_sla_samples


def free_port() -> int:
    """Ask the OS for an unused port, so a check never collides with a dev server."""
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
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1):
                return proc
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("mock server exited during startup")
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("mock server did not become ready within 15s")


def run_level(port: int, queries: int, target_qps: float, warmup: int = 0,
              abort_after: int = 50, outdir: str = "") -> dict:
    """Run one open-loop level against the mock and parse its two contract lines."""
    env = {**os.environ, **CHECK_ENV,
           "VLLM_API_URL": f"http://127.0.0.1:{port}/v1",
           "SERVER_TARGET_QPS": str(target_qps),
           "MIN_QUERY_COUNT": str(queries),
           "WARMUP_REQUESTS": str(warmup),
           "ABORT_AFTER_FAILURES": str(abort_after),
           "LOADGEN_OUTDIR": outdir or f"/tmp/verify-openloop-{queries}-{target_qps}"}
    done = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "main.py"), "arrival_rate"],
        env=env, capture_output=True, text=True, timeout=1800)
    out = {"stdout": done.stdout, "qps_row": {}, "success": {}, "aborted": False}
    for line in done.stdout.splitlines():
        if line.startswith("QPS_ROW "):
            out["qps_row"] = dict(token.split("=", 1)
                                 for token in line.split()[1:] if "=" in token)
        elif line.startswith("REQUEST_SUCCESS "):
            out["success"] = dict(token.split("=", 1)
                                 for token in line.split()[1:] if "=" in token)
        elif "ABORTING LEVEL" in line:
            out["aborted"] = True
    return out


def static_checks() -> None:
    """Configuration and arithmetic checks. No LoadGen, no network."""
    import config

    print("static checks:")
    # The arithmetic itself, against the figures read out of loadgen 6.0.17's source.
    check("early-stopping requirement is 459 queries at zero violations",
          queries_needed(0) == 459, f"got {queries_needed(0)}")
    check("early-stopping requirement is 662 queries at one violation",
          queries_needed(1) == 662, f"got {queries_needed(1)}")

    floor = int(config.PROTOCOLS["server_qps"]["env"]["MIN_QUERY_COUNT"])
    check(f"server_qps query floor ({floor}) can certify a clean level",
          floor >= queries_needed(0),
          f"{floor} < {queries_needed(0)}: no level can ever be certified")
    check("server_qps query floor tolerates a few over-SLA samples",
          floor >= queries_needed(2), f"{floor} < {queries_needed(2)}")
    check("server_qps configures warmup",
          int(config.PROTOCOLS["server_qps"]["env"].get("WARMUP_REQUESTS", 0)) > 0)
    check("server_qps configures an inter-level settle",
          config.PROTOCOLS["server_qps"].get("settle_s", 0) > 0)
    check("QSL pool exceeds the query floor (no sample repeats)",
          config.QSL_SIZE > floor, f"QSL_SIZE={config.QSL_SIZE} floor={floor}")

    # The runner must record open-loop success data, which is what makes the exclusion machinery apply.
    import inspect

    import suite
    source = inspect.getsource(suite.sweep_qps_levels)
    check("runner records REQUEST_SUCCESS for open-loop levels",
          "parse_success" in source and "rep.success" in source)
    check("runner probes server liveness per level", "server_alive" in source)
    check("runner probes server liveness per rep",
          "server_alive" in inspect.getsource(suite.main))
    check("a 0%-success level is flagged for exclusion",
          suite.degraded_levels({4000: {"ok": 0, "total": 1000, "ok_fraction": 0.0,
                                        "transport": 1000, "server": 0}}) == [4000])
    check("liveness probe reports False for a dead endpoint",
          suite.server_alive("http://127.0.0.1:9/v1", "anything") is False)


def wire_checks() -> None:
    """End-to-end checks driving the real harness against the mock server."""
    import suite

    print("wire checks (LoadGen + mock server):")

    # -- certification threshold, proved rather than assumed ---------------------
    port = free_port()
    mock = start_mock(port)
    try:
        below = run_level(port, queries=400, target_qps=25, warmup=5)
        above = run_level(port, queries=500, target_qps=25, warmup=5)
        check("400 queries cannot certify a p99 (below the 459 floor)",
              below["qps_row"].get("early_stopping") == "UNDERSAMPLED",
              str(below["qps_row"].get("early_stopping")))
        check("500 queries certifies a clean level (LoadGen VALID)",
              above["qps_row"].get("loadgen_result") == "VALID",
              str(above["qps_row"].get("loadgen_result")))

        # -- warmup isolation ---------------------------------------------------
        warmed = run_level(port, queries=500, target_qps=25, warmup=20)
        check("warmup requests stay out of the reported total",
              warmed["success"].get("total") == "500",
              f"total={warmed['success'].get('total')} with 20 warmup requests")
        check("REQUEST_SUCCESS is emitted at all (it was not before 2026-08-12)",
              bool(warmed["success"]))
        check("REQUEST_SUCCESS reports full success against a healthy server",
              warmed["success"].get("ok_fraction") == "1.0000",
              str(warmed["success"].get("ok_fraction")))
    finally:
        mock.kill()
        mock.wait()

    # -- server failures are attributed, not silently absorbed -------------------
    port = free_port()
    mock = start_mock(port, fail_every=3)   # every 3rd request returns HTTP 500
    try:
        broken = run_level(port, queries=300, target_qps=25, warmup=0)
        ok_fraction = float(broken["success"].get("ok_fraction", 1.0))
        level = int(broken["success"].get("concurrency", 0))
        check("injected HTTP 500s are counted as SERVER failures",
              int(broken["success"].get("server_failures", 0)) > 0,
              f"server_failures={broken['success'].get('server_failures')}")
        check("a level losing a third of its requests fails the success floor",
              ok_fraction < 0.95, f"ok_fraction={ok_fraction}")
        check("the runner would exclude that rep",
              suite.degraded_levels({level: {
                  "ok": int(broken["success"]["ok"]),
                  "total": int(broken["success"]["total"]),
                  "ok_fraction": ok_fraction,
                  "transport": int(broken["success"]["transport_failures"]),
                  "server": int(broken["success"]["server_failures"])}}) == [level])
    finally:
        mock.kill()
        mock.wait()

    # -- vanished server: breaker opens, and LoadGen's own verdict is not trusted -
    port = free_port()
    mock = start_mock(port)
    killer = subprocess.Popen(
        [sys.executable, "-c",
         f"import os,signal,time; time.sleep(12); os.kill({mock.pid}, signal.SIGTERM)"])
    try:
        dead = run_level(port, queries=600, target_qps=25, warmup=5, abort_after=50)
        check("circuit breaker opens when the server stops answering", dead["aborted"])
        ok_fraction = float(dead["success"].get("ok_fraction", 1.0))
        check("a vanished server drives ok_fraction below the floor",
              ok_fraction < 0.95, f"ok_fraction={ok_fraction}")
        # The point of the exercise: LoadGen can pass a dead server, whose failures complete as instant responses.
        loadgen_said = dead["qps_row"].get("sla")
        check("REQUEST_SUCCESS contradicts LoadGen when LoadGen is wrong",
              ok_fraction < 0.95,
              f"LoadGen said sla={loadgen_said}; only the success line catches this")
        print(f"        (for the record: LoadGen reported sla={loadgen_said}, "
              f"loadgen_result={dead['qps_row'].get('loadgen_result')} on a dead server)")
    finally:
        killer.kill()
        mock.kill()
        mock.wait()


def main() -> int:
    """Run all checks; exit nonzero if any FAILED. Missing LoadGen skips wire checks."""
    for key, value in CHECK_ENV.items():
        os.environ.setdefault(key, value)

    static_checks()

    try:
        import mlperf_loadgen  # noqa: F401
        has_loadgen = True
    except ImportError:
        has_loadgen = False

    if has_loadgen:
        wire_checks()
    else:
        print("wire checks: SKIPPED — mlcommons-loadgen not installed.\n"
              "  pip install -e '.[loadgen]' (needs a C++ toolchain)")

    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
