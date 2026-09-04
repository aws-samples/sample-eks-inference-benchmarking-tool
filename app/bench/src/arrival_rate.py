"""Open-loop arrival-rate harness: does the server hold its SLA at a target requests/sec.

Arrivals follow a Poisson process at a fixed rate, independent of completions, which is a
structurally different question from the closed-loop harnesses: they pin in-flight depth at
N and ask what N concurrent users each experience, this asks whether a given arrival rate is
within capacity. The two are not comparable and must never share a table, which is why every
figure here is emitted with an `openloop_` prefix or a QPS_ROW key. MLPerf LoadGen does the
percentile accounting and models arrival jitter itself, rather than us hand-rolling a Poisson
driver and nearest-rank percentiles. Needs requirements-loadgen.txt, kept separate so the
closed-loop harnesses stay aiohttp-only.
"""
from __future__ import annotations

import array
import asyncio
import os
import sys
import threading
import time

import aiohttp

import config
import client
from stats import median
from workload import build_prompts, describe_workload

try:
    import mlperf_loadgen as lg
except ImportError:
    sys.exit("mlperf_loadgen not installed.\n"
             "  pip install -r app/bench/requirements-loadgen.txt\n"
             "It builds from source (no wheels published); a C++ toolchain is required.")

TITLE = "MLPerf LoadGen — OPEN LOOP (Server scenario, Poisson arrivals)"

QSL_SIZE = config.QSL_SIZE
LOOPBACK_HOSTS = config.LOOPBACK_HOSTS


# ---------------------------------------------------------------------------- #
#                               BACKGROUND LOOP                                #
# ---------------------------------------------------------------------------- #

class BackgroundLoop:
    """An asyncio loop on its own thread, because LoadGen issues queries from a C++ thread.

    LoadGen calls issue_queries() from a thread it owns and expects a prompt return. Blocking
    there would distort the very arrival process it is modelling, so requests are handed to
    this loop and completed asynchronously.
    """

    def __init__(self) -> None:
        """Start the loop thread and open the shared session before returning."""
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._serve, daemon=True, name="sut-loop")
        self.thread.start()
        self.session: aiohttp.ClientSession | None = None
        asyncio.run_coroutine_threadsafe(self._open_session(), self.loop).result(timeout=30)

    def _serve(self) -> None:
        """Thread body: own the loop and run it until stopped."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _open_session(self) -> None:
        """Open an unpooled session so the client never becomes the queue.

        limit=0 is unbounded: an open-loop driver whose connection pool throttles arrivals has
        silently become a closed-loop test. force_close=True is also deliberate — each Poisson
        arrival is an independent request, and reusing sockets is an artifact of the harness.
        It also removes a failure class: at low arrival rates a pooled connection sits idle
        long enough to be reaped by a port-forward or keep-alive, and the next request on it
        fails. Measured on a real server: 13 failures of 200 at 1 QPS, 2 at 2 QPS, none at 4
        and above — an artifact appearing only where load is lightest, landing squarely in the
        latency percentiles.
        """
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, force_close=True,
                                           enable_cleanup_closed=True),
            timeout=aiohttp.ClientTimeout(total=None),
        )

    def submit(self, coro):
        """Schedule a coroutine on the loop thread from LoadGen's thread."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self) -> None:
        """Close the session, stop the loop, and join the thread."""
        asyncio.run_coroutine_threadsafe(self._close_session(), self.loop).result(timeout=30)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)

    async def _close_session(self) -> None:
        """Release the shared session on the loop that created it."""
        if self.session:
            await self.session.close()


# ---------------------------------------------------------------------------- #
#                              SYSTEM UNDER TEST                               #
# ---------------------------------------------------------------------------- #

class SystemUnderTest:
    """Bridges LoadGen's callbacks to streaming HTTP requests against the served model."""

    def __init__(self, prompts: list[str]) -> None:
        """Hold the sample pool and the counters reported at the end of the run."""
        self.prompts = prompts
        self.runtime = BackgroundLoop()
        self.completed = 0
        self.failures = 0
        # Failures by exception type. A bare count cannot distinguish a client-side transport
        # drop from a server rejection, and those have opposite meanings for a capacity
        # result: one invalidates the run, the other IS the result.
        self.failure_kinds: dict[str, int] = {}
        self.transport_failures = 0
        self.server_failures = 0
        # Circuit breaker. LoadGen owns the test loop and cannot be stopped early, so once the
        # server is clearly gone the SUT stops attempting requests and fast-completes the rest.
        # Without it, the 2026-08-12 rep whose server was deleted mid-run kept issuing against
        # nothing for seven levels — 50 minutes producing a rep that was invalid from its first
        # failure. Fast-failing keeps the level's verdict identical (its requests are still
        # counted as failures, so the runner excludes it) while ending it in seconds.
        self.consecutive_failures = 0
        self.aborted = False
        # Server-counted input sizes, so a capacity figure carries the shape that produced it.
        self.input_token_counts: list[int] = []
        self._lock = threading.Lock()
        # Response buffers must outlive QuerySamplesComplete. One byte is enough, because
        # PerformanceOnly mode never inspects the payload.
        self._buffer = array.array("B", b"\x00")
        self._buffer_info = self._buffer.buffer_info()

    async def warm_request(self, prompt: str) -> None:
        """Issue one request outside LoadGen, for warmup only.

        Goes through the same session and the same streaming path as a measured arrival —
        a warmup that exercised a different code path would not pay the same one-time
        costs — but reports nothing to LoadGen, which is not running yet.
        """
        url, payload = client.build_request(prompt, config.ARRIVAL_MAX_TOKENS, stream=True)
        await self._stream(url, payload, query_id=None)

    def reset_counters(self) -> None:
        """Clear counters after warmup so no warmup request can enter a reported figure."""
        with self._lock:
            self.completed = 0
            self.failures = 0
            self.failure_kinds = {}
            self.transport_failures = 0
            self.server_failures = 0
            self.input_token_counts = []

    def issue_queries(self, query_samples) -> None:
        """LoadGen callback: hand each arrival to the background loop without blocking."""
        for sample in query_samples:
            prompt = self.prompts[sample.index % len(self.prompts)]
            self.runtime.submit(self._handle(sample.id, prompt))

    def flush_queries(self) -> None:
        """LoadGen callback: nothing is buffered, as Server scenario completes each query."""

    def shutdown(self) -> None:
        """Tear down the background loop once the test is over."""
        self.runtime.shutdown()

    async def _handle(self, query_id: int, prompt: str) -> None:
        """Stream one arrival, reporting first-token and completion timings to LoadGen."""
        if self.aborted:
            # Counted as a failure, not skipped: a request never sent is not a measurement,
            # and the level must fail the runner's success floor rather than look partial.
            self._record_failure("aborted: circuit breaker open", client.TRANSPORT)
            self._complete(query_id, first_token_sent=False, tokens=1)
            return
        url, payload = client.build_request(prompt, config.ARRIVAL_MAX_TOKENS, stream=True)
        outcome = await self._stream(url, payload, query_id)
        if outcome.get("failed"):
            self._record_failure(outcome["error"], outcome.get("kind", client.TRANSPORT))
            # A query that is never completed hangs LoadGen for the whole run, so failures
            # must still be completed. LoadGen has no failed-query concept, so these land in
            # its latency percentiles as if real: any level with failures > 0 has percentiles
            # that must not be quoted.
            self._complete(query_id, outcome["first_token_sent"], 1)
            return
        self._complete(query_id, outcome["first_token_sent"],
                       outcome["output_tokens"] or max(outcome["pieces"], 1))
        with self._lock:
            self.consecutive_failures = 0
            self.completed += 1
            if outcome["input_tokens"]:
                self.input_token_counts.append(outcome["input_tokens"])

    async def _stream(self, url: str, payload: dict, query_id: int | None) -> dict:
        """Consume the SSE stream, signalling FirstTokenComplete on the first generated piece."""
        state = {"pieces": 0, "output_tokens": None, "input_tokens": None,
                 "first_token_sent": False, "failed": False, "error": "",
                 "kind": ""}
        try:
            async with self.runtime.session.post(url, json=payload) as response:
                response.raise_for_status()
                async for chunk in client.iter_sse_chunks(response):
                    output_tokens, input_tokens = client.usage_counts(chunk)
                    if output_tokens is not None:
                        state["output_tokens"] = output_tokens
                    if input_tokens is not None:
                        state["input_tokens"] = input_tokens
                    if not client.token_piece(chunk):
                        continue
                    state["pieces"] += 1
                    if not state["first_token_sent"]:
                        # This is what makes TTFT a measured quantity rather than something
                        # inferred from end-to-end latency.
                        self._send_first_token(query_id)
                        state["first_token_sent"] = True
        except Exception as exc:
            detail = str(exc)[:120]
            state["failed"] = True
            state["error"] = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            # Split transport from server, the same way the closed-loop harnesses do: a
            # dropped connection invalidates the sample, a server rejection IS the result.
            state["kind"] = client.classify_failure(exc)
        return state

    def _send_first_token(self, query_id: int | None) -> None:
        """Report the first generated token for one query to LoadGen.

        query_id is None for warmup requests, which run before the test starts: reporting
        them would either crash LoadGen or, worse, enter its latency record.
        """
        if query_id is None:
            return
        lg.FirstTokenComplete([lg.QuerySampleResponse(
            query_id, self._buffer_info[0], self._buffer_info[1])])

    def _complete(self, query_id: int, first_token_sent: bool, tokens: int) -> None:
        """Close out one query, back-filling a first-token signal if none was ever sent."""
        if not first_token_sent:
            self._send_first_token(query_id)
        lg.QuerySamplesComplete([lg.QuerySampleResponse(
            query_id, self._buffer_info[0], self._buffer_info[1], tokens)])

    def _record_failure(self, error: str, kind: str = client.TRANSPORT) -> None:
        """Count one failure against its exception type and its transport/server class."""
        trip = False
        with self._lock:
            self.failures += 1
            self.failure_kinds[error] = self.failure_kinds.get(error, 0) + 1
            if kind == client.SERVER:
                self.server_failures += 1
            else:
                self.transport_failures += 1
            if not self.aborted:
                self.consecutive_failures += 1
                if self.consecutive_failures >= config.ARRIVAL_ABORT_AFTER_FAILURES:
                    self.aborted = trip = True
        if trip:
            print(f"\n  ABORTING LEVEL: {config.ARRIVAL_ABORT_AFTER_FAILURES} consecutive "
                  f"failures — the server is not answering. Remaining queries are failed "
                  f"without being sent; this level cannot produce a measurement.",
                  flush=True)


# ---------------------------------------------------------------------------- #
#                               LOADGEN SETTINGS                               #
# ---------------------------------------------------------------------------- #

def test_settings() -> "lg.TestSettings":
    """Build LoadGen's test settings from the configured arrival rate and SLAs."""
    settings = lg.TestSettings()
    settings.scenario = lg.TestScenario.Server
    # PerformanceOnly, always. A FindPeakPerformance toggle existed from 2026-08-03 to
    # 2026-08-12 but no protocol set it and no published result used it; removed rather
    # than left as an untested code path (deleted alongside the FIND_PEAK_QPS flag).
    settings.mode = lg.TestMode.PerformanceOnly
    settings.server_target_qps = config.SERVER_TARGET_QPS
    settings.min_query_count = config.MIN_QUERY_COUNT
    settings.min_duration_ms = int(config.ARRIVAL_MIN_DURATION_S * 1000)
    settings.server_coalesce_queries = False  # one query per arrival; batching is the server's job
    # Without use_token_latencies LoadGen reports only end-to-end latency, which for a
    # streaming response conflates prefill and decode into one uninterpretable number.
    settings.use_token_latencies = True
    settings.ttft_latency = int(config.TTFT_SLA_MS * 1_000_000)
    settings.tpot_latency = int(config.TPOT_SLA_MS * 1_000_000)
    return settings


def log_settings() -> "lg.LogSettings":
    """Point LoadGen's own logs at the configured output directory."""
    os.makedirs(config.LOADGEN_OUTDIR, exist_ok=True)
    settings = lg.LogSettings()
    settings.log_output.outdir = config.LOADGEN_OUTDIR
    settings.log_output.copy_summary_to_stdout = True
    settings.enable_trace = False
    return settings


def read_summary(path: str) -> dict[str, str]:
    """Parse LoadGen's key: value summary file, returning {} when it was never written."""
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        return {}
    return {k.strip(): v.strip() for k, _, v in
            (line.partition(":") for line in lines) if v}


def summary_ms(summary: dict, key: str) -> float:
    """Read a nanosecond latency field from the summary and convert it to milliseconds."""
    return summary_number(summary, key) / 1_000_000


def summary_number(summary: dict, key: str) -> float:
    """Read a numeric field from the summary, treating anything unparsable as 0."""
    raw = summary.get(key)
    try:
        return float(raw) if raw else 0.0
    except ValueError:
        return 0.0


def verdict(raw: str, yes: str, no: str) -> str:
    """Map one of LoadGen's yes/no verdicts onto our own label, or UNKNOWN."""
    normalised = raw.strip().lower()
    if normalised.startswith("yes"):
        return yes
    return no if normalised.startswith("no") else "UNKNOWN"


# ---------------------------------------------------------------------------- #
#                                  REPORTING                                   #
# ---------------------------------------------------------------------------- #

def print_run_header() -> None:
    """Print the provenance block, warning if the endpoint is behind a local tunnel."""
    mode = "PerformanceOnly"
    notes = ["NOTE: open-loop results are NOT comparable to the concurrency sweep."]
    if any(host in config.VLLM_API_URL for host in LOOPBACK_HOSTS):
        # Measured through a kubectl port-forward: first byte 276 ms on a fresh connection
        # against 90 ms on a warm one. Because each arrival opens its own connection, that
        # ~185 ms of tunnel setup lands in every TTFT sample.
        notes += ["WARNING: endpoint is loopback, which usually means a port-forward or SSH",
                  "         tunnel. Connection setup through such a tunnel adds roughly",
                  "         150-200 ms to EVERY TTFT sample. TPOT and QPS are unaffected.",
                  "         For a publishable TTFT figure, run this harness IN-CLUSTER."]
    client.print_header(TITLE, {
        "Endpoint": config.VLLM_API_URL,
        "Model": config.MODEL_ID,
        "Mode": mode,
        "Target QPS": config.SERVER_TARGET_QPS,
        "SLA": f"TTFT p99 <= {config.TTFT_SLA_MS:.0f} ms, "
               f"TPOT p99 <= {config.TPOT_SLA_MS:.0f} ms",
        "Query floor": f"{config.MIN_QUERY_COUNT}, duration floor "
                       f"{config.ARRIVAL_MIN_DURATION_S:.0f}s",
        "Max tokens": config.ARRIVAL_MAX_TOKENS,
        "Prompts": describe_workload(),
    }, notes=tuple(notes))


def print_results(sut: SystemUnderTest, summary: dict, wall_seconds: float) -> None:
    """Print the QPS_ROW the protocol runner parses, then the human-readable caveats."""
    # Three distinct verdicts, and conflating them is the easy mistake. The SLA verdict is
    # the capacity answer. Early stopping is whether there were enough samples to certify the
    # percentile — a short run can hold the SLA and still fail it. LoadGen's overall result is
    # INVALID if either fails, so quoting it alone reports an under-sampled run as an SLA
    # violation, which is both wrong and alarming.
    sla = verdict(summary.get("Performance constraints satisfied", ""), "MET", "NOT_MET")
    early = verdict(summary.get("Early stopping satisfied", ""), "OK", "UNDERSAMPLED")
    overall = summary.get("Result is", "UNKNOWN").replace(" ", "_") or "UNKNOWN"
    achieved = (summary_number(summary, "Completed samples per second")
                or summary_number(summary, "Scheduled samples per second"))

    print()
    print("=" * 72)
    print("Summary (open loop — do NOT compare against the concurrency sweep)")
    # key=value rather than an aligned table, deliberately: an open-loop row must not be able
    # to match the runner's closed-loop table regexes and land in a concurrency-keyed
    # aggregate. openloop_tokens_per_sec is named so it cannot be quoted next to the sweep's
    # tokens/sec without the difference being visible in the column name itself.
    print(f"QPS_ROW target_qps={config.SERVER_TARGET_QPS:.3f} achieved_qps={achieved:.3f} "
          f"ttft_p50_ms={summary_ms(summary, '50.00 percentile first token latency (ns)'):.1f} "
          f"ttft_p99_ms={summary_ms(summary, '99.00 percentile first token latency (ns)'):.1f} "
          f"tpot_p50_ms={summary_ms(summary, '50.00 percentile time to output token (ns)'):.2f} "
          f"tpot_p99_ms={summary_ms(summary, '99.00 percentile time to output token (ns)'):.2f} "
          f"openloop_tokens_per_sec={summary_number(summary, 'Completed tokens per second'):.1f} "
          f"input_tokens_median={median(sut.input_token_counts)} "
          f"completed={sut.completed} failed={sut.failures} "
          f"sla={sla} early_stopping={early} loadgen_result={overall}")
    # The same REQUEST_SUCCESS line the closed-loop harnesses emit, so the runner's
    # success-floor exclusion covers open-loop reps too. It did not, and on 2026-08-12 a rep
    # whose server was deleted mid-run — 100% of requests failing at 7 of 8 levels — was
    # still aggregated as "3/3 successful reps", because the failure counts lived only inside
    # QPS_ROW where nothing enforced a floor. Failures are also completed to LoadGen (it has
    # no failed-query concept), so they enter its percentiles as if real: a level with
    # failures has percentiles that must not be quoted, and this line is what makes the
    # runner drop the rep instead of trusting a median to hide it.
    # `concurrency` carries the level key in the runner's units (milli-QPS), matching how
    # every open-loop level is keyed; the field name is fixed by the shared emitter.
    print(client.request_success_line(
        int(round(config.SERVER_TARGET_QPS * 1000)),
        sut.completed, sut.completed + sut.failures,
        sut.transport_failures, sut.server_failures))
    print(f"  wall clock: {wall_seconds:.1f}s   LoadGen logs: {config.LOADGEN_OUTDIR}/")
    # Measured against target, rather than a MET/NOT_MET word. The verdict was ours, not
    # LoadGen's, and it restated the two percentiles printed on the line above it while reading
    # as a failed run. A sweep is meant to walk past the point where latency breaks.
    ttft_p99 = summary_ms(summary, "99.00 percentile first token latency (ns)")
    tpot_p99 = summary_ms(summary, "99.00 percentile time to output token (ns)")
    for name, value, target, unit in (("TTFT p99", ttft_p99, config.TTFT_SLA_MS, ".0f"),
                                      ("TPOT p99", tpot_p99, config.TPOT_SLA_MS, ".2f")):
        mark = "over target" if value > target else "within target"
        # :g on the target, not :.0f, so a fractional target is not reported as its floor. A
        # 0.2 ms target printed as "0 ms", which reads as no target at all.
        print(f"  {name}: {value:{unit}} ms against a {target:g} ms target, {mark}")
    if early == "UNDERSAMPLED":
        print("  NOTE: LoadGen could not certify the p99 from this many samples. That is a "
              "sample-size statement, not a latency result, and it is expected at any rate "
              "past the targets, where nearly every request contributes a breach.")
        print("        Raise MIN_QUERY_COUNT or MIN_DURATION_S only if this rate is inside "
              "the targets and you intend to quote its percentile.")
    print_failures(sut)
    print("=" * 72)


def print_failures(sut: SystemUnderTest) -> None:
    """Print the failure breakdown, and why it invalidates this level's percentiles."""
    if not sut.failures:
        return
    share = 100.0 * sut.failures / max(sut.failures + sut.completed, 1)
    print(f"  WARNING: {sut.failures} request(s) failed ({share:.1f}% of issued).")
    print("           LoadGen has no failed-query concept, so these were completed and")
    print("           ARE INCLUDED in the latency percentiles above. Do not quote the")
    print("           percentiles for this level until the failures are explained.")
    for kind, count in sorted(sut.failure_kinds.items(), key=lambda item: -item[1]):
        print(f"           {count:>6}x  {kind}")


# ---------------------------------------------------------------------------- #
#                                 ENTRY POINT                                  #
# ---------------------------------------------------------------------------- #

def warm_up(sut: "SystemUnderTest", count: int) -> None:
    """Issue `count` requests before LoadGen starts, outside the measurement.

    Without this, the first arrivals of a run carry one-time costs — first CUDA graph
    capture, cold caches on the server, and this client's first TLS/TCP handshakes — and
    they land in the percentiles LoadGen certifies. Measured on GLM-5.2/B300 on
    2026-08-12: p99 TTFT was 6.0 s in rep 1 at 1 QPS against 378 ms in reps 2 and 3, a
    16x artifact that flipped the SLA verdict for that level. A closed-loop run absorbs
    this in its own warmup; the open-loop harness had none.

    Deliberately sequential and untimed: warmup only needs the one-time costs paid, and
    issuing it at the target arrival rate would just be a second, unreported test. The
    counters it moves are reset afterwards so warmup can never enter a reported figure.
    """
    if count <= 0:
        return
    print(f"  warmup: {count} requests (not measured) ...", end="", flush=True)
    futures = [sut.runtime.submit(sut.warm_request(sut.prompts[i % len(sut.prompts)]))
               for i in range(count)]
    for future in futures:
        future.result(timeout=600)
    sut.reset_counters()
    print(" done")


def main() -> int:
    """Run one open-loop test at the configured arrival rate and report the outcome."""
    print_run_header()
    sut = SystemUnderTest(build_prompts(QSL_SIZE))
    warm_up(sut, config.ARRIVAL_WARMUP_REQUESTS)
    handle = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    qsl = lg.ConstructQSL(QSL_SIZE, QSL_SIZE, lambda _s: None, lambda _s: None)

    started = time.perf_counter()
    try:
        lg.StartTestWithLogSettings(handle, qsl, test_settings(), log_settings())
    finally:
        lg.DestroyQSL(qsl)
        lg.DestroySUT(handle)
        sut.shutdown()
    wall_seconds = time.perf_counter() - started

    summary = read_summary(os.path.join(config.LOADGEN_OUTDIR, "mlperf_log_summary.txt"))
    print_results(sut, summary, wall_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
