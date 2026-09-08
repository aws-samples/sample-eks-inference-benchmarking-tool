#!/usr/bin/env python3
"""Protocol-locked benchmark runner that picks one named protocol, runs reps and aggregates medians,
so a measurement cannot be assembled by hand mid-session and later turn out comparable to nothing.
Protocols are data in config.PROTOCOLS and chosen by name, so half of one cannot be set; every
output file carries a PROTOCOL LOCK header recording the environment, harness revision and
endpoint, so a file cannot be silently mis-attributed; and reps are first class, with the aggregate
reporting the median plus spread and flagging any cell whose run-to-run variance is too high to
trust. Run with --dry-run first."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import config
import verify_profiles
from stats import git_revision, shell_output, shell_result, spread_fraction, utc_stamp

# Client CPU fraction above which a level may be measuring the harness, not the server.
CLIENT_CPU_SUSPECT = config.CLIENT_CPU_SUSPECT

# Fraction of the median that a cell's peak-to-peak spread may reach before we distrust it.
MAX_SPREAD = config.MAX_SPREAD

# Sizing for the companion latency pass, matched to the standalone protocol so figures agree.
LATENCY_MIN_SAMPLES = 100
LATENCY_WAVES = 2


# ---------------------------------------------------------------------------- #
#                             OUTPUT PARSING                                   #
# ---------------------------------------------------------------------------- #

# Row parsers keyed to each harness's table, so a format change stops every one of them parsing.
ROW_THROUGHPUT = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)s\s+([\d.]+)s\s*$")
ROW_LATENCY = re.compile(
    r"^\s*(\d+)\s+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)ms\s*$"
)
# key=value on purpose, so an open-loop row cannot match the closed-loop patterns above.
ROW_QPS = re.compile(r"^QPS_ROW\s+(.+)$")
# Trace-replay row, same discipline: one replay is one row and must not look like a sweep level.
ROW_TRACE = re.compile(r"^TRACE_ROW\s+(.+)$")
# Parsed apart from the tables because a missing input size let an output cap be read as one.
ROW_SHAPE = re.compile(r"^TOKEN_SHAPE\s+(.+)$")
# Parsed apart for the same reason: an underfilled level reports a depth it never reached.
ROW_FILL = re.compile(r"^CONCURRENCY_FILL\s+(.+)$")
# Parsed so a rep that lost requests is excluded rather than averaged over its survivors.
ROW_SUCCESS = re.compile(r"^REQUEST_SUCCESS\s+(.+)$")

COLS_THROUGHPUT = ["tokens_per_sec", "req_per_sec", "p50_s", "p99_s"]
COLS_LATENCY = ["ttft_p50_s", "ttft_p99_s", "tpot_p50_ms", "tpot_p99_ms",
                "itl_p50_ms", "itl_p99_ms"]
COLS_QPS = ["achieved_qps", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms",
            "openloop_tokens_per_sec", "input_tokens_median", "completed", "failed"]

# Trace streaming metrics are aiperf's ITL definitions, never interchangeable with latency.py's.
COLS_TRACE = ["tokens_per_sec", "req_per_sec", "ttft_p50_s", "ttft_p99_s",
              "itl_p50_ms", "itl_p99_ms"]
COLS = {"throughput": COLS_THROUGHPUT, "latency": COLS_LATENCY, "arrival_rate": COLS_QPS,
        "aiperf": COLS_THROUGHPUT, "aiperf_trace": COLS_TRACE}

# Row-key label; keys are ints either way, milli-QPS open loop, so median machinery is shared.
LEVEL_LABEL = {"throughput": "conc", "latency": "conc", "arrival_rate": "tgt_qps",
               "aiperf": "conc", "aiperf_trace": "run"}


def fmt_level(harness: str, level: int) -> str:
    """Render a level key for display, converting milli-QPS back to QPS."""
    return f"{level / 1000:.3f}" if harness == "arrival_rate" else str(level)


def split_kv(text: str) -> dict[str, str]:
    """Split a run of key=value tokens into a dict, ignoring anything without an '='."""
    return dict(token.split("=", 1) for token in text.split() if "=" in token)


def to_float(value: str) -> float:
    """Parse a float, treating anything unparsable as 0 so one bad field cannot abort a run."""
    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_token_shape(text: str) -> dict[int, dict]:
    """Extract {level: {input_median, input_min, input_max, output_cap}} from harness stdout."""
    shapes: dict[int, dict] = {}
    for line in text.splitlines():
        match = ROW_SHAPE.match(line.strip())
        if not match:
            continue
        fields = split_kv(match.group(1))
        level = int(to_float(fields.get("concurrency", "-1")))
        shapes[level] = {k: int(to_float(v)) for k, v in fields.items() if k != "concurrency"}
    return shapes


def parse_fill(text: str) -> dict[int, dict]:
    """Extract {level: {achieved, peak, requests, underfilled}} from harness stdout."""
    fills: dict[int, dict] = {}
    for line in text.splitlines():
        match = ROW_FILL.match(line.strip())
        if not match:
            continue
        fields = split_kv(match.group(1))
        level = int(to_float(fields.get("requested", "-1")))
        fills[level] = {
            "achieved": to_float(fields.get("achieved", "0")),
            "peak": int(to_float(fields.get("peak", "0"))),
            "requests": int(to_float(fields.get("requests", "0"))),
            "underfilled": fields.get("underfilled", "no") == "yes",
        }
    return fills


def parse_success(text: str) -> dict[int, dict]:
    """Extract {level: {ok, total, ok_fraction, transport, server}} from harness stdout."""
    success: dict[int, dict] = {}
    for line in text.splitlines():
        match = ROW_SUCCESS.match(line.strip())
        if not match:
            continue
        fields = split_kv(match.group(1))
        level = int(to_float(fields.get("concurrency", "-1")))
        success[level] = {
            "ok": int(to_float(fields.get("ok", "0"))),
            "total": int(to_float(fields.get("total", "0"))),
            "ok_fraction": to_float(fields.get("ok_fraction", "0")),
            "transport": int(to_float(fields.get("transport_failures", "0"))),
            "server": int(to_float(fields.get("server_failures", "0"))),
        }
    return success


def degraded_levels(success: dict[int, dict]) -> list[int]:
    """List levels that fell below the success floor."""
    return sorted(level for level, entry in success.items()
                  if entry["ok_fraction"] < config.MIN_LEVEL_SUCCESS)


def parse_qps_rows(text: str) -> dict[int, dict]:
    """Extract the open-loop rows, keyed by target arrival rate in milli-QPS."""
    rows: dict[int, dict] = {}
    for line in text.splitlines():
        match = ROW_QPS.match(line.strip())
        if not match:
            continue
        fields = split_kv(match.group(1))
        if "target_qps" not in fields:
            continue
        row: dict = {name: to_float(fields.get(name, "0")) for name in COLS_QPS}
        # Categorical verdicts carried as text: a run can hold the SLA and still be invalid on samples.
        row["_sla"] = fields.get("sla", "UNKNOWN")
        row["_early"] = fields.get("early_stopping", "UNKNOWN")
        row["_loadgen_result"] = fields.get("loadgen_result", "UNKNOWN")
        rows[int(round(to_float(fields["target_qps"]) * 1000))] = row
    return rows


def parse_trace_rows(text: str) -> dict[int, dict]:
    """Extract the single trace-replay row, keyed by its pseudo-level."""
    rows: dict[int, dict] = {}
    for line in text.splitlines():
        match = ROW_TRACE.match(line.strip())
        if not match:
            continue
        fields = split_kv(match.group(1))
        rows[int(to_float(fields.get("level", "1")))] = {
            name: to_float(fields.get(name, "0")) for name in COLS_TRACE}
    return rows


def parse_rows(text: str, harness: str) -> dict[int, dict]:
    """Extract the summary table from harness stdout, keyed by level."""
    if harness == "arrival_rate":
        return parse_qps_rows(text)
    if harness == "aiperf_trace":
        return parse_trace_rows(text)
    pattern = ROW_THROUGHPUT if harness in ("throughput", "aiperf") else ROW_LATENCY
    columns = COLS[harness]
    rows: dict[int, dict] = {}
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            rows[int(match.group(1))] = {
                name: float(match.group(i + 2)) for i, name in enumerate(columns)
            }
    return rows


# ---------------------------------------------------------------------------- #
#                                PREFLIGHT                                     #
# ---------------------------------------------------------------------------- #

def harness_revision() -> str:
    """Report the harness git revision, preferring the value the launcher passed in.

    An in-cluster pod has no .git, which would record the revision as "unknown" and make the
    result unattributable, so the launcher supplies it.
    """
    return config.HARNESS_GIT_REV or git_revision(config.PROJECT_ROOT, ["app/bench"])


def refuse_loopback_latency(args, protocol: dict, want_latency: bool) -> None:
    """Refuse to measure TTFT through a loopback endpoint.

    A port-forward or SSH tunnel adds roughly 185 ms of connection setup to every TTFT sample,
    measured at 276 ms first byte on a fresh connection against 90 ms warm. That is most of a
    500 ms SLA budget, and it is the harness's own tunnel being reported as the server's
    latency. Refused rather than warned because a warning did not stop it: the 2026-08-08 GLM
    run produced a full latency table through a port-forward and every cell carried a variance
    flag for this reason.

    Throughput-only runs are unaffected — tokens/sec and TPOT do not materially change through
    a tunnel — so this fires only when the run will report TTFT.
    """
    if not any(host in args.endpoint for host in config.LOOPBACK_HOSTS):
        return
    reports_ttft = protocol["harness"] == "latency" or want_latency
    if args.allow_loopback_latency:
        print("  WARNING: driving load over a loopback endpoint "
              "(--allow-loopback-latency).")
        if reports_ttft:
            print("           Every TTFT sample carries ~150-200 ms of tunnel setup.")
        print("           The tunnel may also drop connections mid-run; any level that loses")
        print("           requests is excluded from the medians. Recorded in the lock.")
        return
    sys.exit(
        f"REFUSING TO MEASURE: endpoint {args.endpoint} is loopback.\n\n"
        f"A kubectl port-forward multiplexes every connection through ONE tunnel to the\n"
        f"kubelet. It does not survive hundreds of sustained concurrent streams: on\n"
        f"2026-08-08 it dropped every request in 2 of 3 latency reps with `error creating\n"
        f"error stream ... Timeout occurred`, and the surviving rep's percentiles were the\n"
        f"only thing left to report.\n"
        + (f"\nIt also adds ~150-200 ms of connection setup to every TTFT sample, which is the\n"
           f"harness's own tunnel reported as the server's latency.\n" if reports_ttft else "")
        + f"\nRun in-cluster instead, which reaches the Service over cluster DNS:\n"
        f"  python3 app/bench/src/main.py launch --context <ctx> --namespace <ns> \\\n"
        f"      --service <svc> --protocol {args.protocol} --instance {args.instance} "
        f"--config {args.config} ...\n\n"
        f"To measure anyway, accepting that a dropped connection invalidates a rep:\n"
        f"  --allow-loopback-latency")


def preflight(endpoint: str, model: str) -> None:
    """Refuse to start unless the endpoint is live and serving the expected model."""
    import urllib.request
    url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            served = json.load(response)
    except Exception as exc:
        sys.exit(f"PREFLIGHT FAIL: cannot reach {url}: {type(exc).__name__}: {exc}")
    ids = [entry.get("id") for entry in served.get("data", [])]
    if model not in ids:
        sys.exit(f"PREFLIGHT FAIL: model {model!r} not served. Available: {ids}")
    print(f"  preflight OK: {url} serving {ids}")


def server_alive(endpoint: str, model: str) -> bool:
    """Report whether the endpoint still serves the expected model, without exiting.

    The same check as preflight, but as a question rather than a precondition: mid-run it is
    used to stop early and keep the reps already measured, where preflight's sys.exit would
    discard them.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=30) as response:
            served = json.load(response)
    except Exception:
        return False
    return model in [entry.get("id") for entry in served.get("data", [])]


def recorded_counts(tokenizer: str | None, profile_name: str | None) -> dict | None:
    """Return this tokenizer's recorded counts for a profile, or None if not recorded.

    Read from docs/prompt_tokens/, written by `verify-profiles`. Identical text tokenizes
    differently per model, so the profile's own target_input_tokens is a nominal label rather than
    a measurement: Gemma 4 counts the code profiles about 25% longer than the generator's ratio
    assumed. Whatever is recorded here is what this model actually saw.
    """
    if not (tokenizer and profile_name):
        return None
    path = config.TOKEN_COUNT_DIR / f"{tokenizer.replace('/', '__').replace(':', '_')}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["profiles"][profile_name]
    except (ValueError, KeyError):
        return None


def recorded_raw_size(tokenizer: str | None, profile: dict) -> str:
    """Report the raw input size, preferring this tokenizer's measurement over the nominal label.

    Printing the nominal number alone once meant a Gemma run published `raw in: 8192` for text
    that model reads as 10322 tokens. The nominal value is still shown, marked as such, because it
    is what the filename and the profile's name refer to.
    """
    nominal = profile["target_input_tokens"]
    entry = recorded_counts(tokenizer, profile.get("name"))
    if not entry or entry.get("raw_median") is None:
        return f"{nominal} tok nominal (not measured for this tokenizer)"
    measured, error = entry["raw_median"], entry.get("error_fraction")
    if measured == nominal:
        return f"{measured} tok measured, matching the nominal {nominal}"
    drift = f", {error:+.1%} off nominal" if error is not None else ""
    return f"{measured} tok measured for this tokenizer (nominal {nominal}{drift})"


def recorded_wire_size(tokenizer: str | None, profile_name: str | None) -> str:
    """Report the recorded wire input size for this profile, or how to record it.

    The served model name is not the tokenizer repo id, so this cannot be inferred: --tokenizer
    supplies it. Reported every run rather than left to be remembered, because the number a
    result should quote as its input size lives in that file and nowhere else.
    """
    if not profile_name:
        return "n/a (no profile: generated prompts, input size implicit)"

    install = ("pip install -r app/bench/requirements-tokenizer.txt && "
               "python3 app/bench/src/main.py verify-profiles --model <hf-repo-or-path>")
    if not tokenizer:
        available = sorted(p.stem.replace("__", "/")
                           for p in config.TOKEN_COUNT_DIR.glob("*.json"))
        return (f"UNKNOWN — pass --tokenizer to resolve it. Recorded for: "
                f"{available or 'nothing yet'}. To record: {install}")

    path = config.TOKEN_COUNT_DIR / f"{tokenizer.replace('/', '__').replace(':', '_')}.json"
    if not path.exists():
        return f"NOT RECORDED for {tokenizer} — run: {install}"
    try:
        doc = json.loads(path.read_text())
        entry = doc["profiles"][profile_name]
    except (ValueError, KeyError):
        return f"NOT RECORDED for {profile_name!r} in {path.name} — run: {install}"

    wire = entry.get("wire_median")
    if wire is None:
        return (f"raw {entry.get('raw_median')} tok; wire not measured "
                f"(tokenizer has no chat template)")
    return (f"{wire} tok wire (raw {entry['raw_median']} + template "
            f"{entry['chat_template_overhead']}), from {path.name} "
            f"via transformers {doc.get('measured_with', {}).get('transformers', '?')}")


def reference_thinking_mode(model_tag: str, server_config: str) -> str:
    """Read the thinking mode a stored accuracy reference was captured with.

    Taken FROM the reference rather than passed in, because a hybrid-thinking model produces
    completely different output with the <think> trace on, so a mismatch fails every
    token-count check for the wrong reason. The reference records how it was captured, which
    is the only correct value, so there is nothing for a caller to get wrong.
    """
    # Reuse accuracy.py's locator, since two copies is how one kept pointing at the old layout.
    import accuracy

    ref_file = accuracy.reference_path(model_tag, server_config)
    if not ref_file.exists():
        return "auto"
    try:
        return json.loads(ref_file.read_text()).get("generation", {}).get("thinking", "auto")
    except Exception as exc:
        print(f"  warning: could not read thinking mode from {ref_file.name}: {exc}")
        return "auto"


def accuracy_gate(endpoint: str, model: str, model_tag: str, server_config: str) -> str:
    """Refuse to measure a server whose output does not match its accuracy reference.

    The second half of preflight: preflight proves the endpoint is serving the expected model
    NAME, this proves it is serving the expected model behavior. That distinction is the
    whole point, because the truncated-argv incident passed a name check trivially while
    silently running at 16-bit KV with both parsers missing. Fails closed, since a silent pass
    is the exact failure mode the gate removes.
    """
    command = [sys.executable, str(config.PROJECT_ROOT / "app/bench/src/accuracy.py"),
               "--mode", "fingerprint", "--model-tag", model_tag, "--config", server_config,
               "--thinking", reference_thinking_mode(model_tag, server_config),
               "--endpoint", endpoint, "--model-id", model]
    done = subprocess.run(command, cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    output = (done.stdout + done.stderr).strip()
    if done.returncode != 0:
        print(output)
        sys.exit("\nABORTED: accuracy gate failed. No measurement was taken.\n"
                 "Numbers from this server would not be trustworthy.\n"
                 f"Reference: docs/accuracy/{model_tag}-{server_config}.json\n"
                 "Override with --skip-accuracy only if you understand why it failed.")
    for line in output.splitlines():
        if line.strip().startswith(("reproducible:", "note:")) or "GATE PASSED" in line:
            print(f"  {line.strip()}")
    print(f"  accuracy gate OK: matches docs/accuracy/{model_tag}-{server_config}.json")
    return f"PASSED (docs/accuracy/{model_tag}-{server_config}.json)"


# ---------------------------------------------------------------------------- #
#                               PROVENANCE                                     #
# ---------------------------------------------------------------------------- #

def kubectl(context: str, namespace: str, *args: str, timeout: int = 240) -> str:
    """Run one kubectl command, returning its stdout or a placeholder on failure."""
    output = shell_output(["kubectl", "--context", context, "-n", namespace, *args],
                          timeout=timeout)
    return output or "(command failed or returned nothing)\n"




def discover_pod(context: str, namespace: str) -> tuple[str | None, str]:
    """Find the serving pod to capture provenance from, returning (pod, explanation).

    Nobody should have to look up a pod name to get provenance, and provenance omitted is how
    a result ends up unverifiable after the fact. The match is a Running pod with a container
    named `vllm`, which is what capture_provenance reads logs from anyway.

    Ambiguity is reported rather than guessed: capturing NCCL transport and KV sizing from the
    wrong pod would attach confident, wrong evidence to a result, which is worse than none.
    """
    raw, error = shell_result(["kubectl", "--context", context, "-n", namespace,
                               "get", "pods", "-o", "json"], timeout=60)
    if error:
        # Surfaced rather than reduced to nothing found, since each cause sends the reader elsewhere.
        return None, f"kubectl failed for context {context}: {error.splitlines()[-1][:160]}"
    if not raw.strip():
        return None, f"no pods in namespace {namespace} (context {context})"
    try:
        items = json.loads(raw).get("items", [])
    except ValueError:
        return None, "could not parse kubectl output"

    candidates = [
        pod["metadata"]["name"] for pod in items
        if pod.get("status", {}).get("phase") == "Running"
        and any(c.get("name") == "vllm" for c in pod.get("spec", {}).get("containers", []))
    ]
    if not candidates:
        return None, (f"no Running pod with a `vllm` container in {namespace} "
                      f"(context {context})")
    if len(candidates) > 1:
        return None, (f"ambiguous: {len(candidates)} candidate pods {candidates} — "
                      f"pass --pod to choose")
    return candidates[0], f"auto-discovered {candidates[0]}"


# Placeholder the in-cluster run leaves for the launcher, which owns the only kubectl.
PROVENANCE_SKIPPED = "SERVER PROVENANCE: skipped (--no-provenance)."

# Server-side facts worth recording as (label, regex, reducer), each yielding exactly one line.
PROVENANCE_FACTS = (
    ("NCCL transport", r"NET/Plugin|NET/OFI|NET/Socket|Connected all rings", "last"),
    ("KV cache memory", r"Available KV cache memory: (\S+ GiB)", "first"),
    ("KV cache size", r"GPU KV cache size: (\S+) tokens", "first"),
    ("Max concurrency", r"Maximum concurrency for [\d,]+ tokens per request: (\S+)", "first"),
    ("Prefix cache hit", r"Prefix cache hit rate: (\S+)", "max"),
    ("MoE / seq parallel", r"(sequence parallel|MegaMoE|mega_moe|trtllm|expert.parallel)",
     "last"),
)

_PROMETHEUS_SAMPLE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)$")
_PROMETHEUS_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')


def reduce_matches(logs: str, pattern: str, how: str) -> str:
    """Reduce every match of one legacy log pattern to one reportable value."""
    matches = re.findall(pattern, logs, re.I)
    if not matches:
        return "(not found in server log)"
    values = [m if isinstance(m, str) else next(g for g in m if g) for m in matches]
    if how == "first":
        return values[0]
    if how == "last":
        return values[-1]
    # Max over percentages for cache hit rate, since any non-zero hit voids an uncached claim.
    numeric = [(float(v.rstrip("%")), v) for v in values if v.rstrip("%").replace(".", "", 1)
               .isdigit()]
    return max(numeric)[1] if numeric else values[-1]


def parse_prometheus(metrics: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse the Prometheus text subset needed for provenance, without a new dependency."""
    samples = []
    for line in metrics.splitlines():
        match = _PROMETHEUS_SAMPLE.match(line.strip())
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        labels = {}
        for key, raw in _PROMETHEUS_LABEL.findall(raw_labels or ""):
            try:
                labels[key] = json.loads(f'"{raw}"')
            except (ValueError, json.JSONDecodeError):
                labels[key] = raw
        try:
            value = float(raw_value)
        except ValueError:
            continue
        samples.append((name, labels, value))
    return samples


def metric_info(samples: list[tuple[str, dict[str, str], float]],
                name: str) -> dict[str, str]:
    """Collapse matching info metrics only when every engine reports the same config."""
    infos = [labels for metric, labels, _ in samples if metric == name]
    if not infos:
        return {}
    ignored = {"engine", "model_name", "worker", "rank"}
    signatures = {tuple(sorted((key, value) for key, value in labels.items()
                               if key not in ignored))
                  for labels in infos}
    return infos[0] if len(signatures) == 1 else {"_ambiguous": "true"}


def metric_total(samples: list[tuple[str, dict[str, str], float]],
                 suffix: str) -> float | None:
    """Sum one counter across engines while excluding similarly named created metrics."""
    values = [value for name, _, value in samples if name.endswith(suffix)]
    return sum(values) if values else None


def parse_pod_container(raw_pod: str) -> tuple[dict, dict[str, str], str]:
    """Return the vLLM container, its literal env, and its command text."""
    try:
        pod = json.loads(raw_pod)
    except (TypeError, ValueError):
        return {}, {}, ""
    containers = pod.get("spec", {}).get("containers", [])
    container = next((item for item in containers if item.get("name") == "vllm"),
                     containers[0] if containers else {})
    env = {item.get("name", ""): str(item.get("value", ""))
           for item in container.get("env", []) if "value" in item}
    command = " ".join(str(value) for key in ("command", "args")
                       for value in container.get(key, []))
    return container, env, command


def _flag_token(command: str, flags: tuple[str, ...]) -> str | None:
    """Read a flag value from the shell command, ignoring comments and line continuations."""
    executable = "\n".join(line.split("#", 1)[0] for line in command.splitlines())
    for flag in flags:
        match = re.search(
            rf"(?:^|\s){re.escape(flag)}(?:=|\s+)[\"']?([^\s\"'\\]+)", executable)
        if match:
            return match.group(1)
    return None


def flag_int(command: str, flags: tuple[str, ...], env: dict[str, str],
             default: int) -> int | None:
    """Resolve a literal or `$ENV` integer flag; never guess an unresolved dynamic value."""
    token = _flag_token(command, flags)
    if token is None:
        return default
    cleaned = token.replace(",", "")
    if cleaned.isdigit():
        return int(cleaned)
    variable = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", token)
    if variable:
        value = env.get(variable.group(1), "").replace(",", "")
        return int(value) if value.isdigit() else None
    return None


def serving_parallelism(env: dict[str, str], command: str) -> tuple[str, str]:
    """Describe effective parallel flags and avoid claiming unresolved dynamic values."""
    tp = flag_int(command, ("--tensor-parallel-size",), env, 1)
    pp = flag_int(command, ("--pipeline-parallel-size",), env, 1)
    dp = flag_int(command, ("--data-parallel-size",), env, 1)
    executable = "\n".join(line.split("#", 1)[0] for line in command.splitlines())
    expert = bool(re.search(r"(?:^|\s)--enable-expert-parallel(?:\s|\\|$)", executable))
    sequence = bool(re.search(r"(?:^|\s)--enable-sequence-parallel(?:\s|\\|$)",
                              executable))
    degree = lambda name, value: f"{name}{value}" if value is not None else f"{name}unknown"
    summary = (f"{degree('TP', tp)}, {degree('PP', pp)}, {degree('DP', dp)}; "
               f"expert parallel {'on' if expert else 'off'}; "
               f"sequence parallel {'on' if sequence else 'off'}")
    applicability = ("expert/sequence parallel enabled in serving argv" if expert or sequence
                     else "expert/sequence parallel disabled in serving argv")
    return summary, applicability


def nccl_transport(logs: str) -> str:
    """Report selected NCCL channel paths; qualify backend-only initialization evidence."""
    selected = []
    for line in logs.splitlines():
        if not (re.search(r"NCCL INFO", line, re.I)
                and re.search(r"\bChannel\s+", line, re.I)
                and re.search(r"\s+via\s+", line, re.I)):
            continue
        value = re.split(r"\s+via\s+", line, maxsplit=1, flags=re.I)[1].strip()
        if re.fullmatch(r"(?:P2P|SHM|NET|NVL|NVLS|CollNet|IPC)[A-Za-z0-9_./: -]*",
                        value, re.I):
            selected.append(value)
    selected = list(dict.fromkeys(selected))
    if selected:
        return ", ".join(selected)
    backends = re.findall(r"\b(NET/(?:Plugin|OFI|Socket))\b", logs, re.I)
    backends = list(dict.fromkeys(backends))
    if backends:
        return f"unknown (initialized backends: {', '.join(backends)})"
    if re.search(r"Connected all rings", logs, re.I):
        return "unknown (rings connected; selected path not logged)"
    return "(not found in server log)"


def gpu_topology_summary(topology: str) -> str:
    """Parse only the GPU matrix from `nvidia-smi topo -m`, excluding NIC columns."""
    topology = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", topology)
    lines = [line.split() for line in topology.splitlines() if line.split()]
    header = next((parts for parts in lines if parts and parts[0] == "GPU0"), None)
    # A real header has GPU1 right after GPU0, unlike the first GPU row which repeats GPU0.
    header = next((parts for parts in lines
                   if len(parts) > 1 and parts[0] == "GPU0" and parts[1] == "GPU1"), header)
    if not header:
        return "(not available)"
    gpu_columns = [name for name in header if re.fullmatch(r"GPU\d+", name)]
    if not gpu_columns:
        return "(not available)"
    rows = {parts[0]: parts[1:1 + len(gpu_columns)] for parts in lines
            if parts[0] in gpu_columns and len(parts) > len(gpu_columns)}
    pairs = []
    for left_index, left in enumerate(gpu_columns):
        links = rows.get(left, [])
        for right_index in range(left_index + 1, len(gpu_columns)):
            if right_index >= len(links):
                return "(ambiguous topology output)"
            pairs.append((left, gpu_columns[right_index], links[right_index]))
    if not pairs:
        return "single GPU"
    link_types = {link for _, _, link in pairs}
    if len(link_types) == 1:
        return f"all GPU pairs: {next(iter(link_types))}"
    return ", ".join(f"{left}<->{right}: {link}" for left, right, link in pairs)


def _optional_int(value: str | None) -> int | None:
    """Parse a non-negative integer metric label, treating None-like values as absent."""
    if value is None:
        return None
    cleaned = value.replace(",", "")
    return int(cleaned) if cleaned.isdigit() else None


def _optional_float(value: str | None) -> float | None:
    """Parse a finite float metric label, treating changed or malformed values as absent."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def provenance_facts(logs: str, metrics: str, raw_pod: str,
                     topology: str) -> list[tuple[str, str]]:
    """Resolve vLLM 0.26 metrics/pod facts with v0.21+ log-compatible fallbacks."""
    legacy = {label: reduce_matches(logs, pattern, how)
              for label, pattern, how in PROVENANCE_FACTS}
    samples = parse_prometheus(metrics)
    cache = metric_info(samples, "vllm:cache_config_info")
    _, env, command = parse_pod_container(raw_pod)

    cache_ambiguous = cache.get("_ambiguous") == "true"
    block_size = None if cache_ambiguous else _optional_int(cache.get("block_size"))
    gpu_blocks = None if cache_ambiguous else _optional_int(cache.get("num_gpu_blocks"))
    cache_tokens = (None if cache_ambiguous
                    else _optional_int(cache.get("kv_cache_size_tokens")))
    if cache_tokens is None and block_size is not None and gpu_blocks is not None:
        cache_tokens = block_size * gpu_blocks
    max_model_len = flag_int(command, ("--max-model-len",), env, 0) or 0

    if cache_ambiguous:
        cache_memory = cache_size = "mixed across engines; inspect /metrics"
    else:
        cache_memory = legacy["KV cache memory"]
        memory_bytes = _optional_int(cache.get("kv_cache_memory_bytes"))
        if memory_bytes is not None:
            cache_memory = f"{memory_bytes / 1024 ** 3:.2f} GiB"
        elif cache and cache_memory.startswith("(not found"):
            cache_memory = "not reported directly; token/block capacity captured below"
        cache_size = (f"{cache_tokens:,} tokens" if cache_tokens is not None
                      else legacy["KV cache size"])

    raw_max_concurrency = (None if cache_ambiguous
                           else _optional_float(cache.get("kv_cache_max_concurrency")))
    if cache_ambiguous:
        max_concurrency = "mixed across engines; inspect /metrics"
    elif raw_max_concurrency is not None:
        max_concurrency = f"{raw_max_concurrency:.4f}x"
    elif cache_tokens is not None and max_model_len:
        max_concurrency = f"{cache_tokens / max_model_len:.4f}x"
    else:
        max_concurrency = legacy["Max concurrency"]
    if max_model_len and not max_concurrency.startswith(("(", "mixed")):
        max_concurrency += f" at {max_model_len:,} tokens/request"

    enabled_value = cache.get("enable_prefix_caching")
    if enabled_value in ("True", "False"):
        prefix_enabled = enabled_value == "True"
    elif "--no-enable-prefix-caching" in command:
        prefix_enabled = False
    elif "--enable-prefix-caching" in command:
        prefix_enabled = True
    else:
        prefix_enabled = None
    queries = metric_total(samples, "prefix_cache_queries_total")
    hits = metric_total(samples, "prefix_cache_hits_total")
    if queries is not None and hits is not None:
        if queries:
            prefix_hit = f"{100 * hits / queries:.1f}% ({hits:,.0f}/{queries:,.0f})"
        else:
            prefix_hit = "0.0% (0 hits / 0 queries)"
        prefix_io = f"{queries:,.0f} queries / {hits:,.0f} hits"
    else:
        prefix_hit = legacy["Prefix cache hit"]
        prefix_io = "(not exposed by this vLLM version)"

    parallelism, parallel_features = serving_parallelism(env, command)
    moe_or_sequence = legacy["MoE / seq parallel"]
    if moe_or_sequence.startswith("(not found") and command:
        moe_or_sequence = parallel_features

    facts = [("NCCL transport", nccl_transport(logs)),
             ("GPU topology", gpu_topology_summary(topology))]
    if env.get("CUDA_VISIBLE_DEVICES"):
        facts.append(("CUDA devices", env["CUDA_VISIBLE_DEVICES"]))
    if cache.get("cache_dtype") not in (None, "None"):
        facts.append(("KV cache dtype", cache["cache_dtype"]))
    if block_size is not None:
        facts.append(("KV block size", f"{block_size} tokens"))
    if gpu_blocks is not None:
        facts.append(("KV GPU blocks", f"{gpu_blocks:,}"))
    facts += [("KV cache memory", cache_memory),
              ("KV cache size", cache_size),
              ("Max concurrency", max_concurrency)]
    if prefix_enabled is not None:
        facts.append(("Prefix caching", "enabled" if prefix_enabled else "disabled"))
    if queries is not None or hits is not None:
        facts.append(("Prefix cache I/O", prefix_io))
    facts += [("Prefix cache hit", prefix_hit)]
    if command:
        facts.append(("Parallelism", parallelism))
    facts.append(("MoE / seq parallel", moe_or_sequence))
    return facts


def collect_provenance(pod: str, context: str, namespace: str) -> list[str]:
    """Collect metrics/pod provenance, retaining v0.21+ server-log fallbacks."""
    logs = kubectl(context, namespace, "logs", pod, "-c", "vllm", "--tail=200000")
    gpus = kubectl(context, namespace, "exec", pod, "-c", "vllm", "--", "nvidia-smi",
                   "--query-gpu=name,memory.total", "--format=csv,noheader")
    topology = kubectl(context, namespace, "exec", pod, "-c", "vllm", "--",
                       "nvidia-smi", "topo", "-m")
    raw_pod = kubectl(context, namespace, "get", "pod", pod, "-o", "json")
    container, _, _ = parse_pod_container(raw_pod)
    ports = [item.get("containerPort") for item in container.get("ports", [])
             if item.get("containerPort")]
    port = ports[0] if ports else 8000
    metrics_script = ("import urllib.request; "
                      f"print(urllib.request.urlopen('http://127.0.0.1:{port}/metrics', "
                      "timeout=10).read().decode())")
    metrics = kubectl(context, namespace, "exec", pod, "-c", "vllm", "--",
                      "python3", "-c", metrics_script)
    models = [] if gpus.startswith("(command failed") else sorted(
        {line.strip() for line in gpus.splitlines() if line.strip()})

    lines = ["", "SERVER PROVENANCE (from the serving pod, at measurement time).",
             f"  captured : {utc_stamp('%Y-%m-%d %H:%M:%S')} UTC",
             f"  pod      : {pod}  ({context}/{namespace})"]
    if models:
        lines.append(f"  GPUs     : {len(gpus.splitlines())}x {models[0]}")
    for label, value in provenance_facts(logs, metrics, raw_pod, topology):
        lines.append(f"  {label:<18}: {value}")
    lines.append("  Prefix cache hit must be 0% unless the protocol is deliberately "
                 "cache-seeking;")
    lines.append("  any other value means prefill was skipped and throughput is inflated.")
    return lines


# ---------------------------------------------------------------------------- #
#                          RUN SETUP & PROTOCOL LOCK                           #
# ---------------------------------------------------------------------------- #

def requests_for_level(concurrency: int, min_samples: int, waves: int) -> int:
    """Size a level's request count as whole waves, with enough samples for a p99.

    Whole waves matter because the latency harness issues requests in waves of exactly
    `concurrency`; a partial wave means in-flight depth is lower than the level being
    reported, which silently understates latency.
    """
    target = max(min_samples, waves * concurrency)
    return ((target + concurrency - 1) // concurrency) * concurrency


def load_profile_doc(name: str | None) -> dict | None:
    """Load the requested prompt profile, exiting with the available names if it is unknown."""
    if not name:
        return None
    path = config.PROMPT_PROFILE_DIR / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in config.PROMPT_PROFILE_DIR.glob("*.json"))
        sys.exit(f"unknown --profile {name!r}. Available: {available}")
    return json.loads(path.read_text())


def resolve_profile(args, protocol: dict) -> str | None:
    """Decide which pinned profile this run serves.

    Precedence: an explicit --profile, then the protocol's own default, then none. Defaulting
    to the protocol's profile is what makes pinned input the normal case rather than something
    a caller has to remember; --no-profile is the deliberate opt-out for the generated-prompt
    path, which two protocols still require.
    """
    if args.no_profile:
        return None
    return args.profile or protocol.get("profile")


def build_env(args, protocol: dict, profile: dict | None) -> dict:
    """Build the child environment: protocol settings, then the profile, then run identity.

    A profile overrides the protocol's prompt settings AND its output cap. Half-applying it
    would produce a run labelled with a profile whose output shape came from elsewhere.
    """
    env = config.inherited_environment()
    # Caller targets outrank the protocol's shipped values, which env.update would otherwise discard.
    caller_targets = {k: env[k] for k in ("TTFT_SLA_MS", "TPOT_SLA_MS") if k in env}
    env.update(protocol["env"])
    env.update(caller_targets)
    # A flag outranks an env var and is the only form that survives the trip into the Job.
    for flag, key in (("ttft_target", "TTFT_SLA_MS"), ("tpot_target", "TPOT_SLA_MS")):
        value = getattr(args, flag, None)
        if value is not None:
            env[key] = f"{value:g}"
    if profile:
        env["PROMPT_PROFILE"] = args.profile
        env["MAX_TOKENS"] = str(profile["output_tokens"])
        env.pop("PROMPT_TOKENS", None)
        env.pop("PROMPT_MODE", None)
    if args.trace:
        # Travels as SUITE_ARGS, unlike an exported env var the Job would never see.
        env["TRACE_FILE"] = args.trace
    if env.get("AIPERF_MODE") == "trace" and not env.get("TRACE_FILE"):
        sys.exit(f"protocol {args.protocol!r} replays a trace but none is set: pass "
                 f"--trace <repo-relative-path> (see app/bench/data/traces/README.md)")
    env.update({
        "VLLM_API_URL": args.endpoint,
        "MODEL_ID": args.model_id,
        "INSTANCE_TYPE": args.instance,
        "QUANTIZATION": args.quant,
        "PYTHONUNBUFFERED": "1",
    })
    return env


def build_lock(args, protocol: dict, profile: dict | None, env: dict | None = None) -> list[str]:
    """Compose the PROTOCOL LOCK header recorded at the top of every result file.

    Values come from the RESOLVED environment, not from the protocol table, so a caller who
    overrides a latency target sees the number the run actually used. Printing the table
    instead reported the shipped value no matter what was in force.
    """
    overridden = ("PROMPT_MODE", "PROMPT_TOKENS", "MAX_TOKENS")
    effective = {k: (env or {}).get(k, v) for k, v in protocol["env"].items()}
    lock = [
        "PROTOCOL LOCK",
        f"  protocol      : {args.protocol}",
        f"  harness       : {config.HARNESS_PATHS[protocol['harness']]}",
        f"  harness rev   : {harness_revision()}",
        f"  server config : {args.config}",
        f"  instance      : {args.instance}",
        f"  endpoint      : {args.endpoint}",
        f"  model         : {args.model_id}",
        f"  reps          : {args.reps}",
        f"  started       : {utc_stamp('%Y-%m-%d %H:%M:%S')} UTC",
        "  environment   :",
        *[f"      {k}={v}"
          + ("   [OVERRIDDEN by --profile]" if profile and k in overridden else "")
          + ("   [OVERRIDDEN by the caller]"
             if v != protocol["env"][k] and not (profile and k in overridden) else "")
          for k, v in sorted(effective.items())],
    ]
    lock += swept_axis_lines(protocol)
    lock += trace_lines(args, protocol)
    lock += profile_lines(profile,
                          recorded_wire_size(args.tokenizer, args.profile),
                          recorded_raw_size(args.tokenizer, profile) if profile else "")
    lock.append(f"  note          : {protocol['note']}")
    return lock


def trace_lines(args, protocol: dict) -> list[str]:
    """Record the replayed trace's path AND content hash in the lock.

    The hash is the load-bearing part: a trace file replaced in place would change the
    workload while every other lock line stayed identical, and two results claiming the
    same trace would not be comparable. With the digest recorded, "same trace" is
    checkable from the results files alone.
    """
    trace = args.trace or protocol["env"].get("TRACE_FILE")
    if not trace or protocol["env"].get("AIPERF_MODE") != "trace":
        return []
    path = config.PROJECT_ROOT / trace
    if not path.exists():
        sys.exit(f"trace file not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    records = sum(1 for line in path.read_text().splitlines() if line.strip())
    return [f"  trace         : {trace}  ({records} records, sha256:{digest})"]


def swept_axis_lines(protocol: dict) -> list[str]:
    """Record which axis was swept, since it is set per invocation rather than in the env."""
    lines = []
    if protocol.get("levels"):
        lines.append(f"  concurrency   : {protocol['levels']}")
    if protocol.get("ctx_levels"):
        lines.append(f"  PROMPT_TOKENS : {protocol['ctx_levels']}  "
                     f"(at fixed concurrency {protocol['concurrency']})")
    if protocol.get("qps_levels"):
        lines.append(f"  target QPS    : {protocol['qps_levels']}")
    return lines


def profile_lines(profile: dict | None, wire_size: str = "", raw_size: str = "") -> list[str]:
    """Record the pinned prompt profile, or state plainly that none was used."""
    if not profile:
        return ["  prompt profile: NONE (generated prompts; input size is implicit — "
                "prefer --profile for anything published)"]
    duplicates = len(profile.get("intentional_duplicate_indices", []))
    lines = [
        f"  prompt profile: {profile['name']}  "
        f"(app/bench/data/prompt_profiles/{profile['name']}.json)",
        f"      prompts   : {profile['prompt_count']} pinned, "
        f"{profile['distinct_prompt_count']} distinct, {duplicates} declared duplicates",
        f"      raw in    : {raw_size}  "
        f"(prompt text only; NOT what the server counts)",
        f"      wire in   : {wire_size or 'not resolved'}",
        f"      output cap: {profile['output_tokens']} tok  (from the profile, "
        f"overriding the protocol)",
    ]
    if profile.get("legacy"):
        lines.append("      WARNING   : legacy shape, historical continuity only")
    if profile.get("fixed_prompt_set"):
        lines.append("      WARNING   : fixed prompt set — repeats; invalid as throughput "
                     "if prefix caching is on")
    return lines


# ---------------------------------------------------------------------------- #
#                            HARNESS INVOCATION                                #
# ---------------------------------------------------------------------------- #

def run_harness(harness: str, env: dict) -> tuple[str, str, float]:
    """Invoke the harness once, returning (stdout, file body, client CPU fraction).

    The CPU fraction is child CPU seconds over wall seconds. Near or above 1.0 means the
    harness itself was pegging a core, so streaming latency from that level is suspect: we
    would be measuring the client rather than the server.
    """
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-u", str(config.PROJECT_ROOT / config.HARNESS_PATHS[harness])],
        cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True)
    wall = max(time.perf_counter() - started, 1e-6)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    body = done.stdout + ("\nSTDERR:\n" + done.stderr if done.stderr.strip() else "")
    return done.stdout, body, cpu / wall


def harness_self_id_lines(rep: "RepResult") -> list[str]:
    """Collect a harness's self-identification lines (version, generated argv) for the lock.

    Emitted by harnesses that wrap an external tool, where the tool's version and the exact
    argv handed to it are part of the conditions that produced the number. Deterministic
    given the pinned dependency and the environment, so recording them from rep 1 adds
    provenance without letting a result be reinterpreted.
    """
    lines = []
    for body in rep.bodies:
        for line in body.splitlines():
            if line.startswith(("AIPERF_VERSION ", "AIPERF_ARGV ")):
                key, _, value = line.partition(" ")
                lines.append(f"  {key.lower():<14}: {value}")
    # One argv per level is noise; the first shows the template, the count shows the sweep.
    argv_lines = [l for l in lines if l.startswith("  aiperf_argv")]
    keep = [l for l in lines if not l.startswith("  aiperf_argv")]
    if argv_lines:
        keep.append(argv_lines[0])
        if len(argv_lines) > 1:
            keep.append(f"  aiperf_argv   : (+{len(argv_lines) - 1} more levels, "
                        f"identical except --concurrency and --output-artifact-dir)")
    return keep


def saturation_flag(cpu_fraction: float) -> str:
    """Return a warning marker when the client may have been the bottleneck."""
    return "  <-- CLIENT MAY BE SATURATED" if cpu_fraction > CLIENT_CPU_SUSPECT else ""


class RepResult:
    """Accumulates one rep's parsed rows, token shapes, client CPU samples and raw bodies."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.rows: dict[int, dict] = {}
        self.shapes: dict[int, dict] = {}
        self.fills: dict[int, dict] = {}
        self.success: dict[int, dict] = {}
        self.bodies: list[str] = []

    def add_body(self, header: str, body: str) -> None:
        """Append one invocation's raw output under a describing header."""
        self.bodies.append(f"----- {header} -----\n{body}")


def sweep_context_levels(protocol: dict, env: dict, rep: RepResult,
                         client_cpu: dict[int, list[float]]) -> None:
    """Sweep PROMPT_TOKENS at fixed concurrency, re-keying rows by context length.

    The latency harness keys its summary row by concurrency, which is constant here, so every
    invocation would return the same key and overwrite the previous one.
    """
    concurrency = protocol["concurrency"]
    count = requests_for_level(concurrency, protocol["min_samples"], protocol["waves"])
    for context_tokens in protocol["ctx_levels"]:
        level_env = dict(env, CONCURRENCY=str(concurrency), NUM_REQUESTS=str(count),
                         PROMPT_TOKENS=str(context_tokens))
        print(f"  ctx={context_tokens:<6} c={concurrency} requests={count:<5} ...",
              end="", flush=True)
        stdout, body, cpu = run_harness(protocol["harness"], level_env)
        parsed, shapes = parse_rows(stdout, protocol["harness"]), parse_token_shape(stdout)
        # Re-keyed on context length, matching how this protocol keys its rows.
        for entry in parse_success(stdout).values():
            rep.success[context_tokens] = entry
        if concurrency in parsed:
            rep.rows[context_tokens] = parsed[concurrency]
        if concurrency in shapes:
            rep.shapes[context_tokens] = shapes[concurrency]
        client_cpu.setdefault(context_tokens, []).append(cpu)
        if context_tokens in rep.rows:
            print(f" client_cpu={cpu:.2f} "
                  f"ttft_p50={rep.rows[context_tokens]['ttft_p50_s']:.3f}s{saturation_flag(cpu)}")
        else:
            print(f" NO ROWS PARSED{saturation_flag(cpu)}")
        rep.add_body(f"prompt_tokens={context_tokens} concurrency={concurrency} "
                     f"requests={count} client_cpu={cpu:.2f}", body)


def sweep_qps_levels(protocol: dict, env: dict, rep: RepResult,
                     client_cpu: dict[int, list[float]], scratch: Path,
                     server_config: str, rep_number: int) -> None:
    """Run one invocation per target arrival rate.

    Unlike the closed-loop sweep there is no request count to size: LoadGen's own query and
    duration floors govern how long each level runs.
    """
    settle_s = protocol.get("settle_s", 0)
    for index, qps in enumerate(protocol["qps_levels"]):
        # Let the server drain, since only open loop can leave a backlog for the next level's percentiles.
        if index and settle_s:
            print(f"  settle {settle_s}s (drain before next level) ...", flush=True)
            time.sleep(settle_s)
        # Liveness per level, so a rep that loses its server stops rather than measuring nothing for minutes.
        if not server_alive(env["VLLM_API_URL"], env["MODEL_ID"]):
            print(f"  !! server not serving {env['MODEL_ID']!r} — abandoning the rest of this "
                  f"rep at target_qps={qps}. Levels already measured are kept; this rep will "
                  f"be excluded for incomplete request success.")
            return
        level_env = dict(env, SERVER_TARGET_QPS=str(qps),
                         # Keep LoadGen's own logs out of the results directory, which holds one file per run.
                         LOADGEN_OUTDIR=str(scratch /
                                            f"loadgen_{server_config}_qps{qps}_rep{rep_number}"))
        print(f"  target_qps={qps:<5} ...", end="", flush=True)
        stdout, body, cpu = run_harness(protocol["harness"], level_env)
        parsed = parse_rows(stdout, protocol["harness"])
        rep.rows.update(parsed)
        level_key = int(round(float(qps) * 1000))
        # Re-keyed onto this protocol's level units, without which degraded_levels() excluded nothing.
        for entry in parse_success(stdout).values():
            rep.success[level_key] = entry
        client_cpu.setdefault(level_key, []).append(cpu)
        if parsed:
            row = next(iter(parsed.values()))
            print(f" client_cpu={cpu:.2f} achieved={row['achieved_qps']:.2f} "
                  f"ttft_p99={row['ttft_p99_ms']:.0f}ms "
                  f"tpot_p99={row['tpot_p99_ms']:.1f}ms{saturation_flag(cpu)}")
        else:
            print(f" NO ROWS PARSED{saturation_flag(cpu)}")
        rep.add_body(f"target_qps={qps} client_cpu={cpu:.2f}", body)


def sweep_per_level(protocol: dict, env: dict, rep: RepResult,
                    client_cpu: dict[int, list[float]]) -> None:
    """Run one invocation per concurrency level, sized so every level runs whole waves."""
    for level in protocol["levels"]:
        count = requests_for_level(level, protocol["min_samples"], protocol["waves"])
        level_env = dict(env, CONCURRENCY=str(level), NUM_REQUESTS=str(count))
        print(f"  c={level:<4} requests={count:<5} ({count // level} whole waves) ...",
              end="", flush=True)
        stdout, body, cpu = run_harness(protocol["harness"], level_env)
        parsed = parse_rows(stdout, protocol["harness"])
        rep.rows.update(parsed)
        rep.shapes.update(parse_token_shape(stdout))
        rep.fills.update(parse_fill(stdout))
        rep.success.update(parse_success(stdout))
        client_cpu.setdefault(level, []).append(cpu)
        print(f" client_cpu={cpu:.2f}{saturation_flag(cpu)}" if parsed
              else f" NO ROWS PARSED{saturation_flag(cpu)}")
        rep.add_body(f"concurrency={level} requests={count} client_cpu={cpu:.2f}", body)


def run_single_sweep(protocol: dict, env: dict, rep: RepResult) -> None:
    """Run the harness once and let it walk its own concurrency sweep internally."""
    stdout, body, cpu = run_harness(protocol["harness"], env)
    rep.rows.update(parse_rows(stdout, protocol["harness"]))
    rep.shapes.update(parse_token_shape(stdout))
    rep.fills.update(parse_fill(stdout))
    rep.success.update(parse_success(stdout))
    rep.bodies.append(body)
    print(f"  levels {sorted(rep.rows) or 'NONE'}  client_cpu={cpu:.2f}")


def run_latency_companion(protocol: dict, env: dict, rep: RepResult) -> dict[int, dict]:
    """Run the streaming latency harness across the same levels as the throughput sweep.

    A throughput run is non-streaming by design, so it can report end-to-end latency but not
    TTFT, TPOT or ITL — those need per-token timing. This second pass supplies them for the
    same concurrency axis, reported as its OWN table beneath the throughput one. The two are
    never merged: they come from different requests under different wire settings, and a
    single row implying otherwise would invite exactly the comparison that is invalid.
    """
    levels = swept_levels(protocol)
    rows: dict[int, dict] = {}
    for level in levels:
        count = requests_for_level(level, LATENCY_MIN_SAMPLES, LATENCY_WAVES)
        level_env = dict(env, CONCURRENCY=str(level), NUM_REQUESTS=str(count))
        print(f"  [latency] c={level:<4} requests={count:<5} ...", end="", flush=True)
        stdout, body, cpu = run_harness("latency", level_env)
        parsed = parse_rows(stdout, "latency")
        rows.update(parsed)
        print(f" client_cpu={cpu:.2f}{saturation_flag(cpu)}" if parsed
              else f" NO ROWS PARSED{saturation_flag(cpu)}")
        rep.add_body(f"[latency companion] concurrency={level} requests={count} "
                     f"client_cpu={cpu:.2f}", body)
    return rows


def swept_levels(protocol: dict) -> list[int]:
    """Return the concurrency levels a protocol sweeps, for the companion pass."""
    return config.swept_concurrency(protocol) or list(config.CANONICAL_SWEEP)


def run_rep(args, protocol: dict, env: dict, scratch: Path, rep_number: int,
            client_cpu: dict[int, list[float]]) -> RepResult:
    """Run one rep using whichever sweep shape the protocol declares."""
    rep = RepResult()
    if protocol.get("ctx_levels"):
        sweep_context_levels(protocol, env, rep, client_cpu)
    elif protocol.get("qps_levels"):
        sweep_qps_levels(protocol, env, rep, client_cpu, scratch, args.config, rep_number)
    elif protocol.get("per_level"):
        sweep_per_level(protocol, env, rep, client_cpu)
    else:
        run_single_sweep(protocol, env, rep)
    return rep


# ---------------------------------------------------------------------------- #
#                              AGGREGATION                                     #
# ---------------------------------------------------------------------------- #

def metric_table(harness: str, label: str, levels: list[int],
                 reps: list[dict[int, dict]]) -> tuple[list[str], list[str]]:
    """Build the median metric table, plus a warning for any cell with untrustworthy spread."""
    columns = COLS[harness]
    # Fixed column width, so a long metric name cannot run into its neighbour and corrupt the table.
    widths = {name: max(16, len(name) + 2) for name in columns}
    header = f"{label:>8}" + "".join(f"{name:>{widths[name]}}" for name in columns)
    lines, warnings = [header, "-" * len(header)], []
    for level in levels:
        row = f"{fmt_level(harness, level):>8}"
        for name in columns:
            values = [rep[level][name] for rep in reps if level in rep]
            middle = statistics.median(values)
            row += f"{middle:>{widths[name]}.3f}"
            if spread_fraction(values) > MAX_SPREAD:
                warnings.append(
                    f"  {label}={fmt_level(harness, level)} {name}: spread "
                    f"{min(values):.3f}..{max(values):.3f} is >{MAX_SPREAD:.0%} of median "
                    f"{middle:.3f} — distrust this cell")
        lines.append(row)
    return lines, warnings


def latency_targets(env: dict | None) -> tuple[float, float]:
    """Resolve the latency targets actually in force, so the report cannot quote a stale one."""
    source = env or {}

    def number(key: str, fallback: float) -> float:
        try:
            return float(source[key])
        except (KeyError, TypeError, ValueError):
            return fallback

    return number("TTFT_SLA_MS", config.TTFT_SLA_MS), number("TPOT_SLA_MS", config.TPOT_SLA_MS)


def targets_section(levels: list[int], reps: list[dict[int, dict]],
                    targets: tuple[float, float]) -> tuple[list[str], list[str]]:
    """Report each arrival rate against the configured targets, naming what bound, not a verdict.

    This replaced a per-level MET/NOT_MET/VALID/INVALID column. Those labels were ours, not
    LoadGen's, and they were computed from the two percentile columns printed immediately
    above them, so they carried no information the reader did not already have. What they did
    carry was alarm: a sweep exists to find the arrival rate where latency breaks, so most
    levels sitting past it is the sweep succeeding, and a table of NOT_MET read as a failed
    run. Naming the binding target instead answers the question a reader actually has, which
    is which number to argue about.
    """
    ttft_target, tpot_target = targets

    def median_of(level: int, name: str) -> float:
        return statistics.median([rep[level][name] for rep in reps if level in rep])

    lines = ["",
             "LATENCY TARGETS. These are inputs you chose for this run, not a standard, and the",
             "highest arrival rate a server sustains is as much a property of them as of the",
             f"server:  TTFT p99 <= {ttft_target:g} ms   TPOT p99 <= {tpot_target:g} ms"
             f"   (override with --ttft-target / --tpot-target)"]

    over: dict[int, list[str]] = {}
    relaxing: list[tuple[int, float]] = []
    for level in levels:
        ttft, tpot = median_of(level, "ttft_p99_ms"), median_of(level, "tpot_p99_ms")
        names = [n for n, value, target in (("TTFT", ttft, ttft_target),
                                           ("TPOT", tpot, tpot_target)) if value > target]
        over[level] = names
        # The smallest TPOT target holding a rate is its own p99, but only while TTFT is inside target.
        if names == ["TPOT"]:
            relaxing.append((level, tpot))

    held = [level for level in levels if not over[level]]
    if held:
        lines.append(f"  Highest arrival rate holding both targets: "
                     f"{fmt_level('arrival_rate', max(held))} QPS")
    else:
        lines.append("  No arrival rate in this sweep held both targets. Either the targets are "
                     "tighter than this server can serve, or the sweep started above its knee.")

    lines.append("  Target exceeded per level, reading the p99 columns above for the values:")
    for level in levels:
        detail = " and ".join(over[level]) if over[level] else "neither, both targets held"
        lines.append(f"     {fmt_level('arrival_rate', level):>9}   {detail}")

    if relaxing:
        # Round the budget up, since rounding down states a budget the level would actually breach.
        lines.append(f"  Holding TTFT at {ttft_target:g} ms, the smallest TPOT target that")
        for level, tpot in relaxing:
            lines.append(f"    would hold {fmt_level('arrival_rate', level):>9} QPS is "
                         f"{math.ceil(tpot):.0f} ms")
    ttft_bound = [level for level in levels if "TTFT" in over[level]]
    if ttft_bound:
        # Not "TTFT is over too": TTFT alone caps the rate here, so a looser TPOT buys nothing.
        lines.append(f"  From {fmt_level('arrival_rate', min(ttft_bound))} QPS up TTFT is over "
                     f"its target, so a looser TPOT target alone would not raise the rate that "
                     f"holds.")

    # Sample sufficiency is the one verdict the columns cannot give, and only matters inside targets.
    uncertified = [level for level in levels
                   if any(rep[level].get("_early") == "UNDERSAMPLED"
                          for rep in reps if level in rep)]
    quotable = [level for level in uncertified if not over[level]]
    if uncertified:
        listed = ", ".join(fmt_level('arrival_rate', level) for level in uncertified)
        lines += ["  LoadGen could not certify the p99 from the samples taken at "
                  f"{listed} QPS.",
                  "    A sample-size statement, not a latency result. Expected at any rate past",
                  "    the targets, where nearly every request contributes a breach and the",
                  "    certification requirement grows with them."]
    warnings = [f"  tgt_qps={fmt_level('arrival_rate', level)} held its targets but LoadGen could "
                f"not certify the p99 from the sample size. Raise MIN_QUERY_COUNT or "
                f"MIN_DURATION_S before quoting this percentile"
                for level in quotable]
    return lines, warnings


def token_shape_table(harness: str, shapes: dict[int, dict]) -> list[str]:
    """Report the server-counted input size behind each level's figures."""
    if not shapes:
        return []
    lines = ["", "MEASURED TOKEN SHAPE (server-counted, per level).",
             "  tokens/sec elsewhere in this file counts OUTPUT tokens only. These are",
             "  the INPUT sizes that produced those figures.",
             "  Input INCLUDES the chat template, because that is what the server received.",
             "  A profile's target_input_tokens counts the prompt text alone and reads lower;",
             "  docs/prompt_tokens/<model>.json records raw, wire and the overhead between.",
             f"  {'level':>8} {'input median':>13} {'input min':>10} {'input max':>10} "
             f"{'output cap':>11}"]
    for level in sorted(shapes):
        shape = shapes[level]
        lines.append(f"  {fmt_level(harness, level):>8} {shape.get('input_median', 0):>13} "
                     f"{shape.get('input_min', 0):>10} {shape.get('input_max', 0):>10} "
                     f"{shape.get('output_cap', 0):>11}")
    return lines


def latency_companion_table(reps: list[dict[int, dict]]) -> tuple[list[str], list[str]]:
    """Render the streaming latency figures as a separate table below the throughput one."""
    if not reps or not any(reps):
        return [], []
    levels = sorted({level for rep in reps for level in rep})
    lines = ["", "=" * 60,
             "LATENCY (streaming) — SEPARATE MEASUREMENT, same concurrency sweep.",
             "  Produced by a second pass with streaming enabled, because a non-streaming",
             "  throughput run cannot observe TTFT/TPOT/ITL. Different requests from the",
             "  table above: read them side by side, never as one row.", ""]
    table, warnings = metric_table("latency", "conc", levels, reps)
    lines += table
    lines += ["", "per-rep raw values (latency):"]
    for level in levels:
        for name in COLS_LATENCY:
            values = [rep[level][name] for rep in reps if level in rep]
            lines.append(f"  conc={level:<9} {name:<16} {values}")
    return lines, [f"  [latency] {w.strip()}" for w in warnings]


def success_table(harness: str, per_rep: list[dict[int, dict]],
                  excluded: list[int]) -> tuple[list[str], list[str]]:
    """Report request success per level per rep, and why any rep was excluded.

    Reported per rep rather than aggregated: a median hides exactly the case this exists to
    expose, where one rep is intact and another lost most of its requests. Before this, the
    only record of a lost request was an `(n/total ok)` line inside a rep file, and the
    2026-08-08 latency run published percentiles from a single surviving rep while two others
    had failed every request to a collapsed port-forward.
    """
    if not per_rep or not any(per_rep):
        return [], []
    levels = sorted({level for rep in per_rep for level in rep})
    lines = ["", "REQUEST SUCCESS per level, per rep.",
             f"  A rep with any level below {config.MIN_LEVEL_SUCCESS:.0%} is EXCLUDED from "
             f"the medians above.",
             "  transport = the client's connection broke, so the sample is invalid.",
             "  server    = the server rejected the request, which IS a result.",
             f"  {'level':>8}  " + "  ".join(f"{'rep' + str(i + 1):>22}"
                                             for i in range(len(per_rep)))]
    for level in levels:
        cells = []
        for rep in per_rep:
            entry = rep.get(level)
            if not entry:
                cells.append(f"{'—':>22}")
                continue
            cells.append(f"{entry['ok']}/{entry['total']} "
                         f"({entry['ok_fraction']:.0%}) t{entry['transport']}/s{entry['server']}"
                         .rjust(22))
        lines.append(f"  {fmt_level(harness, level):>8}  " + "  ".join(cells))

    warnings = []
    for number in excluded:
        degraded = degraded_levels(per_rep[number - 1]) if number - 1 < len(per_rep) else []
        transport = sum(per_rep[number - 1][lv]["transport"] for lv in degraded)
        lines.append(f"  rep{number} EXCLUDED: levels {degraded} below "
                     f"{config.MIN_LEVEL_SUCCESS:.0%}")
        warnings.append(
            f"  rep{number} excluded from the medians: levels {degraded} lost requests "
            f"({transport} transport failures). "
            + ("Transport failures are the client breaking, not the server — run in-cluster "
               "with `main.py launch`." if transport else
               "Server-side failures: the server rejected load at these levels."))
    return lines, warnings


def fill_table(harness: str, fills: dict[int, dict]) -> tuple[list[str], list[str]]:
    """Report the in-flight depth each level actually achieved, and flag any shortfall.

    The concurrency column is a REQUEST, not a measurement. When a level's request count is
    below its concurrency and there is no duration floor, the surplus workers exit without
    issuing anything and the level caps at the request count while still being labelled with
    the concurrency it never reached.
    """
    if not fills:
        return [], []
    lines = ["", "ACHIEVED IN-FLIGHT DEPTH (time-averaged, per level).",
             "  The concurrency column is what was REQUESTED. This is what was APPLIED.",
             f"  {'level':>8} {'achieved':>10} {'peak':>7} {'requests':>10} {'verdict':>12}"]
    warnings = []
    for level in sorted(fills):
        fill = fills[level]
        verdict = "UNDER-FILLED" if fill["underfilled"] else "ok"
        lines.append(f"  {fmt_level(harness, level):>8} {fill['achieved']:>10.1f} "
                     f"{fill['peak']:>7} {fill['requests']:>10} {verdict:>12}")
        if fill["underfilled"]:
            warnings.append(
                f"  conc={level} applied only {fill['achieved']:.1f} in-flight "
                f"(peak {fill['peak']}, {fill['requests']} requests) — this row does NOT "
                f"measure concurrency={level}. Do not quote it as capacity at that level.")
    return lines, warnings


def raw_values_table(harness: str, label: str, levels: list[int],
                     reps: list[dict[int, dict]]) -> list[str]:
    """List every rep's raw value per cell, so a median can always be checked by hand."""
    lines = ["", "per-rep raw values:"]
    for level in levels:
        for name in COLS[harness]:
            values = [rep[level][name] for rep in reps if level in rep]
            lines.append(f"  {label}={fmt_level(harness, level):<9} {name:<16} {values}")
    return lines


def client_cpu_table(harness: str, label: str,
                     client_cpu: dict[int, list[float]]) -> tuple[list[str], list[str]]:
    """Report client CPU per level and flag levels that may be measuring the harness."""
    if not client_cpu:
        return [], []
    lines = ["", "client CPU utilization per level (child CPU seconds / wall seconds):",
             f"  >{CLIENT_CPU_SUSPECT} means the HARNESS was near a saturated core, so streaming",
             "  latency at that level may be measuring the client, not the server."]
    warnings = []
    for level in sorted(client_cpu):
        values = client_cpu[level]
        middle = statistics.median(values)
        mark = "   <-- SUSPECT" if middle > CLIENT_CPU_SUSPECT else ""
        lines.append(f"  {label}={fmt_level(harness, level):<9} median {middle:.2f}   "
                     f"values {[round(v, 2) for v in values]}{mark}")
        if middle > CLIENT_CPU_SUSPECT:
            warnings.append(f"  {label}={fmt_level(harness, level)} client CPU {middle:.2f} — "
                            f"cross-check TPOT against server-side aggregate tok/s before "
                            f"quoting this level")
    return lines, warnings


def preserve_existing(path: Path) -> Path | None:
    """
    Rename a result file already at this path so a new run cannot overwrite it.

    A result file is evidence, so it is moved aside rather than replaced: two runs can share a
    stem and still be different measurements. Shared with the launcher, which writes the same
    files on the in-cluster path, because one copy of this rule cannot drift from another.

    Args:
        path: The result file about to be written

    Returns:
        Where the previous file was moved, or None when there was nothing to preserve
    """
    if not path.exists():
        return None
    superseded = path.with_name(f"{path.stem}.superseded-{utc_stamp('%Y%m%dT%H%M%SZ')}{path.suffix}")
    path.rename(superseded)
    print(f"previous result moved aside -> {superseded}")
    return superseded


def build_aggregate(args, protocol: dict, lock: list[str], reps: list[dict[int, dict]],
                    shapes: dict[int, dict], client_cpu: dict[int, list[float]],
                    fills: dict[int, dict] | None = None,
                    latency_reps: list[dict[int, dict]] | None = None,
                    success_reps: list[dict[int, dict]] | None = None,
                    excluded: list[int] | None = None,
                    targets: tuple[float, float] | None = None
                    ) -> tuple[list[str], list[str]]:
    """Assemble the aggregate report, returning its lines and the warnings to echo."""
    harness = protocol["harness"]
    # A protocol may re-key rows onto another axis, so it can override the column label.
    label = protocol.get("level_label", LEVEL_LABEL[harness])
    levels = sorted({level for rep in reps for level in rep})

    lines = ["\n".join(lock), "=" * 60,
             f"AGGREGATE over {len(reps)}/{args.reps} successful reps (median, with spread)",
             ""]
    if harness == "arrival_rate":
        lines += ["OPEN LOOP — target arrival rate, not in-flight concurrency.",
                  "Do NOT compare these figures against any concurrency-sweep result.", ""]

    table, warnings = metric_table(harness, label, levels, reps)
    lines += table
    if harness == "arrival_rate":
        target_lines, target_warnings = targets_section(levels, reps, targets
                                                        or latency_targets(None))
        lines += target_lines
        warnings += target_warnings
    success_lines, success_warnings = success_table(harness, success_reps or [],
                                                    excluded or [])
    lines += success_lines
    warnings += success_warnings
    fill_lines, fill_warnings = fill_table(harness, fills or {})
    lines += fill_lines
    warnings += fill_warnings
    lines += token_shape_table(harness, shapes)
    lines += raw_values_table(harness, label, levels, reps)
    # Stacked beneath the throughput tables, as its own clearly-labelled block.
    latency_lines, latency_warnings = latency_companion_table(latency_reps or [])
    lines += latency_lines
    warnings += latency_warnings
    cpu_lines, cpu_warnings = client_cpu_table(harness, label, client_cpu)
    lines += cpu_lines
    warnings += cpu_warnings

    if warnings:
        lines += ["", "VARIANCE / VALIDITY WARNINGS (run more reps or investigate):", *warnings]
    else:
        lines += ["", f"variance: all cells within {MAX_SPREAD:.0%} across reps; "
                      f"client never saturated."]
    return lines, warnings


# ---------------------------------------------------------------------------- #
#                               ENTRY POINT                                    #
# ---------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=sorted(config.PROTOCOLS))
    parser.add_argument("--instance", required=True,
                        help="full instance type, e.g. p6-b300.48xlarge. Not a family tag: "
                             "'g7e' once covered a 2xlarge, a 12xlarge and a 24xlarge, which "
                             "made three different measurements look like one.")
    parser.add_argument("--config", required=True,
                        help="server config tag, e.g. baseline|dspark|ep")
    parser.add_argument("--model-tag", default="kimik3")
    parser.add_argument("--quant", default="mxfp4")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default="kimi-k3", help="served-model-name")
    parser.add_argument("--out-dir", required=True,
                        help="results directory, normally docs/results. Required rather than "
                             "defaulted: a hardcoded default once wrote one instance's results "
                             "into every other instance's directory.")
    parser.add_argument("--pod", default=None,
                        help="pod to capture server provenance from. Auto-discovered when "
                             "omitted: the Running pod with a `vllm` container in "
                             "--namespace. Pass this only to disambiguate.")
    parser.add_argument("--no-provenance", action="store_true",
                        help="skip server-side provenance capture. The result then carries no "
                             "evidence of NCCL transport, KV sizing or GPU topology.")
    # Defaults to whatever kubectl points at, since a hardcoded cluster once misdirected capture.
    parser.add_argument("--k8s-context", default=None,
                        help="context used to capture server provenance. Defaults to the "
                             "current kubectl context.")
    # No default, since a hardcoded namespace once pointed discovery at an unrelated one.
    parser.add_argument("--namespace", default=None,
                        help="namespace used for pod discovery and provenance capture only. "
                             "Required unless --no-provenance is given.")
    parser.add_argument("--tokenizer", default=None,
                        help="HF repo id or local path of the served model's tokenizer, used "
                             "only to resolve the recorded wire input size from "
                             "docs/prompt_tokens/. Not the same as --model-id, which is the "
                             "served-model-name.")
    parser.add_argument("--allow-loopback-latency", action="store_true",
                        help="measure TTFT through a loopback endpoint anyway. Adds "
                             "~150-200 ms of tunnel setup to every TTFT sample; the result is "
                             "recorded as unpublishable for latency.")
    parser.add_argument("--no-latency", action="store_true",
                        help="skip the companion streaming-latency pass that a throughput "
                             "protocol runs after its sweep. Roughly halves GPU time, at the "
                             "cost of reporting no TTFT/TPOT/ITL for the run.")
    parser.add_argument("--no-profile", action="store_true",
                        help="serve generated prompts instead of the protocol's default "
                             "pinned profile. The input size then becomes implicit, so avoid "
                             "it for anything published.")
    parser.add_argument("--profile", default=None,
                        help="pinned prompt profile from app/bench/data/prompt_profiles/. Defaults to "
                             "the protocol's own profile. Overrides the "
                             "protocol's PROMPT_MODE/PROMPT_TOKENS and its MAX_TOKENS, "
                             "because a profile pins BOTH ends of the shape. Strongly "
                             "preferred for any published figure: the input becomes a "
                             "committed artifact.")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="skip the correctness gate. Results are then unverified: the run "
                             "cannot show the server was serving the intended configuration.")
    parser.add_argument("--trace", default=None,
                        help="repo-relative path to a timestamped trace for the trace-replay "
                             "protocols, e.g. app/bench/data/traces/agentx_32k_seed42.jsonl. "
                             "Required by trace_replay (which pins no trace); overrides the "
                             "protocol's committed default otherwise. Keep traces under "
                             "app/bench/data/traces/ so the launcher payload carries them.")
    parser.add_argument("--ttft-target", type=float, default=None, metavar="MS",
                        help="time-to-first-token target in ms for the open-loop protocol, "
                             "against which each arrival rate is reported. The shipped value "
                             "is a starting point, not a standard: pick the number your use "
                             "case actually needs, because the highest arrival rate a server "
                             "sustains is a property of this target as much as of the server. "
                             "Reaches an in-cluster run, which an exported TTFT_SLA_MS does not.")
    parser.add_argument("--tpot-target", type=float, default=None, metavar="MS",
                        help="time-per-output-token target in ms. 50 ms is one output token "
                             "every 50 ms, or 20 per second, which suits interactive and "
                             "robotics work and is strict for a large model on smaller GPUs. "
                             "Batch and summarisation workloads are usually far looser.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # Checked before the sweep, so a missing namespace cannot surface after the GPU time is spent.
    if not args.namespace and not args.no_provenance:
        parser.error("--namespace is required to capture server provenance. Pass the namespace "
                     "the server is running in, or --no-provenance to record the result without "
                     "any evidence of NCCL transport, KV sizing or GPU topology.")
    return args


def output_stem(args, protocol: dict, env: dict | None = None) -> str:
    """Build the result filename stem, encoding everything that identifies the measurement.

    Mirrors the recipe naming, <model>-<quant>-<instance>[-<config>]-<protocol>[-<profile>], so a
    result and the recipe that produced it share a stem and sort next to each other. This is why
    --instance takes the full instance type rather than a short tag: g7e alone once covered a
    2xlarge, a 12xlarge and a 24xlarge in the same directory.

    Open-loop results get an openloop suffix, because enforcing the separation in the filename is
    more reliable than a warning in a README nobody reads at analysis time. The profile is part of
    a result's identity too: the same protocol at two input sizes is two different measurements
    and must not overwrite itself.
    """
    parts = [args.model_tag, args.quant, args.instance]
    # The config is dropped when it merely repeats the quantisation, which it usually does.
    if args.config and args.config != args.quant:
        parts.append(args.config)
    parts.append(args.protocol)
    if args.profile:
        parts.append(args.profile)
    if protocol["harness"] == "arrival_rate":
        parts.append("openloop")
        # An overridden target is part of a result's identity, so it joins the stem only when it differs.
        shipped, actual = latency_targets(protocol["env"]), latency_targets(env)
        for label, was, now in (("ttft", shipped[0], actual[0]),
                                ("tpot", shipped[1], actual[1])):
            if now != was:
                parts.append(f"{label}{now:g}ms")
    return "-".join(parts)


def print_transport_remedy(args, rep_number: int, total_reps: int) -> None:
    """Print the exact in-cluster command for this run after a transport failure.

    The harness can say the connection broke, but only the runner knows the protocol, instance
    and tags to put in the replacement command. Printing it filled in removes the step where
    the reader has to work out what to run.
    """
    skipped = total_reps - rep_number
    print()
    print("=" * 72)
    print(f"RUN ABANDONED after rep {rep_number}/{total_reps} — the client's connections are "
          f"failing.")
    if skipped:
        print(f"  {skipped} further rep(s) skipped. They would measure the same broken "
              f"connection.")
    print()
    print("  Re-run in-cluster. Fill in --context/--namespace/--service for your cluster:")
    print()
    forwarded = [
        f"--protocol {args.protocol}", f"--instance {args.instance}",
        f"--config {args.config}", f"--model-tag {args.model_tag}",
        f"--quant {args.quant}", f"--model-id {args.model_id}",
        f"--reps {args.reps}",
    ]
    if args.profile:
        forwarded.append(f"--profile {args.profile}")
    if args.tokenizer:
        forwarded.append(f"--tokenizer {args.tokenizer}")
    if args.out_dir:
        forwarded.append(f"--out-dir {args.out_dir}")
    print("    python3 app/bench/src/main.py launch \\")
    print(f"        --context {args.k8s_context} --namespace {args.namespace} "
          f"--service <SERVICE> \\")
    for index, chunk in enumerate(_chunk(forwarded, 3)):
        tail = " \\" if index < (len(forwarded) - 1) // 3 else ""
        print(f"        {' '.join(chunk)}{tail}")
    print()
    print("  Find the Service name with:")
    print(f"    kubectl --context {args.k8s_context} -n {args.namespace} get svc")
    print("=" * 72)


def _chunk(items: list[str], size: int):
    """Yield successive slices of `size` items, for readable multi-line command output."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def run_accuracy_gate(args) -> str:
    """Run or explicitly skip the correctness gate, returning what to record in the lock."""
    if args.skip_accuracy:
        print("  accuracy gate SKIPPED (--skip-accuracy) — results are UNVERIFIED")
        return "SKIPPED (--skip-accuracy) — results unverified"
    return accuracy_gate(args.endpoint, args.model_id, args.model_tag, args.config)


def main() -> int:
    """Run the selected protocol for the requested number of reps and write the results."""
    args = parse_args()
    protocol = config.PROTOCOLS[args.protocol]
    # Resolved before anything reads it, so env, lock and filename agree on the profile served.
    args.profile = resolve_profile(args, protocol)
    profile = load_profile_doc(args.profile)
    out_dir = Path(args.out_dir)
    # LoadGen's logs go to scratch, outside the results tree that holds one file per run.
    scratch = Path(tempfile.mkdtemp(prefix="bench-scratch-"))
    # Resolved before the stem, because an overridden latency target belongs in the filename.
    env = build_env(args, protocol, profile)
    stem = output_stem(args, protocol, env)

    # Recorded before the lock so it reports measured sizes, and only when --tokenizer was given.
    if args.tokenizer and args.profile:
        print(f"  {verify_profiles.ensure_counts(args.tokenizer)}")

    lock = build_lock(args, protocol, profile, env)
    print("\n".join(lock))

    # A throughput sweep is non-streaming, so TTFT/TPOT/ITL come from a second pass in its own table.
    want_latency = protocol["harness"] == "throughput" and not args.no_latency

    if args.dry_run:
        print(f"[dry-run] would write {out_dir}/{stem}.txt")
        print(f"[dry-run] command: python3 -u {config.HARNESS_PATHS[protocol['harness']]}")
        if want_latency:
            print(f"[dry-run] plus companion latency pass over "
                  f"{swept_levels(protocol)} per rep")
            print(f"[dry-run] command: python3 -u {config.HARNESS_PATHS['latency']}")
        return 0

    refuse_loopback_latency(args, protocol, want_latency)
    loopback = any(h in args.endpoint for h in config.LOOPBACK_HOSTS)
    lock.append("  transport     : " + (
        "LOOPBACK tunnel — connections may drop mid-run; TTFT carries tunnel setup and is "
        "NOT publishable" if loopback else "direct to Service over cluster DNS"))

    preflight(args.endpoint, args.model_id)
    # Recorded in every rep file, so an unverified run stays identifiable after the fact.
    lock.append(f"  accuracy gate : {run_accuracy_gate(args)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    per_rep: list[dict[int, dict]] = []
    latency_per_rep: list[dict[int, dict]] = []
    success_per_rep: list[dict[int, dict]] = []
    shapes: dict[int, dict] = {}
    fills: dict[int, dict] = {}
    client_cpu: dict[int, list[float]] = {}
    for rep_number in range(1, args.reps + 1):
        print(f"\n=== rep {rep_number}/{args.reps} — {args.protocol} / {args.config} ===")
        # Re-check before every rep, since a server can disappear inside an hours-long run.
        if rep_number > 1 and not server_alive(args.endpoint, args.model_id):
            print(f"  !! server no longer serving {args.model_id!r} at rep {rep_number}. "
                  f"Stopping and aggregating the {rep_number - 1} completed rep(s); a rep "
                  f"measured against a vanished server is not a measurement.")
            break
        # Distinct seed per rep: different prompts, still reproducible.
        env["PROMPT_SEED"] = str(1337 + rep_number)
        rep = run_rep(args, protocol, env, scratch, rep_number, client_cpu)
        if rep_number == 1:
            lock += harness_self_id_lines(rep)
        shapes.update(rep.shapes)
        fills.update(rep.fills)
        if want_latency:
            latency_per_rep.append(run_latency_companion(protocol, env, rep))

        # One results file per run; per-rep numbers live in its raw-values table, not in separate files.
        if not rep.rows:
            print(f"  !! rep {rep_number} produced no parsable summary rows. "
                  f"Harness output follows:")
            for body in rep.bodies:
                print(body)
        per_rep.append(rep.rows)
        success_per_rep.append(rep.success)
        degraded = degraded_levels(rep.success)
        if degraded:
            print(f"  !! rep {rep_number} EXCLUDED: levels {degraded} below "
                  f"{config.MIN_LEVEL_SUCCESS:.0%} request success")
        # Stop after the first transport-broken rep, since continuing measures the same dead connection.
        if any(rep.success[level]["transport"] for level in degraded):
            print_transport_remedy(args, rep_number, args.reps)
            break

    # A rep that lost requests is dropped and named in the aggregate rather than silently included.
    excluded = [n for n, success in enumerate(success_per_rep, start=1)
                if degraded_levels(success)]
    good = [rows for n, rows in enumerate(per_rep, start=1)
            if rows and n not in excluded]
    if not good:
        print("\nFAILED: no rep produced usable results "
              f"({len(excluded)} excluded for request failures). Nothing aggregated.")
        return 1

    lines, warnings = build_aggregate(args, protocol, lock, good, shapes, client_cpu, fills,
                                      [rows for rows in latency_per_rep if rows],
                                      success_per_rep, excluded,
                                      targets=latency_targets(env))
    # Provenance is a section of the one results file, so a result cannot be read without it.
    lines += provenance_lines(args)

    results_path = out_dir / f"{stem}.txt"
    preserve_existing(results_path)
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nresults -> {results_path}")
    for warning in warnings:
        print("WARN" + warning)
    return 0


def current_kube_context() -> str | None:
    """The context kubectl is pointed at, so provenance follows the cluster actually in use."""
    found = shell_output(["kubectl", "config", "current-context"]).strip()
    return found or None


def provenance_lines(args) -> list[str]:
    """Return the provenance section, or a line stating why there isn't one.

    Advisory: the measurement is already taken by the time this runs, so a missing pod must not
    discard a good result. The reason is recorded IN the results file rather than only printed,
    so a result without provenance is never silently unexplained.
    """
    if args.no_provenance:
        return ["", PROVENANCE_SKIPPED]
    if not args.k8s_context:
        args.k8s_context = current_kube_context()
        if not args.k8s_context:
            return ["", "SERVER PROVENANCE: not captured — no --k8s-context given and "
                        "`kubectl config current-context` returned nothing."]
    pod, why = (args.pod, f"--pod {args.pod}") if args.pod \
        else discover_pod(args.k8s_context, args.namespace)
    if not pod:
        print(f"  provenance NOT captured: {why}")
        return ["", f"SERVER PROVENANCE: not captured — {why}."]
    print(f"  provenance pod: {why}")
    return collect_provenance(pod, args.k8s_context, args.namespace)


if __name__ == "__main__":
    sys.exit(main())
