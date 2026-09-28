#!/usr/bin/env python3
"""Periodic status report for a queue of benchmark lanes: what is measuring, what is deploying,
what is queued, and an approximate ETA for each.

A lane is one model server plus the result file its measurement will write. Lanes come from a
local file (see --lanes) so no cluster name lives in this script. Each line is:

    name | kube-context | namespace | deployment | result-glob | after | deploy_min | measure_min

`after` names the lane this one waits on ("-" for none). deploy_min and measure_min are the
expected minutes for a queued lane; a measuring lane's ETA comes from the server itself.

Progress for a measuring lane is read from the vLLM counter vllm:request_success_total, summed
over every pod of the deployment, against the fixed schedule of the `corpus` protocol: a
throughput pass then a streaming latency pass, each 100/104/128/256/512/1024 requests at
concurrency 1/8/32/128/256/512. The suite logs nothing during the throughput pass, so the
server's own counter is the only live progress signal. The ETA fits this run's speed to the
schedule's known per-level shape (see RATE_SHAPE and fitted_eta_s).

Run once with --once, or leave it looping (default every 15 min). It writes the latest report
to --out and raises a macOS notification whenever a lane changes state.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

LEVELS = [1, 8, 32, 128, 256, 512]
REQUESTS = [100, 104, 128, 256, 512, 1024]
# Throughput pass then latency pass; one segment per (pass, level).
SEGMENTS = [(p, c, n) for p in ("throughput", "latency") for c, n in zip(LEVELS, REQUESTS)]
TOTAL_REQUESTS = sum(n for _, _, n in SEGMENTS)
KUBECTL_TIMEOUT_S = 20


# ------------------------------------------------------------------ lanes and state

def load_lanes(path: str) -> list[dict]:
    """Parse the lanes file, failing loudly on a malformed line rather than skipping it."""
    lanes = []
    for number, raw in enumerate(open(path), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split("|")]
        if len(fields) != 8:
            sys.exit(f"{path}:{number}: expected 8 '|'-separated fields, got {len(fields)}")
        name, ctx, ns, app, result_glob, after, deploy_min, measure_min = fields
        lanes.append({"name": name, "ctx": ctx, "ns": ns, "app": app, "glob": result_glob,
                      "after": None if after == "-" else after,
                      "deploy_min": float(deploy_min), "measure_min": float(measure_min)})
    return lanes


def load_state(path: str) -> dict:
    """Read the previous poll's lane states; an absent file is a fresh start."""
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def save_state(path: str, state: dict) -> None:
    """Persist lane states atomically so an interrupted write cannot corrupt them."""
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(state, handle)
    os.replace(tmp, path)


# ------------------------------------------------------------------ cluster reads

class ClusterUnreachable(Exception):
    """kubectl failed for a reason other than a missing object: auth, network, or timeout."""


def kubectl(ctx: str, ns: str, *args: str) -> str | None:
    """Run kubectl and return stdout; None only when the object does not exist.

    Any other failure raises ClusterUnreachable, so an expired credential is reported as
    such instead of reading as "not deployed yet". kubectl runs in its own process group
    and the whole group is killed on timeout: its exec credential plugin (aws eks get-token)
    otherwise keeps the output pipe open and the poll blocks forever.
    """
    cmd = ["kubectl", "--context", ctx, "-n", ns, "--request-timeout=15s", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=KUBECTL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise ClusterUnreachable(f"{ctx}: kubectl timed out")
    if proc.returncode == 0:
        return out
    if "NotFound" in err or "not found" in err:
        return None
    raise ClusterUnreachable(f"{ctx}: {err.strip().splitlines()[-1] if err.strip() else 'kubectl failed'}")


def deployment_info(lane: dict) -> dict | None:
    """Return the deployment's creation time and ready count, or None if it does not exist."""
    raw = kubectl(lane["ctx"], lane["ns"], "get", "deployment", lane["app"], "-o", "json")
    if raw is None:
        return None
    d = json.loads(raw)
    return {"created": d["metadata"]["creationTimestamp"],
            "replicas": d["spec"].get("replicas", 1),
            "ready": d.get("status", {}).get("readyReplicas", 0) or 0}


def running_bench_job(lane: dict) -> dict | None:
    """Return the running benchmark Job aimed at this lane's Service, if there is one."""
    raw = kubectl(lane["ctx"], lane["ns"], "get", "jobs", "-o", "json")
    if raw is None:
        return None
    for job in json.loads(raw)["items"]:
        name = job["metadata"]["name"]
        active = job.get("status", {}).get("active", 0)
        if not name.startswith("bench-suite-") or not active:
            continue
        if f"//{lane['app']}." in json.dumps(job["spec"]["template"]["spec"]):
            return {"name": name, "started": job["status"].get("startTime")}
    return None


def bucket_count(metrics: str, le: str) -> float:
    """Finished requests in the vllm:request_params_max_tokens histogram at bucket `le`."""
    pattern = r'^vllm:request_params_max_tokens_bucket\{[^}]*le="' + re.escape(le) + r'"[^}]*\}\s+([0-9.eE+]+)'
    return sum(float(m) for m in re.findall(pattern, metrics, re.M))


def completed_requests(lane: dict) -> tuple[int, int] | None:
    """Return (measurement requests finished, all requests finished), summed over the deployment.

    Measurement requests are the ones asking for more than 500 output tokens: the corpus protocol
    on this profile asks for 1,024, while the accuracy reference and gate (256) and a TTFT probe
    (128) ask for fewer. Counting by size rather than subtracting a fixed setup allowance keeps the
    progress figure right however many setup requests a run happened to make.
    """
    raw = kubectl(lane["ctx"], lane["ns"], "get", "pods", "-l", f"app={lane['app']}",
                  "-o", "jsonpath={.items[*].metadata.name}")
    if not raw:
        return None
    measured, total = 0.0, 0.0
    for pod in raw.split():
        metrics = kubectl(lane["ctx"], lane["ns"], "exec", pod, "-c", "vllm", "--",
                          "curl", "-s", "localhost:8000/metrics")
        if metrics is None:
            return None
        everything = bucket_count(metrics, "+Inf")
        measured += everything - bucket_count(metrics, "500.0")
        total += everything
    return int(measured), int(total)


def init_progress(lane: dict) -> str:
    """Describe what a not-yet-ready pod is doing: syncing weights, or loading the model."""
    raw = kubectl(lane["ctx"], lane["ns"], "get", "pods", "-l", f"app={lane['app']}", "-o", "json")
    if raw is None:
        return "unknown"
    pods = json.loads(raw)["items"]
    if not pods:
        return "no pod yet (waiting for the node)"
    status = pods[0].get("status", {})
    for c in status.get("initContainerStatuses", []):
        if not c.get("ready") and "running" in c.get("state", {}):
            return "syncing weights from S3"
        if "waiting" in c.get("state", {}):
            return f"init {c['state']['waiting'].get('reason', 'waiting')}"
    return "loading model / compiling graphs"


# ------------------------------------------------------------------ progress and ETA

# Requests/s at each concurrency relative to concurrency 1, averaged over two complete corpus
# runs on this profile (Muse on one g7e card: 1, 5.0, 8.8, 10, 10, 10; Nemotron on B200 TP8:
# 1, 4.6, 8.4, 11.2, 11.7, 11.7). Both saturate by concurrency 128, so the shape carries across
# servers of very different speed; only the scale differs, and the scale is fitted live.
RATE_SHAPE = {1: 1.0, 8: 4.8, 32: 8.6, 128: 10.6, 256: 10.9, 512: 10.9}
# Job time before the first measured request: dependency install plus the accuracy gate.
SETUP_S = 180


def unit_time(done: int) -> float:
    """Seconds to complete `done` scheduled requests on a server whose concurrency-1 rate is 1 req/s."""
    seconds, left = 0.0, done
    for _, level, n in SEGMENTS:
        take = min(left, n)
        seconds += take / RATE_SHAPE[level]
        left -= take
        if left <= 0:
            break
    return seconds


def fitted_eta_s(done: int, elapsed_s: float) -> float | None:
    """Fit this run's speed from progress so far and project the rest of the schedule.

    Time is linear in 1/speed, so speed = unit_time(done) / elapsed, and the remaining
    time is (unit_time(total) - unit_time(done)) / speed. Fitting against the whole
    schedule rather than the latest window keeps slow low-concurrency levels from being
    extrapolated onto the fast saturated ones.
    """
    if done <= 0 or elapsed_s <= 0:
        return None
    speed = unit_time(done) / elapsed_s
    return (unit_time(TOTAL_REQUESTS) - unit_time(done)) / speed


def locate(done: int) -> int:
    """Map completed measurement requests to the index of the segment in progress."""
    edge = 0
    for index, (_, _, n) in enumerate(SEGMENTS):
        edge += n
        if done < edge:
            return index
    return len(SEGMENTS) - 1


# ------------------------------------------------------------------ per-lane status

def parse_ts(stamp: str | None) -> float:
    """Parse a Kubernetes RFC 3339 timestamp to epoch seconds, 0 when absent."""
    if not stamp:
        return 0.0
    return dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def result_file(lane: dict, results_dir: str) -> str | None:
    """Return the newest result file matching the lane's glob."""
    matches = sorted(glob.glob(os.path.join(results_dir, lane["glob"])), key=os.path.getmtime)
    return matches[-1] if matches else None


def measuring_status(lane: dict, job: dict, now: float) -> dict:
    """Progress and ETA for a lane whose benchmark Job is running."""
    counts = completed_requests(lane)
    if counts is None:
        return {"state": "MEASURING", "detail": f"job {job['name']} (server metrics unreadable)",
                "eta_s": lane["measure_min"] * 60}
    measured, total = counts
    if measured == 0:
        return {"state": "GATING", "detail": f"accuracy gate / setup ({total} short requests so far)",
                "eta_s": lane["measure_min"] * 60}
    done = min(measured, TOTAL_REQUESTS)
    phase, level, _ = SEGMENTS[locate(done)]
    pct = 100.0 * done / TOTAL_REQUESTS
    detail = f"{phase} pass, concurrency {level}: {done}/{TOTAL_REQUESTS} requests ({pct:.0f}%)"
    elapsed = now - parse_ts(job["started"]) - SETUP_S
    eta = fitted_eta_s(done, elapsed)
    return {"state": "MEASURING", "detail": detail,
            "eta_s": eta if eta is not None else lane["measure_min"] * 60}


def lane_status(lane: dict, results_dir: str, now: float) -> dict:
    """Classify a lane as DONE, MEASURING, DEPLOYING, or QUEUED, with detail and ETA."""
    done_file = result_file(lane, results_dir)
    if done_file:
        return {"state": "DONE", "detail": os.path.basename(done_file), "eta_s": 0.0}
    job = running_bench_job(lane)
    if job:
        return measuring_status(lane, job, now)
    info = deployment_info(lane)
    if info is None:
        return {"state": "QUEUED", "detail": f"after {lane['after']}" if lane["after"] else "not started", "eta_s": None}
    elapsed = now - parse_ts(info["created"])
    ready = f"{info['ready']}/{info['replicas']} ready"
    if info["ready"] >= info["replicas"]:
        return {"state": "DEPLOYING", "detail": f"{ready}; server up, gate/measurement starting",
                "eta_s": lane["measure_min"] * 60}
    left = max(0.0, lane["deploy_min"] * 60 - elapsed)
    stalled = " - LONGER THAN EXPECTED, check it" if elapsed > 2 * lane["deploy_min"] * 60 else ""
    return {"state": "DEPLOYING",
            "detail": f"{ready}, {init_progress(lane)}, {elapsed / 60:.0f} min in{stalled}",
            "eta_s": left + lane["measure_min"] * 60}


def chain_etas(lanes: list[dict], statuses: dict) -> None:
    """Give each queued lane an ETA: its predecessor's ETA plus its own deploy and measure time."""
    for lane in lanes:
        status = statuses[lane["name"]]
        if status["state"] != "QUEUED":
            continue
        before = statuses.get(lane["after"], {}).get("eta_s") if lane["after"] else 0.0
        if before is not None:
            status["eta_s"] = before + (lane["deploy_min"] + lane["measure_min"]) * 60


# ------------------------------------------------------------------ report and notify

def fmt_eta(seconds: float | None, now: float, tz_label: str, offset_h: float) -> str:
    """Render an ETA as a duration plus a local clock time."""
    if seconds is None:
        return "ETA unknown"
    if seconds <= 0:
        return "done"
    clock = dt.datetime.fromtimestamp(now + seconds, dt.timezone.utc) + dt.timedelta(hours=offset_h)
    hours, minutes = divmod(int(seconds // 60), 60)
    span = f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"
    return f"~{span} (about {clock:%H:%M} {tz_label})"


def render(lanes: list[dict], statuses: dict, now: float, tz_label: str, offset_h: float,
           block_end: str | None) -> str:
    """Build the text report: one line per lane in queue order, then what runs next."""
    local = dt.datetime.fromtimestamp(now, dt.timezone.utc) + dt.timedelta(hours=offset_h)
    lines = [f"BENCHMARK STATUS  {local:%Y-%m-%d %H:%M} {tz_label}", ""]
    for lane in lanes:
        s = statuses[lane["name"]]
        lines.append(f"  {s['state']:<10} {lane['name']:<22} {fmt_eta(s['eta_s'], now, tz_label, offset_h)}")
        lines.append(f"             {s['detail']}")
    queued = [l["name"] for l in lanes if statuses[l["name"]]["state"] == "QUEUED"]
    lines += ["", f"  Next up: {queued[0]}" if queued else "  Next up: nothing queued"]
    if block_end:
        left_h = (parse_ts(block_end) - now) / 3600
        lines.append(f"  Capacity block ends in {left_h:.1f} h")
    lines.append("  ETAs are approximate; a measuring lane's ETA sharpens as it reaches high concurrency.")
    return "\n".join(lines)


def notify_changes(lanes: list[dict], statuses: dict, state: dict) -> None:
    """Raise a macOS notification for every lane whose state changed since the last poll."""
    previous = state.setdefault("_states", {})
    for lane in lanes:
        now_state = statuses[lane["name"]]["state"]
        if previous.get(lane["name"]) not in (None, now_state):
            message = f"{lane['name']}: {previous[lane['name']]} -> {now_state}"
            subprocess.run(["osascript", "-e", f'display notification "{message}" with title "Benchmarks"'],
                           capture_output=True)
        previous[lane["name"]] = now_state


def poll(args) -> str:
    """Take one sample of every lane and return the rendered report."""
    lanes = load_lanes(args.lanes)
    state = load_state(args.state)
    now = time.time()
    statuses = {}
    for lane in lanes:
        try:
            statuses[lane["name"]] = lane_status(lane, args.results_dir, now)
        except ClusterUnreachable as why:
            statuses[lane["name"]] = {"state": "UNKNOWN", "eta_s": None,
                                      "detail": f"cluster unreachable ({why}); refresh AWS credentials?"}
    chain_etas(lanes, statuses)
    notify_changes(lanes, statuses, state)
    save_state(args.state, state)
    report = render(lanes, statuses, now, args.tz_label, args.tz_offset, args.block_end)
    extras = [open(path).read().strip() for path in args.extra if os.path.exists(path)]
    if extras:
        report += "\n\n  Also running:\n" + "\n".join(f"    {line}" for text in extras for line in text.splitlines())
    with open(args.out, "w") as handle:
        handle.write(report + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lanes", required=True, help="lanes file (keep it out of git; it names clusters)")
    parser.add_argument("--results-dir", default="docs/results")
    parser.add_argument("--out", default="docs/results/bench-status.txt")
    # .txt so the existing docs/results/*.txt ignore rule covers it; the content is JSON.
    parser.add_argument("--state", default="docs/results/bench-status-state.txt")
    parser.add_argument("--interval-min", type=float, default=15)
    parser.add_argument("--tz-label", default="UTC")
    parser.add_argument("--tz-offset", type=float, default=0.0, help="hours from UTC, e.g. -7 for MST")
    parser.add_argument("--block-end", help="capacity block end, RFC 3339 UTC, e.g. 2026-09-25T11:30:00Z")
    parser.add_argument("--extra", action="append", default=[],
                        help="file whose current contents are appended to each report (repeatable)")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        print(poll(args), flush=True)
        if args.once:
            return
        print(f"\n--- next report in {args.interval_min:.0f} min ---\n", flush=True)
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
