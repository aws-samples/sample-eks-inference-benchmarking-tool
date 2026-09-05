#!/usr/bin/env python3
"""Run the benchmark suite in-cluster against the vLLM Service over cluster DNS, which is the required
path for any figure that reports TTFT. Through a port-forward, first byte on a fresh connection
measured 276 ms against 90 ms on a warm one, and the open-loop harness opens a connection per
arrival, so roughly 185 ms of tunnel setup lands in every TTFT sample, most of a 500 ms SLA budget;
TPOT and tokens per second are not materially affected, but TTFT is. The Job runs the suite rather
than a bare harness because the suite owns reps, medians, spread flags, the protocol lock and the
accuracy gate, which also means the companion streaming-latency pass runs in-cluster automatically.
Anything this does not recognise is forwarded verbatim to main.py suite, so protocol, profile and
gate flags need no duplicate definition here."""
from __future__ import annotations

import argparse
import base64
import io
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import config
import verify_profiles
from stats import shell_result, utc_stamp

# Job manifest templated here, with the suite's flags travelling as one string not one env var each.
JOB_TEMPLATE = config.PROJECT_ROOT / "app" / "bench" / "suite-job.yaml"

# Only what the suite and gate read, listed so the payload cannot silently outgrow the ConfigMap.
PAYLOAD_PATHS = ("app/bench", "docs/accuracy", "docs/prompt_tokens")

# ConfigMap objects cap at 1 MiB, so refuse below that rather than fail obscurely at apply time.
PAYLOAD_LIMIT_B = 900_000

# Results come back between these markers in stdout, so nothing depends on the pod surviving.
RESULT_BLOCK = re.compile(
    r"^===== BEGIN RESULT FILE: (.+?) =====$(.*?)^===== END RESULT FILE: \1 =====$",
    re.M | re.S)
EXIT_CODE = re.compile(r"runner exit code: (\d+)")


# ---------------------------------------------------------------------------- #
#                                  KUBECTL                                     #
# ---------------------------------------------------------------------------- #

def kubectl_base(context: str, namespace: str) -> list[str]:
    """Build the kubectl prefix used for every call."""
    return ["kubectl", "--context", context, "-n", namespace]


def kubectl(context: str, namespace: str, *args: str, timeout: int = 120) -> tuple[str, str]:
    """Run one kubectl command, returning (stdout, error text)."""
    return shell_result([*kubectl_base(context, namespace), *args], timeout=timeout)


def require_service(context: str, namespace: str, service: str) -> None:
    """Exit unless the target Service exists, before anything is created."""
    _, error = kubectl(context, namespace, "get", "svc", service)
    if error:
        sys.exit(f"PREFLIGHT FAIL: Service {service!r} not found in {namespace}: {error}")


# ---------------------------------------------------------------------------- #
#                               HARNESS PAYLOAD                                #
# ---------------------------------------------------------------------------- #

def pack_harness(repo: Path) -> bytes:
    """Tar and base64 the harness tree, returning the encoded bytes.

    Built in memory with tarfile rather than by shelling out to tar: BSD tar on macOS treats
    trailing options as filenames, so the shell version this replaces needed --exclude ordered
    before the paths and broke subtly if that was disturbed.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in PAYLOAD_PATHS:
            source = repo / relative
            if not source.exists():
                sys.exit(f"payload path missing: {relative}")
            archive.add(source, arcname=relative, filter=_skip_caches)
    return base64.b64encode(buffer.getvalue())


def _skip_caches(entry: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Drop bytecode caches from the payload."""
    name = Path(entry.name).name
    if "__pycache__" in entry.name or name.endswith((".pyc", ".pyo")):
        return None
    return entry


def harness_revision(repo: Path) -> str:
    """Report the harness revision, flagging uncommitted changes.

    Passed into the Job because a pod has no .git, and without it the protocol lock records
    the revision as "unknown" and the result cannot be tied to a commit.
    """
    rev, error = shell_result(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    revision = rev.strip() if not error else ""
    if not revision:
        return "unknown"
    dirty, _ = shell_result(["git", "status", "--porcelain", *PAYLOAD_PATHS], cwd=repo)
    return revision + ("-dirty" if dirty.strip() else "")


# ---------------------------------------------------------------------------- #
#                                 JOB RENDERING                                #
# ---------------------------------------------------------------------------- #

def suite_args_string(forwarded: list[str], endpoint: str,
                      out_dir: bool = True) -> str:
    """Join the forwarded suite flags into the single string the Job expands.

    Refuses whitespace inside an argument: the Job expands this unquoted in bash, so an
    argument containing a space would split into two and silently change the run.
    """
    args = [*forwarded, "--endpoint", endpoint]
    if out_dir:
        args += ["--out-dir", "/work/out"]
    bad = [a for a in args if any(c.isspace() for c in a)]
    if bad:
        sys.exit(f"cannot forward arguments containing whitespace: {bad}")
    return " ".join(args)


def extra_requirements(forwarded: list[str]) -> str:
    """Resolve the extra requirements file for the forwarded protocol's harness, or ''.

    Read from config rather than guessed from flags, so a protocol whose harness wraps an
    external tool (aiperf) gets its pinned dependencies installed in the Job without the
    Job template hardcoding them.
    """
    if "--protocol" not in forwarded:
        return ""
    protocol = config.PROTOCOLS.get(forwarded[forwarded.index("--protocol") + 1])
    if not protocol:
        return ""  # the suite will reject the unknown protocol with its own message
    return config.HARNESS_EXTRA_REQUIREMENTS.get(protocol["harness"], "")


def render_job(job_name: str, namespace: str, configmap: str, suite_args: str,
               revision: str, run_command: str = "suite",
               extra_reqs: str = "") -> str:
    """Substitute the Job template's placeholders."""
    if not JOB_TEMPLATE.exists():
        sys.exit(f"Job template missing: {JOB_TEMPLATE}")
    text = JOB_TEMPLATE.read_text()
    for placeholder, value in (("__JOB_NAME__", job_name),
                               ("__NAMESPACE__", namespace),
                               ("__CONFIGMAP__", configmap),
                               ("__RUN_COMMAND__", run_command),
                               ("__SUITE_ARGS__", suite_args),
                               ("__EXTRA_REQUIREMENTS__", extra_reqs),
                               ("__HARNESS_GIT_REV__", revision)):
        text = text.replace(placeholder, value)
    leftover = sorted(set(re.findall(r"__[A-Z_]+__", text)))
    if leftover:
        sys.exit(f"Job template has unsubstituted placeholders: {leftover}")
    return text


# ---------------------------------------------------------------------------- #
#                                   RUNNING                                    #
# ---------------------------------------------------------------------------- #

def apply_manifest(context: str, namespace: str, manifest: str) -> None:
    """Apply a rendered manifest from stdin."""
    done = subprocess.run([*kubectl_base(context, namespace), "apply", "-f", "-"],
                          input=manifest, capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        sys.exit(f"failed to apply Job: {done.stderr.strip()}")


def wait_for_pod(context: str, namespace: str, job: str, timeout_s: int = 300) -> str:
    """Wait for the Job's pod to appear, returning its name."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out, _ = kubectl(context, namespace, "get", "pods", "-l", f"job-name={job}",
                         "-o", "jsonpath={.items[0].metadata.name}")
        if out.strip():
            return out.strip()
        time.sleep(5)
    sys.exit(f"no pod appeared for job/{job} within {timeout_s}s")


def stream_logs(context: str, namespace: str, pod: str) -> str:
    """Follow the pod's logs to the terminal, returning everything the RUN produced.

    Streamed rather than fetched at the end because a run takes hours and silence for that
    long is indistinguishable from a hang. But the stream is not the source of truth: on
    previous tests the API server sent an HTTP/2 GOAWAY roughly two hours into a run, the follow
    ended early, and the launcher extracted nothing from a Job that was still working — then
    deleted it, taking the only copy of the results with it.

    So the stream is for watching, and the authoritative copy is re-fetched afterwards with a
    plain (non-following) `kubectl logs`, which returns the pod's whole log regardless of how
    many times the follow broke. The follow is also resumed while the pod is still running,
    so a dropped connection does not turn the rest of a long run into silence.
    """
    captured: list[str] = []
    while True:
        process = subprocess.Popen(
            [*kubectl_base(context, namespace), "logs", "-f", pod],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)
        process.wait()
        phase, _ = kubectl(context, namespace, "get", "pod", pod,
                           "-o", "jsonpath={.status.phase}")
        if phase.strip() not in ("Running", "Pending"):
            break
        print(f"\n  [launcher] log stream ended while the pod is still {phase.strip()}; "
              f"resuming follow. The run itself is unaffected.\n", flush=True)
        time.sleep(5)

    # Authoritative read. --tail=-1 asks for the entire log rather than the default window.
    complete, error = kubectl(context, namespace, "logs", pod, "--tail=-1", timeout=600)
    if complete.strip():
        return complete
    if error:
        print(f"  [launcher] could not re-read the pod log ({error.strip()}); "
              f"falling back to what the stream captured")
    return "".join(captured)


def extract_results(log: str, out_dir: Path) -> list[Path]:
    """Write the files embedded in the Job log, returning the paths that landed.

    A name beginning `docs/accuracy/` is an accuracy reference and belongs in the repo, next to
    the others the gate reads. Everything else is a result and goes to --out-dir.
    """
    written = []
    for match in RESULT_BLOCK.finditer(log):
        name = match.group(1).strip()
        if name.startswith("docs/accuracy/"):
            destination = config.PROJECT_ROOT / name
        else:
            destination = out_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(match.group(2).lstrip("\n"), encoding="utf-8")
        print(f"  wrote {destination}")
        written.append(destination)
    return written


def fill_in_provenance(paths: list[Path], context: str, namespace: str,
                       pod_hint: str | None) -> None:
    """Capture server provenance locally and splice it into each extracted result.

    The suite normally does this itself, but it cannot from inside the Job: the runner image is
    python:3.12, with no kubectl and no credentials to use one, so every in-cluster result came
    back with an empty provenance block. This machine has both, and it has just used them to
    create, stream and clean up the Job.

    Nothing here touches the measurement. The Job has already exited and every rep is aggregated
    by this point, so the two kubectl calls cannot perturb what was measured; the provenance
    block is also the last section of the file, so substituting it cannot disturb the tables.
    """
    import suite

    pod, why = (pod_hint, f"--pod {pod_hint}") if pod_hint \
        else suite.discover_pod(context, namespace)
    if not pod:
        print(f"  WARNING: provenance NOT captured: {why}")
        print("           Results carry no KV cache sizing, GPU topology or NCCL transport")
        print("           evidence, so nothing proves the server was configured as claimed.")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace(suite.PROVENANCE_SKIPPED,
                                         f"SERVER PROVENANCE: not captured — {why}."),
                            encoding="utf-8")
        return

    block = "\n".join(suite.collect_provenance(pod, context, namespace)).lstrip("\n")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if suite.PROVENANCE_SKIPPED not in text:
            print(f"  WARNING: no provenance placeholder in {path.name}; left unchanged")
            continue
        path.write_text(text.replace(suite.PROVENANCE_SKIPPED, block), encoding="utf-8")
    print(f"  provenance captured from {pod} ({why}) and written into "
          f"{len(paths)} result file(s)")


def cleanup(context: str, namespace: str, job: str, configmap: str, keep: bool,
            extracted: int = -1, runner_exit: int = 0) -> None:
    """Delete the Job and its code ConfigMap unless asked to keep them.

    A run that produced no result files is kept regardless: the pod log is then the only
    copy of whatever the run did, and deleting it destroys the evidence needed to tell a
    failed measurement from a failed extraction. That distinction cost a two-hour run on
    previous tests, when a broken log stream looked identical to a failed benchmark.

    A run whose runner exited non-zero is kept for the same reason, and the file count alone
    does not catch it. Every run echoes the accuracy reference files, so the extractor writes
    several of them even when the benchmark never started, and the count never reaches zero.
    On 2026-08-13 a preflight failure was therefore cleaned up on the strength of five
    round-tripped references, and the pod log holding the actual reason had to be reproduced
    by launching the run a second time with --keep-job.
    """
    if extracted == 0 or runner_exit != 0:
        why = ("no result files were extracted" if extracted == 0
               else f"the runner exited {runner_exit}")
        print(f"  KEEPING job/{job} and configmap/{configmap}: {why}, so the pod log is the "
              f"only remaining copy of what this run did.\n"
              f"  Recover with:\n"
              f"    kubectl --context {context} -n {namespace} logs job/{job} --tail=-1 "
              f"> /tmp/{job}.log\n"
              f"  Then delete them by hand once you have what you need.")
        return
    if keep:
        print(f"  keeping job/{job} and configmap/{configmap} (--keep-job)")
        return
    kubectl(context, namespace, "delete", "job", job, "--ignore-not-found")
    kubectl(context, namespace, "delete", "configmap", configmap, "--ignore-not-found")
    print(f"  cleaned up job/{job} and configmap/{configmap}")


# ---------------------------------------------------------------------------- #
#                                 ENTRY POINT                                  #
# ---------------------------------------------------------------------------- #

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse this launcher's own flags, leaving the rest for the suite."""
    parser = argparse.ArgumentParser(
        prog="main.py launch", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", required=True, help="kubectl context")
    # No default, since this builds the endpoint, places the Job and finds the Service on every path.
    parser.add_argument("--namespace", required=True,
                        help="namespace holding the vLLM Service; the runner Job is created here "
                             "too")
    parser.add_argument("--service", required=True,
                        help="vLLM Service name; the endpoint is derived from it over "
                             "cluster DNS")
    parser.add_argument("--port", default="8000")
    parser.add_argument("--out-dir", required=True,
                        help="local directory for extracted results, normally docs/results, "
                             "which is flat: the filename carries model, quantisation and "
                             "instance. Required rather than defaulted, because a hardcoded "
                             "default once wrote one instance's results into another's.")
    parser.add_argument("--keep-job", action="store_true",
                        help="leave the Job and ConfigMap in place for debugging")
    parser.add_argument("--capture-accuracy-reference", action="store_true",
                        help="capture the accuracy reference for this server instead of "
                             "benchmarking it, and bring the JSON back into docs/accuracy/. "
                             "Runs in-cluster like a measurement, so it needs no port-forward. "
                             "Forward --mode, --model-tag, --config and --model-id through to it. "
                             "Deliberately a separate run from the benchmark: capturing and "
                             "checking in one pass would compare a server against itself.")
    parser.add_argument("--dry-run", action="store_true",
                        help="pack and render, create nothing. Needs no cluster access.")
    args, forwarded = parser.parse_known_args()
    # argparse's required accepts an empty string, and an unexported variable arrives as exactly that.
    for flag in ("context", "namespace", "service"):
        if not getattr(args, flag).strip():
            parser.error(f"--{flag} is empty. It is probably an unset environment variable — "
                         f"check EKS_CLUSTER_NAME and NAMESPACE are exported.")
    return args, forwarded


def main() -> int:
    """Launch the suite in-cluster and bring its results back."""
    args, forwarded = parse_args()
    repo = config.PROJECT_ROOT
    stamp = utc_stamp("%Y%m%d-%H%M%S")
    job = f"bench-suite-{stamp}"
    configmap = f"bench-code-{stamp}"
    endpoint = f"http://{args.service}.{args.namespace}.svc.cluster.local:{args.port}/v1"
    out_dir = Path(args.out_dir)

    revision = harness_revision(repo)

    # The Job has no kubectl, so it skips provenance and leaves a placeholder filled in from here.
    capturing = args.capture_accuracy_reference
    user_declined_provenance = "--no-provenance" in forwarded
    pod_hint = forwarded[forwarded.index("--pod") + 1] if "--pod" in forwarded else None
    if capturing:
        # accuracy.py writes the reference into the repo tree, returned through the same stdout markers.
        in_job_args = [*forwarded, "--write-reference"]
        suite_args = suite_args_string(in_job_args, endpoint, out_dir=False)
    else:
        in_job_args = forwarded if user_declined_provenance \
            else [*forwarded, "--no-provenance"]
        suite_args = suite_args_string(in_job_args, endpoint)

    print("=== preflight ===")
    print(f"  endpoint  : {endpoint}")
    print(f"  job       : {job}")
    print(f"  harness   : {revision}")
    print(f"  out dir   : {out_dir}")
    print(f"  suite args: {suite_args}")
    if revision.endswith("-dirty"):
        print("  WARNING: uncommitted harness changes; results record a dirty revision")

    # Counted before packing, since the Job image has no route to Hugging Face to measure it itself.
    tokenizer = forwarded[forwarded.index("--tokenizer") + 1] if "--tokenizer" in forwarded \
        else None
    if tokenizer:
        print(f"  {verify_profiles.ensure_counts(tokenizer)}")

    print("=== packing harness ===")
    payload = pack_harness(repo)
    print(f"  payload   : {len(payload) // 1024} KiB base64 "
          f"(limit {PAYLOAD_LIMIT_B // 1024} KiB)")
    if len(payload) > PAYLOAD_LIMIT_B:
        sys.exit("payload too large for a ConfigMap. Narrow PAYLOAD_PATHS.")

    manifest = render_job(job, args.namespace, configmap, suite_args, revision,
                          run_command="accuracy" if capturing else "suite",
                          extra_reqs=extra_requirements(forwarded))

    if args.dry_run:
        print("[dry-run] rendered Job and packed payload; created nothing")
        print(f"[dry-run] would extract results to {out_dir}")
        return 0

    require_service(args.context, args.namespace, args.service)

    payload_file = repo / f".bench-payload-{stamp}.b64"
    try:
        payload_file.write_bytes(payload)
        _, error = kubectl(args.context, args.namespace, "create", "configmap", configmap,
                           f"--from-file=code.tar.gz.b64={payload_file}")
        if error:
            sys.exit(f"failed to create ConfigMap: {error}")
    finally:
        payload_file.unlink(missing_ok=True)
    print(f"  configmap : {configmap}")

    print("=== launching ===")
    apply_manifest(args.context, args.namespace, manifest)
    pod = wait_for_pod(args.context, args.namespace, job)
    print(f"  pod       : {pod}")

    print("=== streaming ===")
    log = stream_logs(args.context, args.namespace, pod)

    print("=== extracting results ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = extract_results(log, out_dir)
    print(f"  {len(written)} result file(s) extracted")
    if not written:
        print("  WARNING: no result files in the Job log. The run likely failed; inspect the")
        print("           output above before treating this as a measurement.")

    # After the Job exits so it cannot influence the measurement, and while the server is still up.
    if capturing:
        print("  accuracy reference captured; run the benchmark as a separate launch")
    elif written and not user_declined_provenance:
        print("=== capturing server provenance ===")
        fill_in_provenance(written, args.context, args.namespace, pod_hint)
    elif written:
        print("  provenance skipped (--no-provenance); results say so explicitly")

    match = EXIT_CODE.search(log)
    code = int(match.group(1)) if match else 1
    print(f"=== runner exit code: {code} ===")
    cleanup(args.context, args.namespace, job, configmap, args.keep_job, len(written), code)
    return code


if __name__ == "__main__":
    sys.exit(main())
