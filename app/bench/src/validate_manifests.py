#!/usr/bin/env python3
"""Prove a vLLM serve recipe passes the flags it is written to pass, before a deploy trusts it, scoped
to the recipe manifests and never to the deliberately unrunnable examples. This is the check that
was missing when a stray comment inside a backslash continuation silently truncated a vllm serve
command and the server came up healthy serving traffic at the wrong precision with both parsers
dropped, which bash -n cannot catch because the truncated command is still valid shell. Every
recipe gets a shell syntax check, a scan for that comment signature, a comparison of the argv that
actually reaches vLLM with the script re-run against stubbed binaries, a check that the Service
forwards to the port vLLM binds, and a check for any variable the script reads that nothing
defines. Values for the stubbed run come from each manifest's own env block, so a recipe is checked
with its own numbers, and it exits non-zero if anything fails so it can gate a deploy."""
from __future__ import annotations

import argparse
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

import config

REQUIRED_FLAGS = config.REQUIRED_FLAGS

# Platform-injected variables, given arbitrary non-empty values since only which flags arrive matters.
INJECTED_STUBS = {
    "HOSTNAME": "stub-leader-0",
    "LEADER": "stub-leader-0",
    "LWS_LEADER_ADDRESS": "stub-leader-0",
    "LWS_WORKER_INDEX": "0",
    "POD_NAMESPACE": "stub-namespace",
    "RANK": "0",
}

# Hardware and vLLM itself, replaced so the script reaches its serve invocation with no GPU.
SHELL_STUBS = """
nvidia-smi() { echo "GPU0 NV18 NV18"; return 0; }
# The recipes probe `vllm serve --help=all` to confirm a flag exists before passing it. Report
# every flag as supported so the probe does not prune arguments the manifest means to pass.
vllm() { case "$1 $2" in *--help*) echo "--all-flags-supported"; return 0;; esac
         printf '%s\\n' "$@"; }
export -f vllm 2>/dev/null || true
"""

# A `$VAR` or `${VAR}` reference in a recipe script.
SHELL_VAR = re.compile(r"\$\{?([A-Z_][A-Z0-9_]*)")
# An assignment the script makes itself, with or without `export`.
SHELL_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=", re.M)

MANIFEST_GLOB = config.MANIFEST_GLOB


# ---------------------------------------------------------------------------- #
#                            MANIFEST EXTRACTION                               #
# ---------------------------------------------------------------------------- #

def load_yaml():
    """Import PyYAML, explaining the dependency if it is missing."""
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML not installed: pip install pyyaml")
    return yaml


def serving_container(yaml, path: pathlib.Path) -> dict:
    """Return the container that runs vLLM, across the three workload kinds the recipes use.

    A LeaderWorkerSet carries two templates; the leader is the one that runs `vllm serve`.
    """
    for doc in (d for d in yaml.safe_load_all(path.read_text()) if d):
        kind = doc.get("kind")
        if kind in ("Deployment", "StatefulSet"):
            pod = doc["spec"]["template"]["spec"]
        elif kind == "LeaderWorkerSet":
            pod = doc["spec"]["leaderWorkerTemplate"]["leaderTemplate"]["spec"]
        else:
            continue
        return pod["containers"][0]
    raise ValueError(f"{path.name}: no Deployment, StatefulSet or LeaderWorkerSet found")


def invocation(container: dict) -> tuple[str, str]:
    """Classify how a container builds its vLLM argv, and return that argv's source.

    Recipes come in two shapes. Most pass flags as a YAML `args:` list, where the argv is
    already literal and a stray comment cannot remove an entry. The rest build the command in an
    inline shell script, which is the shape that can silently lose flags and so needs executing
    to find out what actually arrives.
    """
    parts = (container.get("command") or []) + (container.get("args") or [])
    joined = "\n".join(str(part) for part in parts)
    if "vllm serve" in joined or "\n" in str(parts[-1] if parts else ""):
        return "script", str(parts[-1])
    return "args", joined


def serve_block(script: str) -> str:
    """Return the `vllm serve` invocation including its backslash-continuation lines.

    The flags a recipe intends to pass live in this block. Reading them from here rather than
    from the whole script avoids picking up flags that appear in comments or in the manifests'
    own pre-flight checking loops.
    """
    lines = script.splitlines()
    # Two things mention vllm serve without being the invocation: prose and a --help=all probe.
    start = next((i for i, line in enumerate(lines)
                  if "vllm serve" in line
                  and not line.lstrip().startswith("#")
                  and "--help" not in line), None)
    if start is None:
        return ""
    block = [lines[start]]
    while block[-1].rstrip().endswith("\\") and start + len(block) < len(lines):
        block.append(lines[start + len(block)])
    return "\n".join(block)


def served_port(argv: str) -> str | None:
    """Extract the port vLLM will bind, from either `--port=8000` or `--port 8000`.

    Scans from the end: a script may echo a diagnostic line that also mentions --port, and the
    binding that counts is the one in the invocation itself.
    """
    tokens = argv.replace("=", "\n").split()
    for index in range(len(tokens) - 2, -1, -1):
        if tokens[index] == "--port":
            return tokens[index + 1]
    return None


def continuation_comments(script: str) -> list[int]:
    """Line numbers where a comment sits inside a backslash-continued command.

    This is the static signature of the incident this tool was built for. When a line ends with a
    backslash, the next line joins the same command; a `#` there opens a comment that runs to the
    end of that joined line, taking its trailing backslash with it. The command therefore ends at
    that point and every flag below is silently discarded, while the server starts and looks
    healthy. Detecting it needs no execution, so it works for the multinode recipes whose scripts
    deliberately refuse to run without EFA or a formed Ray cluster.
    """
    offenders = []
    continued = False
    for number, line in enumerate(script.splitlines(), start=1):
        if continued and line.lstrip().startswith("#"):
            offenders.append(number)
        continued = line.rstrip().endswith("\\")
    return offenders


def service_target_ports(yaml, path: pathlib.Path) -> list[str]:
    """Ports the manifest's own Services forward to, for cross-checking against the argv.

    A Service aimed at a port vLLM is not listening on yields a deployment that looks healthy
    and refuses every connection, which no flag assertion would catch.
    """
    ports = []
    for doc in (d for d in yaml.safe_load_all(path.read_text()) if d):
        if doc.get("kind") != "Service":
            continue
        for port in doc["spec"].get("ports", []):
            ports.append(str(port.get("targetPort", port.get("port"))))
    return ports


def intended_flags(script: str) -> list[str]:
    """Flags the serve invocation is written to pass, with comments stripped.

    Comparing these against the argv that actually reaches vllm is what catches truncation:
    the original incident was a comment inside the continuation block swallowing the lines
    after it, so flags present in the source never reached the server.
    """
    flags = []
    for line in serve_block(script).splitlines():
        code = line.split("#", 1)[0]                      # a trailing comment is not an argument
        for token in code.split():
            if token.startswith("--") and token.strip("\\") not in flags:
                flags.append(token.strip("\\").split("=")[0])
    return flags


def declared_env(container: dict) -> dict[str, str]:
    """Literal env values the manifest sets on the serving container.

    Entries sourced from a fieldRef or a secretKeyRef have no literal to read, so they are left
    out and fall through to INJECTED_STUBS.
    """
    return {entry["name"]: str(entry["value"])
            for entry in (container.get("env") or []) if "value" in entry}


def stub_prelude(container: dict) -> str:
    """Build the shell prelude for one manifest, preferring its own declared env values."""
    values = {**INJECTED_STUBS, **declared_env(container)}
    assignments = "\n".join(f"{name}={shlex.quote(value)}"
                            for name, value in sorted(values.items()))
    return assignments + SHELL_STUBS


def undefined_vars(container: dict, script: str) -> list[str]:
    """Variables the script reads that nothing defines, so they would expand to empty.

    An empty expansion is worth failing on rather than tolerating: it removes a flag's value while
    leaving the flag itself in argv, which is invisible to the flag comparison and would let a
    recipe pass while carrying `--dist-init-addr :29500` or `--tensor-parallel-size` with nothing
    after it.
    """
    known = set(INJECTED_STUBS) | set(declared_env(container)) \
        | set(SHELL_ASSIGN.findall(script))
    return sorted(set(SHELL_VAR.findall(script)) - known)


def write_temp_script(body: str) -> str:
    """Write a script body to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(body)
        return handle.name


# ---------------------------------------------------------------------------- #
#                               VALIDATION                                     #
# ---------------------------------------------------------------------------- #

def check_syntax(script: str) -> tuple[bool, str]:
    """Syntax-check the script with `bash -n`, returning (ok, first line of any error)."""
    done = subprocess.run(["bash", "-n", write_temp_script(script)],
                          capture_output=True, text=True)
    return done.returncode == 0, done.stderr.strip()[:120]


def resolved_argv(container: dict, script: str) -> str:
    """Run the script with vLLM stubbed out and return the argv it would have passed.

    `exec` is neutralized and the weight-download loop stripped, so the script reaches its
    serve invocation without side effects.
    """
    neutralized = script.replace("exec vllm serve", "vllm serve")
    neutralized = re.sub(r"^\s*for f in .*?done$", "", neutralized, flags=re.S | re.M)
    body = stub_prelude(container) + "\n" + neutralized
    done = subprocess.run(["bash", write_temp_script(body)],
                          capture_output=True, text=True, timeout=60)
    return done.stdout


def validate(yaml, path: pathlib.Path) -> bool:
    """Validate one manifest and print its verdict, returning True if it passed."""
    container = serving_container(yaml, path)
    shape, source = invocation(container)

    if shape == "script":
        syntax_ok, syntax_error = check_syntax(source)
        truncations = continuation_comments(source)
        undefined = undefined_vars(container, source)
        executed = resolved_argv(container, source)
        # Reaching vLLM means flags arrived, since multinode recipes print diagnostics before proceeding.
        reached = any(line.startswith("--") for line in executed.splitlines())
        # Prefer what the stubbed run produced, falling back to the written invocation behind a guard.
        argv = executed if reached else serve_block(source)
        dropped = ([flag for flag in intended_flags(source) if flag not in executed]
                   if reached else [])
    else:
        # A YAML list is already the literal argv: no shell, so nothing to truncate.
        syntax_ok, syntax_error, truncations, dropped, reached = True, "", [], [], True
        undefined = []
        argv = executed = source

    expected = REQUIRED_FLAGS.get(recipe_id(path), [])
    missing = [flag for flag in expected if flag not in argv]

    # The port is not asserted to be 8000, only that one is set and the Service forwards to it.
    port = served_port(argv)
    targets = service_target_ports(yaml, path)
    port_mismatch = bool(port and targets and port not in targets)

    passed = (syntax_ok and not truncations and not dropped and not missing
              and port is not None and not port_mismatch and not undefined)
    print(f"[{'PASS' if passed else 'FAIL'}] {recipe_id(path)}  ({shape})")
    if shape == "script":
        print(f"        bash -n: {'ok' if syntax_ok else syntax_error}"
              f"{'' if reached else ', halted at a precondition: checked as written, not executed'}")
    flags = [token for token in argv.split() if token.startswith("--")]
    print(f"        flags in invocation: {len(flags)}, serving port {port or 'UNSET'}"
          f"{'' if expected else '  (no pinned flag list)'}")
    if truncations:
        print(f"        COMMENT INSIDE LINE CONTINUATION at line(s) {truncations}: every flag "
              f"below it is silently dropped")
    if undefined:
        print(f"        UNDEFINED VARIABLES {undefined}: nothing declares these, so they expand "
              f"to empty and the flags they fill lose their values. Declare them in the "
              f"manifest's env block, or add them to INJECTED_STUBS if the platform supplies "
              f"them at run time.")
    if dropped:
        print(f"        WRITTEN BUT NOT REACHING VLLM: {dropped}")
    if missing:
        print(f"        MISSING FROM ARGV: {missing}")
    if port is None:
        print("        NO --port IN ARGV")
    if port_mismatch:
        print(f"        PORT MISMATCH: vllm binds {port}, Service forwards to {targets}")
    return passed


# ---------------------------------------------------------------------------- #
#                               ENTRY POINT                                    #
# ---------------------------------------------------------------------------- #

def recipe_id(path: pathlib.Path) -> str:
    """Identity of a recipe: its path relative to vllm-recipes/, model directory included.

    Two models share a basename after the model prefix moved into the directory
    (fp8-g7e.4xlarge.yaml, fp8-p5en.48xlarge.yaml), so the basename alone is ambiguous.
    """
    return f"{path.parent.name}/{path.name}"


def select(requested: list[str], available: list[pathlib.Path]) -> list[pathlib.Path]:
    """Narrow the recipe set to the ones named.

    Accepts `<model>/<file>.yaml`, any longer path ending in that, or a bare filename when only
    one model has it. A bare filename that two models share is rejected rather than guessed.
    """
    by_id = {recipe_id(path): path for path in available}
    chosen, unknown, ambiguous = [], [], []
    for item in requested:
        parts = pathlib.Path(item).parts
        key = "/".join(parts[-2:]) if len(parts) >= 2 else None
        if key in by_id:
            chosen.append(by_id[key])
            continue
        name = pathlib.Path(item).name
        matches = [rid for rid in by_id if rid.endswith("/" + name)]
        if len(matches) == 1:
            chosen.append(by_id[matches[0]])
        elif len(matches) > 1:
            ambiguous.append((item, sorted(matches)))
        else:
            unknown.append(item)
    if ambiguous:
        lines = "\n".join(f"  {item}: {opts}" for item, opts in ambiguous)
        sys.exit(f"ambiguous recipe name, name the model directory too:\n{lines}")
    if unknown:
        sys.exit(f"not a recipe in {MANIFEST_GLOB}: {unknown}\n"
                 f"available: {sorted(by_id)}")
    return chosen


def parse_args() -> argparse.Namespace:
    """Parse the optional recipe filter."""
    parser = argparse.ArgumentParser(
        prog="main.py validate-manifests",
        description=f"Validate the vLLM serve recipes in {MANIFEST_GLOB}.")
    parser.add_argument("recipe", nargs="*",
                        help="recipe filename or path to check. Repeatable. Checks every recipe "
                             "in vllm-recipes/ when omitted, which is what a pre-deploy gate "
                             "wants; name one to check just the recipe you are about to apply.")
    return parser.parse_args()


def main() -> int:
    """Validate the selected recipes, returning non-zero if any failed."""
    args = parse_args()
    yaml = load_yaml()
    available = sorted(config.PROJECT_ROOT.glob(MANIFEST_GLOB))
    if not available:
        sys.exit(f"no recipes matched {MANIFEST_GLOB}")

    manifests = select(args.recipe, available) if args.recipe else available
    scope = (f"{len(manifests)} named recipe(s)" if args.recipe
             else f"all {len(manifests)} recipes in {MANIFEST_GLOB}")
    print(f"validating {scope}\n")

    # A key matching no file asserts nothing, and only the full set can prove it stale.
    orphans = ([] if args.recipe
               else sorted(set(REQUIRED_FLAGS) - {recipe_id(path) for path in available}))

    results = [validate(yaml, path) for path in manifests]
    if orphans:
        print(f"\nSTALE config.REQUIRED_FLAGS KEYS — no recipe matches these, so their "
              f"assertions never run:")
        for name in orphans:
            print(f"  {name}")
    print(f"\n{sum(results)}/{len(results)} recipes passed"
          f"{f', {len(orphans)} stale key(s)' if orphans else ''}")
    return 0 if all(results) and not orphans else 1


if __name__ == "__main__":
    sys.exit(main())
