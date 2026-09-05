"""
CLI utilities for the EKS Inference Benchmarking Tool entry point.

ORGANIZATION:
    1. Repository paths and legacy-command metadata - layout and argparse-layer facts
    2. Command tree - Leaf and the flat COMMAND_TREE registry
    3. Tree traversal - enumerate and validate leaves
    4. Console rendering - rich console, tables, and the delegated help surfaces
    5. Command execution - exec the unchanged benchmark entry point

ARCHITECTURE:
    cli.py holds only Click framework wiring; everything reusable lives here, so the entry
    point stays focused on its framework and the command vocabulary stays in one table.

    Nothing here changes behavior. Every leaf resolves to a legacy `main.py` command, so the
    argparse layer, the harness modules and the in-cluster invocation are untouched.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

# =============================================================================
# Repository Paths
# =============================================================================

# This file lives at app/utils.py, so the repository root is two levels up. Anchoring to
# parents[1] rather than parent is load-bearing: every path below derives from it.
REPO_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# Legacy-Command Metadata
# =============================================================================

BENCH_SRC = REPO_ROOT / "app" / "bench" / "src"
MAIN_SCRIPT = BENCH_SRC / "main.py"

if str(BENCH_SRC) not in sys.path:
    sys.path.insert(0, str(BENCH_SRC))

HELP_FLAGS = ("-h", "--help")

# Legacy commands that own an argparse parser, so their help can be captured and restyled
# rather than re-declared here.
PARSER_COMMANDS = {
    "accuracy",
    "build-profiles",
    "fetch-eval-data",
    "launch",
    "suite",
    "validate-manifests",
    "verify-profiles",
}

# Legacy commands that run a load generator in-process and therefore accept the profile flags.
HARNESS_COMMANDS = {"throughput", "latency", "arrival_rate"}
HARNESS_FLAGS = (
    ("--profile NAME", "Serve a pinned prompt profile and use its output-token cap"),
    ("--list-profiles", "List the available pinned prompt profiles and exit"),
)

# The pip distribution name, used to resolve the installed version.
DISTRIBUTION_NAME = "eks-inference-benchmarking-tool"

# The short console script alias, used in documentation and in the usage lines this module
# still renders. Both names install and behave identically; this is the one docs quote.
PROG_NAME = "eks-bench"

import main as benchmark_main

# =============================================================================
# Command Tree
# =============================================================================


class Leaf:
    """One runnable command: its name in the tree, and what it execs."""

    def __init__(self, name: str, legacy: str, summary: str, inject: tuple[str, ...] = ()) -> None:
        """
        Define a runnable command.

        Args:
            name: Name in the tree, e.g. "arrival-rate"
            legacy: main.py command this execs, e.g. "arrival_rate"
            summary: One-line description shown in help
            inject: Flags always passed to the legacy command, e.g. ("--write-reference",)
        """
        self.name = name
        self.legacy = legacy
        self.summary = summary
        self.inject = inject


COMMAND_TREE: dict[str, Leaf] = {
    "measure": Leaf(
        "measure", "launch",
        "Benchmark a server in-cluster: runs the client as a Job beside it, over cluster DNS. "
        "Required for any TTFT figure, and the path every runbook uses",
    ),
    "measure-local": Leaf(
        "measure-local", "suite",
        "Benchmark a server you can already reach, with the client running here. For vLLM on "
        "bare EC2, or from inside the cluster. Refuses to report TTFT over a loopback tunnel",
    ),
    "validate-manifests": Leaf(
        "validate-manifests", "validate-manifests",
        "Prove a serving recipe passes the flags it is written to pass",
    ),
    # The accuracy gate's three actions sit at the top level rather than under an `accuracy`
    # group. A group name has to read as a countable thing an operator has several of, the way
    # `gh pr` and `docker image` do, or the pair reads as two unrelated commands side by side.
    # Accuracy is a quality, not a collection, so these are verb-noun compounds instead,
    # matching validate-manifests and measure-local.
    #
    # capture-reference runs in-cluster, which is the path docs/accuracy/README.md prescribes
    # and every runbook uses. It reaches `launch`, not `accuracy`, so an operator never has to
    # know the in-cluster capture is a flag on the benchmark command.
    "capture-reference": Leaf(
        "capture-reference", "launch",
        "Capture an accuracy reference in-cluster, over cluster DNS. Every benchmark is gated "
        "against one, and this is the path every runbook uses",
        inject=("--capture-accuracy-reference",),
    ),
    "capture-reference-local": Leaf(
        "capture-reference-local", "accuracy",
        "Capture an accuracy reference from a server you can already reach, with the client "
        "running here",
        inject=("--write-reference",),
    ),
    "check-accuracy": Leaf(
        "check-accuracy", "accuracy",
        "Check a live server against its captured reference. Every measure run does this "
        "automatically, so run it by hand to diagnose a gate failure",
    ),
    "fetch-eval-data": Leaf(
        "fetch-eval-data", "fetch-eval-data",
        "Pin the GSM8K subset the accuracy gate scores against",
    ),
    # Prompt profiles pin the input text and output cap a run is measured at. Named
    # "prompts" rather than "profiles" because profiling means Nsight in this repo.
    "build-prompts": Leaf(
        "build-prompts", "build-profiles",
        "Regenerate the pinned prompt profile JSON files, deterministically",
    ),
    "verify-prompts": Leaf(
        "verify-prompts", "verify-profiles",
        "Measure the pinned prompt profiles against a real tokenizer, raw and wire",
    ),
    "list-prompts": Leaf(
        "list-prompts", "verify-profiles",
        "List the available pinned prompt profiles",
        inject=("--list",),
    ),
    # Running one load generator directly produces a single unlocked reading: no protocol lock,
    # accuracy gate, reps, medians or provenance. The harness- prefix keeps that visible at the
    # call site, because the output otherwise resembles a result.
    "harness-throughput": Leaf(
        "harness-throughput", "throughput",
        "One closed-loop throughput reading, non-streaming. Unlocked, so not a result",
    ),
    "harness-latency": Leaf(
        "harness-latency", "latency",
        "One closed-loop latency reading, streaming: TTFT/TPOT/ITL. Unlocked, so not a result",
    ),
    "harness-arrival-rate": Leaf(
        "harness-arrival-rate", "arrival_rate",
        "One open-loop capacity reading at a target arrival rate (MLPerf LoadGen). Unlocked",
    ),
    # Regression checks. Each needs no GPU and no server; the wire checks are skipped with a
    # notice when their optional extra is absent.
    "selftest-aiperf": Leaf(
        "selftest-aiperf", "verify-aiperf",
        "Regression checks for the AIPerf adapter, against a local mock",
    ),
    "selftest-openloop": Leaf(
        "selftest-openloop", "verify-openloop",
        "Regression checks for the LoadGen harness, against a local mock",
    ),
    "selftest-accuracy-gate": Leaf(
        "selftest-accuracy-gate", "verify-accuracy-gate",
        "Regression checks for the correctness gate, fully local",
    ),
}

# =============================================================================
# Tree Traversal
# =============================================================================


def all_leaves() -> list[tuple[str, Leaf]]:
    """
    Enumerate every runnable command in the tree.

    Returns:
        List of (command path, leaf) pairs, where the path is what an operator types
        after the program name, e.g. "measure" or "capture-reference"
    """
    return list(COMMAND_TREE.items())


def unknown_legacy_commands() -> list[str]:
    """
    Find tree entries that reference a command main.py does not dispatch.

    Returns:
        Sorted list of offending legacy command names, empty when the tree is consistent
    """
    referenced = {leaf.legacy for _, leaf in all_leaves()}
    return sorted(referenced - set(benchmark_main.COMMANDS))


# Fail at import if the tree has drifted from main.py, so a rename there surfaces here rather
# than as a confusing runtime error from the exec'd process.
_unknown = unknown_legacy_commands()
if _unknown:
    raise SystemExit(f"cli.py references commands main.py does not define: {_unknown}")

# =============================================================================
# Console Rendering
# =============================================================================


def build_console() -> Console:
    """
    Build a console that respects standard no-color behavior.

    Returns:
        A rich Console honoring the NO_COLOR environment variable
    """
    return Console(no_color=bool(os.environ.get("NO_COLOR")), force_terminal=None)


def summary_table(rows: tuple[tuple[str, str], ...]) -> Table:
    """
    Build the two-column name/description table used by every help surface.

    Args:
        rows: (name, description) pairs, rendered in order

    Returns:
        A borderless rich Table with the name column highlighted
    """
    table = Table(show_header=False, box=None, padding=(0, 2), collapse_padding=True)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    for name, description in rows:
        table.add_row(name, description)
    return table


def print_usage(line: str) -> None:
    """
    Print a usage line.

    Rendered with markup disabled so bracketed tokens such as [flags] survive rather than
    being parsed as rich style tags.

    Args:
        line: The invocation to show, without the "Usage:" label
    """
    console = build_console()
    console.print("[bold]Usage:[/bold]")
    console.print(f"  {line}", markup=False)


def print_simple_help(path: str, leaf: Leaf) -> None:
    """
    Render help for a leaf whose legacy command owns no argparse parser.

    Args:
        path: The command name as typed, e.g. "harness-throughput"
        leaf: The leaf to describe
    """
    console = build_console()
    print_usage(f"{PROG_NAME} {path} [flags]")
    console.print()
    console.print(leaf.summary)
    console.print()
    console.print("[bold]Flags:[/bold]")
    rows = list(HARNESS_FLAGS) if leaf.legacy in HARNESS_COMMANDS else []
    rows.append(("-h, --help", "Show this message and exit"))
    console.print(summary_table(tuple(rows)))


def print_styled_parser_output(path: str, leaf: Leaf, output: str) -> None:
    """
    Style captured argparse help without reconstructing its option definitions.

    Args:
        path: The command path as typed, substituted into the usage line
        leaf: The leaf whose legacy command produced the output
        output: Raw stdout from `main.py <legacy> --help`
    """
    console = build_console()
    replacement = f"usage: {PROG_NAME} {path}"
    explicit = f"usage: main.py {leaf.legacy}"
    output = output.replace(explicit, replacement, 1) if explicit in output else output.replace(
        "usage: main.py", replacement, 1
    )
    for line in output.rstrip().splitlines():
        stripped = line.strip().lower()
        if stripped in ("options:", "optional arguments:"):
            console.print("Flags:", style="bold")
            continue
        if stripped == "positional arguments:":
            console.print("Positional Arguments:", style="bold")
            continue
        if stripped.startswith("usage:"):
            console.print(line, style="bold")
            continue
        text = Text(line)
        text.highlight_regex(r"(?<!\w)--?[a-zA-Z][a-zA-Z0-9-]*", style="cyan")
        console.print(text, soft_wrap=True)


def print_parser_help(path: str, leaf: Leaf) -> None:
    """
    Capture an existing parser's help and render it without running the command.

    Args:
        path: The command path as typed, e.g. "measure"
        leaf: The leaf whose legacy command owns the parser

    Raises:
        click.ClickException: If the legacy command could not produce help output
    """
    completed = subprocess.run(
        [sys.executable, str(MAIN_SCRIPT), leaf.legacy, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise click.ClickException(f"Unable to render help for {path!r}: {detail}")
    print_styled_parser_output(path, leaf, completed.stdout)


def print_leaf_help(path: str, leaf: Leaf) -> None:
    """
    Render a leaf's help, from its argparse parser when it has one.

    Args:
        path: The command name as typed, e.g. "capture-reference"
        leaf: The leaf to describe
    """
    if leaf.legacy in PARSER_COMMANDS:
        print_parser_help(path, leaf)
    else:
        print_simple_help(path, leaf)


# =============================================================================
# Command Execution
# =============================================================================


def execute_leaf(leaf: Leaf, arguments: tuple[str, ...]) -> None:
    """
    Replace this process with the unchanged benchmark entry point.

    Uses execv rather than a subprocess so the harness owns the terminal directly, keeping
    streamed output and the final exit code unmediated.

    Args:
        leaf: The leaf whose legacy command to run
        arguments: Operator-supplied arguments, forwarded verbatim after any injected flags
    """
    argv = [sys.executable, str(MAIN_SCRIPT), leaf.legacy, *leaf.inject, *arguments]
    os.execv(sys.executable, argv)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Paths
    "REPO_ROOT",
    "BENCH_SRC",
    "MAIN_SCRIPT",

    # Legacy-command metadata
    "HELP_FLAGS",
    "PARSER_COMMANDS",
    "HARNESS_COMMANDS",
    "HARNESS_FLAGS",
    "DISTRIBUTION_NAME",
    "PROG_NAME",
    "benchmark_main",

    # Command tree
    "Leaf",
    "COMMAND_TREE",

    # Tree traversal
    "all_leaves",
    "unknown_legacy_commands",

    # Console rendering
    "build_console",
    "summary_table",
    "print_usage",
    "print_simple_help",
    "print_styled_parser_output",
    "print_parser_help",
    "print_leaf_help",

    # Command execution
    "execute_leaf",
]
