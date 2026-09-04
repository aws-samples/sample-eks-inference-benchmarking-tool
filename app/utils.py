"""Bootstrap and shared constants for the serving-sweep CLI."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# This file lives at app/utils.py, so the repository root is two levels up. Anchoring to
# parents[1] rather than parent is load-bearing: every path below derives from it, and a wrong
# root fails at import with a missing-installer error rather than anything informative.
REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
INSTALL_BENCHMARK_CLIENT = REPO_ROOT / "scripts" / "install-benchmark-client.sh"
CLI_SCRIPT = REPO_ROOT / "app" / "cli.py"

if Path(sys.prefix).resolve() != VENV_DIR.resolve():
    if not VENV_PYTHON.exists():
        subprocess.run([str(INSTALL_BENCHMARK_CLIENT)], cwd=REPO_ROOT, check=True)
    os.execv(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(CLI_SCRIPT), *sys.argv[1:]],
    )

BENCH_SRC = REPO_ROOT / "app" / "bench" / "src"
MAIN_SCRIPT = BENCH_SRC / "main.py"

if str(BENCH_SRC) not in sys.path:
    sys.path.insert(0, str(BENCH_SRC))

HELP_FLAGS = ("-h", "--help")
PARSER_COMMANDS = {
    "accuracy",
    "build-profiles",
    "fetch-eval-data",
    "launch",
    "suite",
    "validate-manifests",
    "verify-profiles",
}
HARNESS_COMMANDS = {"throughput", "latency", "arrival_rate"}
HARNESS_FLAGS = (
    ("--profile NAME", "Serve a pinned prompt profile and use its output-token cap"),
    ("--list-profiles", "List the available pinned prompt profiles and exit"),
)
CLI_USAGE = "python3 app/cli.py <command> [flags]"

import main as benchmark_main
