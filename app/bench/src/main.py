#!/usr/bin/env python3
"""Benchmark client entry point.

Run one of the three harnesses against a vLLM endpoint, or invoke the protocol-locked
suite runner for repeatable, multi-rep measurements.

Usage:
    # Single harness, optionally on a pinned prompt profile:
    python main.py throughput --profile chat_typical
    python main.py latency --profile coding_assistant
    python main.py arrival_rate

    # Protocol-locked suite (reps, medians, accuracy gate):
    python main.py suite --protocol corpus --profile chat_typical \
        --instance g7e.12xlarge --config fp8 --reps 3

    # One-off tools:
    python main.py fetch-eval-data [--verify]
    python main.py build-profiles [--check]
    python main.py verify-profiles --model <repo-or-path>
    python main.py validate-manifests

Harness flags (throughput / latency / arrival_rate):
    --profile NAME  Serve a pinned profile from app/bench/data/prompt_profiles/. Overrides
                    PROMPT_PROFILE and, like the suite runner, takes the profile's
                    own output_tokens as MAX_TOKENS so both ends of the prompt
                    shape come from one committed artifact.
    --list-profiles Show the available profile names and exit.

Environment:
    VLLM_API_URL    Endpoint to benchmark (default: http://vllm-server:8000/v1)
    MODEL_ID        Served model name as reported by /v1/models
    INSTANCE_TYPE   Hardware tag recorded in results
    QUANTIZATION    Precision tag recorded in results

    Full variable list: config.py
"""
import argparse
import json
import os
import sys

# Commands that run a harness in-process and therefore accept the profile flag. The suite
# runner is excluded: it has its own --profile and spawns harnesses as subprocesses.
HARNESS_COMMANDS = ("throughput", "latency", "arrival_rate")

COMMANDS = {
    "throughput": "Closed-loop throughput (non-streaming)",
    "latency": "Closed-loop latency (streaming: TTFT/TPOT/ITL)",
    "arrival_rate": "Open-loop capacity (MLPerf LoadGen)",
    "suite": "Protocol-locked multi-rep runner",
    "accuracy": "Correctness gate: check a server, or capture its reference",
    "fetch-eval-data": "Pin the GSM8K subset for the accuracy gate",
    "build-profiles": "Generate prompt profile JSON files",
    "verify-profiles": "Measure profiles against a model's tokenizer",
    "validate-manifests": "Prove K8s manifests pass correct vLLM flags",
    "verify-aiperf": "Regression checks for the aiperf adapter (local, no GPU)",
    "verify-openloop": "Regression checks for the LoadGen harness (local, no GPU)",
    "verify-accuracy-gate": "Regression checks for the correctness gate (local, no GPU)",
    "launch": "Run the suite IN-CLUSTER over cluster DNS (required for TTFT)",
}


def available_profiles() -> list[str]:
    """List the pinned profile names on disk."""
    import config
    return sorted(path.stem for path in config.PROMPT_PROFILE_DIR.glob("*.json"))


def apply_profile(name: str) -> None:
    """Export a profile's settings, then reload config so they take effect.

    config.py reads the environment once at import, and locating the profile requires
    importing config, so setting the variables alone is not enough: the values are already
    frozen by then. The reload is what makes the flag actually apply, and it is safe here
    because nothing else has imported config yet — the harness modules are imported after.

    MAX_TOKENS is set alongside PROMPT_PROFILE to mirror the suite runner: a profile pins the
    input AND the output cap, and applying half of it produces a run labelled with a profile
    whose output shape came from somewhere else.
    """
    import importlib

    import config
    path = config.PROMPT_PROFILE_DIR / f"{name}.json"
    if not path.exists():
        sys.exit(f"unknown --profile {name!r}. Available: {available_profiles()}")
    doc = json.loads(path.read_text())

    os.environ["PROMPT_PROFILE"] = name
    os.environ["MAX_TOKENS"] = str(doc["output_tokens"])
    # A profile supersedes the generated-prompt knobs entirely; leaving them set would imply
    # they still influenced the run.
    os.environ.pop("PROMPT_MODE", None)
    os.environ.pop("PROMPT_TOKENS", None)
    importlib.reload(config)

    print(f"profile: {name} ({doc['prompt_count']} pinned prompts, "
          f"nominal input {doc['target_input_tokens']} tok, "
          f"output cap {doc['output_tokens']} tok)")


def parse_harness_flags() -> None:
    """Handle the harness-level flags, leaving the rest of argv untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--list-profiles", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.list_profiles:
        for name in available_profiles():
            print(name)
        raise SystemExit(0)
    if args.profile:
        apply_profile(args.profile)
    sys.argv = [sys.argv[0]] + remaining


def print_usage() -> None:
    """Print the module docstring and the command list."""
    print(__doc__)
    print("Available commands:")
    for name, description in COMMANDS.items():
        print(f"  {name:<18} {description}")


def main() -> int:
    """Dispatch to the requested subcommand."""
    if len(sys.argv) < 2:
        print_usage()
        return 1

    command = sys.argv[1]
    # Remove the subcommand from argv so downstream argparse sees its own args.
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command in HARNESS_COMMANDS:
        parse_harness_flags()

    if command == "throughput":
        import throughput
        throughput.main()
    elif command == "latency":
        import latency
        latency.main()
    elif command == "arrival_rate":
        import arrival_rate
        arrival_rate.main()
    elif command == "suite":
        import suite
        return suite.main()
    elif command == "accuracy":
        import accuracy
        return accuracy.main()
    elif command == "fetch-eval-data":
        import fetch_eval_data
        return fetch_eval_data.main()
    elif command == "build-profiles":
        import build_profiles
        return build_profiles.main()
    elif command == "verify-profiles":
        import verify_profiles
        return verify_profiles.main()
    elif command == "validate-manifests":
        import validate_manifests
        return validate_manifests.main()
    elif command == "verify-aiperf":
        import verify_aiperf
        return verify_aiperf.main()
    elif command == "verify-openloop":
        import verify_openloop
        return verify_openloop.main()
    elif command == "verify-accuracy-gate":
        import verify_accuracy_gate
        return verify_accuracy_gate.main()
    elif command == "launch":
        import launcher
        return launcher.main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run without arguments to see available commands.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
