#!/usr/bin/env python3
"""Benchmark client entry point that runs one of the load-generating harnesses against a vLLM
endpoint, invokes the protocol-locked suite runner for repeatable multi-rep measurements, or
dispatches the one-off tools. It depends only on the standard library and aiohttp so it can run in
a stock Python image in-cluster, and each command's flags are owned by its own argparse parser. The
full environment variable contract lives in config.py."""
import argparse
import json
import os
import sys

# Commands running a harness in-process, excluding the suite, which owns its own --profile.
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
    # A profile supersedes the generated-prompt knobs, so leaving them set would imply they applied.
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
