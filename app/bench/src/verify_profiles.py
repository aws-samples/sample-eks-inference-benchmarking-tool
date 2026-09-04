#!/usr/bin/env python3
"""Measure the pinned prompt profiles against one model's tokenizer and record the counts.

The profiles carry literal prompt text, so there is nothing to calibrate: every model gets
byte-identical input. This records what a given tokenizer makes of that text, which differs by
a few percent between models and is worth knowing when reading two models' results side by
side. It also enforces each profile's claim — if the measured median drifts from
target_input_tokens beyond the tolerance, the profile is misdescribing what it serves and a run
should not be published under that label. Counts are written per model under
docs/prompt_tokens/, deliberately outside data/, because the profiles must stay usable for any
model and nothing in them may key on a model name. Needs transformers and jinja2; downloads
tokenizer files only, never weights.

    python3 app/bench/src/verify_profiles.py --model nvidia/GLM-5.2-NVFP4
    python3 app/bench/src/verify_profiles.py --model /models/kimi-k3 --profile coding_assistant
    python3 app/bench/src/verify_profiles.py --list
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import config
from stats import utc_stamp

TOLERANCE = config.PROFILE_TOLERANCE


# ---------------------------------------------------------------------------- #
#                            PROFILE SELECTION                                 #
# ---------------------------------------------------------------------------- #

def safe_filename(model: str) -> str:
    """Turn a model id into a filename, since repo ids contain path separators."""
    return model.replace("/", "__").replace(":", "_")


def load_profiles(only: list[str] | None) -> list[pathlib.Path]:
    """Return the profile files to measure, exiting if a requested name does not exist."""
    files = sorted(config.PROMPT_PROFILE_DIR.glob("*.json"))
    if not only:
        return files
    wanted = set(only)
    selected = [f for f in files if f.stem in wanted]
    missing = wanted - {f.stem for f in selected}
    if missing:
        sys.exit(f"no such profile(s): {sorted(missing)}")
    return selected


def models_measured_for(profile_name: str) -> list[str]:
    """List the models that already have recorded counts for one profile."""
    found = []
    for path in sorted(config.TOKEN_COUNT_DIR.glob("*.json")):
        if profile_name in json.loads(path.read_text()).get("profiles", {}):
            found.append(path.stem.replace("__", "/"))
    return found


def list_profiles(only: list[str] | None) -> int:
    """Print each profile's declared shape and which models it has been measured for."""
    print(f"{'profile':<24} {'prompts':>8} {'target in':>10} {'out':>6}  measured for")
    print("-" * 78)
    for path in load_profiles(only):
        doc = json.loads(path.read_text())
        measured = ", ".join(models_measured_for(doc["name"])) or "-"
        print(f"{doc['name']:<24} {doc['prompt_count']:>8} {doc['target_input_tokens']:>10} "
              f"{doc['output_tokens']:>6}  {measured}")
    return 0


# ---------------------------------------------------------------------------- #
#                              MEASUREMENT                                     #
# ---------------------------------------------------------------------------- #

def load_tokenizer(model: str):
    """Load the tokenizer for a model, explaining the dependency if it is unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("this tool needs a tokenizer, which the benchmark client does not:\n"
                 "  pip install -r app/bench/requirements-tokenizer.txt\n"
                 "Tokenizer only — it does not pull torch.")
    print(f"loading tokenizer: {model}")
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def wire_counts(tokenizer, prompts: list[str]) -> list[int] | None:
    """Token counts as the SERVER will see them, with the chat template applied.

    Every protocol drives BENCH_ENDPOINT=chat, so the model never receives the raw prompt: it
    receives the prompt wrapped in role markers. That wrapper is what the server bills in
    usage.prompt_tokens, what fills KV cache, and what the harness prints as TOKEN_SHAPE. On a
    short prompt it dominates — nine raw tokens reach GLM-5.2 as twenty-one — so measuring
    only the raw text produces a figure that never appears anywhere else in a run.

    Returns None for a tokenizer with no chat template, where raw text IS what is sent.
    """
    if not getattr(tokenizer, "chat_template", None):
        return None
    try:
        return sorted(len(_template_ids(tokenizer, prompt)) for prompt in prompts)
    except Exception as exc:
        print(f"  warning: chat template failed, wire counts unavailable: {exc}")
        return None


def _template_ids(tokenizer, prompt: str) -> list[int]:
    """Return the token ids for one templated prompt, across transformers return shapes.

    transformers 5.x returns a BatchEncoding from apply_chat_template(tokenize=True), so
    len() on the result counts its KEYS — two — rather than tokens. That silently reported
    every profile as a 2-token wire size. Unwrapping input_ids, and flattening the batch
    dimension when present, is what makes the count a token count.
    """
    out = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=True, add_generation_prompt=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    # A batched call returns [[...]]; a single call returns [...].
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


def measure_profile(tokenizer, doc: dict) -> dict:
    """Measure one profile both as raw text and as it will reach the server.

    Two numbers because a run legitimately involves two: the profile pins raw text and is
    model-agnostic, while the server counts that text plus its own chat template. Recording
    both, with the overhead between them stated, is what stops a reader having to reconcile a
    9 against a 21 and guess which one is wrong.
    """
    raw = sorted(len(tokenizer.encode(prompt)) for prompt in doc["prompts"])
    raw_median = int(statistics.median(raw))
    target = doc["target_input_tokens"]
    measured = {
        "raw_median": raw_median,
        "raw_min": raw[0],
        "raw_max": raw[-1],
        "raw_mean": round(sum(raw) / len(raw), 1),
        "prompts_measured": len(raw),
        "nominal_target": target,
        "error_fraction": round((raw_median - target) / target, 4),
    }
    wire = wire_counts(tokenizer, doc["prompts"])
    if wire is None:
        measured["wire_median"] = None
        measured["chat_template_overhead"] = None
        return measured
    measured["wire_median"] = int(statistics.median(wire))
    measured["wire_min"] = wire[0]
    measured["wire_max"] = wire[-1]
    # What the harness will report as input_median, and the gap that explains it.
    measured["chat_template_overhead"] = measured["wire_median"] - raw_median
    return measured


def tokenizer_stack() -> dict[str, str]:
    """Record the library versions that produced these counts.

    Token counts are a property of the tokenizer implementation, not just the model: a
    different transformers or tokenizers release can split the same text differently or render
    a different chat template. Without the versions, a later mismatch is unattributable.
    """
    import importlib.metadata as metadata

    versions = {}
    for package in ("transformers", "tokenizers", "jinja2"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def counts_path(model: str) -> pathlib.Path:
    """Where this model's recorded counts live."""
    return config.TOKEN_COUNT_DIR / f"{safe_filename(model)}.json"


def ensure_counts(model: str) -> str:
    """Measure and record this model's token counts if they are not already on disk.

    Safe to do automatically, unlike an accuracy reference: a count is a pure function of the
    pinned prompt text and the tokenizer, with no server involved, so generating it cannot
    manufacture agreement with anything. It needs the tokenizer extras and network access to fetch
    the tokenizer, so a failure here degrades to the run reporting a nominal input size rather
    than aborting the benchmark.

    Returns a one-line status for the caller to print.
    """
    path = counts_path(model)
    if path.exists():
        return f"input sizes: already recorded for {model}"
    try:
        from transformers import AutoTokenizer            # noqa: F401
    except ImportError:
        return (f"input sizes: NOT recorded for {model} and cannot be measured here — "
                f"pip install -r app/bench/requirements-tokenizer.txt to enable. The run "
                f"continues and reports a nominal input size.")
    try:
        tokenizer = load_tokenizer(model)
        measured = {}
        for profile_path in load_profiles(None):
            doc = json.loads(profile_path.read_text())
            measured[doc["name"]] = measure_profile(tokenizer, doc)
        written = write_counts(model, measured)
        return f"input sizes: measured {model} and wrote {written.name}"
    except SystemExit:
        raise
    except Exception as exc:                              # noqa: BLE001
        return (f"input sizes: could not measure {model} ({type(exc).__name__}: "
                f"{str(exc)[:80]}). The run continues and reports a nominal input size.")


def write_counts(model: str, measured: dict[str, dict]) -> pathlib.Path:
    """Record this model's measured counts as the authoritative input sizes for its runs."""
    config.TOKEN_COUNT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TOKEN_COUNT_DIR / f"{safe_filename(model)}.json"
    path.write_text(json.dumps({
        "schema": "prompt-profile-tokens/v2",
        "model": model,
        "measured_utc": utc_stamp(),
        "measured_with": tokenizer_stack(),
        "note": "Token counts for the pinned prompts in app/bench/data/prompt_profiles/, measured with "
                "this model's tokenizer. raw_* is the prompt text alone, which is what the "
                "profile pins and what target_input_tokens labels. wire_* is that text with "
                "this model's chat template applied, which is what the server actually "
                "receives and reports as usage.prompt_tokens — it is the number the harness "
                "prints as TOKEN_SHAPE input_median. chat_template_overhead is the "
                "difference. Quote wire_median as the input size of a run; the profile's "
                "target_input_tokens is a raw-text size-class label, not a wire measurement.",
        "tolerance": TOLERANCE,
        "profiles": measured,
    }, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------- #
#                               ENTRY POINT                                    #
# ---------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",
                        help="HF repo id or local path whose tokenizer to measure with")
    parser.add_argument("--profile", action="append",
                        help="limit to these profiles (repeatable)")
    parser.add_argument("--list", action="store_true", help="show profiles and exit")
    return parser.parse_args()


def main() -> int:
    """Measure every selected profile and fail if any misdescribes its input size."""
    args = parse_args()
    if args.list:
        return list_profiles(args.profile)
    if not args.model:
        sys.exit("--model is required unless --list is given")

    tokenizer = load_tokenizer(args.model)
    measured: dict[str, dict] = {}
    failures: list[str] = []
    deviations: list[str] = []
    is_reference = args.model == config.PROFILE_REFERENCE_TOKENIZER
    print()
    print("raw  = prompt text alone; this is what target_input_tokens labels.")
    print("wire = raw + this model's chat template; this is what the server receives and")
    print("       what a run reports as TOKEN_SHAPE input_median. Quote wire as the input size.")
    print()
    if is_reference:
        print(f"{args.model} is the reference tokenizer, so a deviation beyond "
              f"{TOLERANCE:.0%} means the")
        print("generator's chars-per-token ratios are wrong and this run fails.")
    else:
        print(f"Reference tokenizer is {config.PROFILE_REFERENCE_TOKENIZER}. For any other model "
              f"the nominal")
        print("target is only a label: identical text tokenizes differently, so deviation here is "
              "recorded,")
        print("not an error. Each run's protocol lock reports the measured value for its own "
              "tokenizer.")
    print(f"\n{'profile':<24} {'target':>8} {'raw':>7} {'err':>7} {'wire':>7} {'tmpl':>6}")
    print("-" * 68)

    for path in load_profiles(args.profile):
        doc = json.loads(path.read_text())
        stats = measure_profile(tokenizer, doc)
        measured[doc["name"]] = stats
        error = stats["error_fraction"]
        flag = ""
        if abs(error) > TOLERANCE:
            if is_reference:
                flag = "   <-- MISDESCRIBED"
                failures.append(f"  {doc['name']}: claims {stats['nominal_target']} raw input "
                                f"tokens, measures {stats['raw_median']} ({error:+.1%}) with the "
                                f"reference tokenizer. Either fix target_input_tokens or "
                                f"regenerate the prompts.")
            else:
                flag = "   <-- differs from nominal"
                deviations.append(f"  {doc['name']}: nominal {stats['nominal_target']}, "
                                  f"measures {stats['raw_median']} ({error:+.1%})")
        wire = stats["wire_median"]
        overhead = stats["chat_template_overhead"]
        wire_text = str(wire) if wire is not None else "-"
        tmpl_text = f"+{overhead}" if overhead is not None else "-"
        print(f"{doc['name']:<24} {stats['nominal_target']:>8} {stats['raw_median']:>7} "
              f"{error:>+7.1%} {wire_text:>7} {tmpl_text:>6}{flag}")

    written = write_counts(args.model, measured)
    print(f"\nwrote {written.relative_to(config.PROJECT_ROOT)}")

    if failures:
        print("\nPROFILES MISDESCRIBE THEIR INPUT SIZE FOR THE REFERENCE TOKENIZER:")
        print("\n".join(failures))
        print("\nA run published under a misdescribed profile repeats exactly the confusion")
        print("these files exist to prevent.")
        return 1

    if deviations:
        print(f"\nRECORDED, NOT AN ERROR — {args.model} reads these profiles differently from "
              f"their nominal label:")
        print("\n".join(deviations))
        print("\nRuns against this model will report the measured value, so nothing published "
              "quotes the")
        print("nominal number. Pass --tokenizer to suite or launch so the lock can resolve it.")
        return 0

    print(f"\nall profiles within {TOLERANCE:.0%} of their stated input size for {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
