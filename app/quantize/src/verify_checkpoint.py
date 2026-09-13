#!/usr/bin/env python3
"""Decide whether a quantized checkpoint is what its own config claims to be, and exit non-zero
when it is not.

This exists because the failure it catches is silent. A checkpoint can carry a `quantization_config`
announcing FP8 while its weights are still BF16, and vLLM then reads BF16 bytes as FP8 and serves
fluent-looking nonsense rather than refusing to start. Nothing upstream fails: the quantize Job
exits 0, the file lands on the shared claim, and every recipe pointing at that path inherits it. An
accuracy gate captured against the same bad checkpoint agrees with itself, so benchmark results can
be published from it.

Deliberately stdlib-only. It parses safetensors headers directly rather than importing torch,
modelopt or safetensors, so it can run in a slim image against a checkpoint produced by any
version of quantize.py, including one from an image older than this repository. A verifier that
only ran inside the producing image could not check the case that motivated it.

Each check is its own function returning (problems, notes), and verify() only concatenates them, so
a new check is a new function rather than another branch in a long one."""
import argparse
import fnmatch
import json
import os
import struct
import sys

# safetensors dtype tags that mean the tensor really was cast, per quantization format.
QUANTIZED_DTYPES = {
    "fp8": {"F8_E4M3", "F8_E5M2"},
    "int8": {"I8"},
}

# Formats whose weight dtype cannot be asserted this simply; packed 4-bit lands in an integer tensor.
DTYPE_CHECK_SKIPPED = {"awq", "int4", "nvfp4", "mxfp4"}

HIGH_PRECISION_DTYPES = {"BF16", "F16", "F32", "F64"}

# One check returns problems that block serving, plus notes that explain what was seen either way.
Findings = tuple[list[str], list[str]]


# ---------------------------------------------------------------------------- #
#                             READING THE CHECKPOINT                           #
# ---------------------------------------------------------------------------- #

def read_safetensors_header(path: str) -> dict:
    """Return one safetensors file's header as a dict of tensor name to metadata.

    The format is an 8-byte little-endian header length followed by that many bytes of JSON, so
    the tensor inventory costs a small read rather than loading an 11 GiB file.
    """
    with open(path, "rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) < 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        (length,) = struct.unpack("<Q", raw_length)
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def collect_tensors(checkpoint_dir: str) -> dict:
    """Merge the headers of every safetensors shard in a checkpoint directory."""
    shards = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors"))
    if not shards:
        raise ValueError(f"{checkpoint_dir}: no .safetensors files found")
    tensors = {}
    for shard in shards:
        tensors.update(read_safetensors_header(os.path.join(checkpoint_dir, shard)))
    return tensors


def load_json(checkpoint_dir: str, name: str):
    """Read a JSON file from the checkpoint, returning None when it is absent."""
    path = os.path.join(checkpoint_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def find_key(obj, wanted: str):
    """Search nested dicts for a key, since quant config layouts differ between exporters."""
    if not isinstance(obj, dict):
        return None
    if wanted in obj:
        return obj[wanted]
    for value in obj.values():
        found = find_key(value, wanted)
        if found is not None:
            return found
    return None


def weight_tensors(tensors: dict) -> dict:
    """Return only the parameter tensors, excluding the scale tensors that accompany them."""
    return {name: meta for name, meta in tensors.items()
            if name.endswith(".weight") and "_scale" not in name and "_amax" not in name}


def dtype_counts(weights: dict) -> dict:
    """Count weight tensors by safetensors dtype tag."""
    counts: dict[str, int] = {}
    for meta in weights.values():
        tag = meta.get("dtype", "?")
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def declares_quantization(model_config, quant_config) -> bool:
    """Whether the checkpoint advertises itself as quantized, regardless of its actual weights."""
    if find_key(model_config, "quantization_config") is not None:
        return True
    return bool(find_key(quant_config, "quant_algo"))


def matching_weights(weights: dict, pattern: str) -> list[str]:
    """Weight names matching an exclusion pattern, tolerating patterns written without wildcards."""
    loose = f"*{pattern.strip('*')}*"
    return [name for name in weights
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name, loose)]


# ---------------------------------------------------------------------------- #
#                                   CHECKS                                     #
# ---------------------------------------------------------------------------- #

def check_quant_config(quant_config) -> Findings:
    """The ModelOpt export writes hf_quant_config.json, and vLLM's loader needs it."""
    if quant_config is None:
        return ([f"hf_quant_config.json is missing. A real ModelOpt export writes it, and vLLM's "
                 f"modelopt loader needs it to read the layout."], [])
    return ([], [f"hf_quant_config.json present, "
                 f"quant_algo={find_key(quant_config, 'quant_algo')!r}, "
                 f"kv_cache_quant_algo={find_key(quant_config, 'kv_cache_quant_algo')!r}"])


def check_unsupported_format(fmt: str) -> Findings:
    """A format this script cannot reason about must not pass silently."""
    if fmt in DTYPE_CHECK_SKIPPED:
        return ([], [f"weight dtype not asserted for {fmt}: it packs into an integer tensor"])
    return ([f"unknown quantization format {fmt!r}; cannot verify weight dtype"], [])


def check_weights_never_cast(weights: dict, fmt: str, expected: set, claims: bool) -> Findings:
    """The failure this script exists for: labelled as quantized, weights never cast."""
    detail = (" config.json nevertheless declares a quantization_config, which is the invalid "
              "legacy export: vLLM will read high-precision bytes as "
              f"{fmt} and emit garbage instead of failing." if claims else "")
    return ([f"no weight tensor uses {' or '.join(sorted(expected))}, so the weights were never "
             f"cast to {fmt}.{detail}"], [])


def check_weight_dtypes(weights: dict, fmt: str, claims: bool) -> Findings:
    """Confirm the weights carry the dtype the requested format implies."""
    counts = dtype_counts(weights)
    seen = ["weight dtypes: " + ", ".join(f"{d}={n}" for d, n in sorted(counts.items()))]
    expected = QUANTIZED_DTYPES.get(fmt)
    if expected is None:
        problems, notes = check_unsupported_format(fmt)
        return (problems, seen + notes)
    present = expected & set(counts)
    if not present:
        problems, notes = check_weights_never_cast(weights, fmt, expected, claims)
        return (problems, seen + notes)
    quantized = sum(counts[tag] for tag in present)
    return ([], seen + [f"{quantized} of {len(weights)} weights are genuinely {fmt}"])


def check_calibration(tensors: dict) -> Findings:
    """Static-activation quantization needs the input_scale tensors calibration populates."""
    scales = [name for name in tensors if name.endswith("input_scale")]
    if not scales:
        return (["no input_scale tensors, so static activation scales were never populated. "
                 "Calibration did not run, and static-activation quantization without it is "
                 "invalid."], [])
    return ([], [f"{len(scales)} input_scale tensors present, so calibration ran"])


def check_one_exclusion(weights: dict, pattern: str) -> Findings:
    """An excluded pattern must have stayed high precision."""
    matched = matching_weights(weights, pattern)
    if not matched:
        return ([], [f"exclusion {pattern!r} matched no weight tensor"])
    wrongly = [n for n in matched if weights[n].get("dtype") not in HIGH_PRECISION_DTYPES]
    if wrongly:
        return ([f"exclusion {pattern!r} did not hold: {len(wrongly)} of {len(matched)} matching "
                 f"weights were quantized anyway, e.g. {wrongly[0]}"], [])
    return ([], [f"exclusion {pattern!r} held across {len(matched)} weights"])


def check_exclusions(weights: dict, patterns: list[str]) -> Findings:
    """Run the exclusion check for every requested pattern."""
    problems: list[str] = []
    notes: list[str] = []
    for pattern in patterns:
        found, seen = check_one_exclusion(weights, pattern)
        problems += found
        notes += seen
    return (problems, notes)


# ---------------------------------------------------------------------------- #
#                                 VERIFICATION                                 #
# ---------------------------------------------------------------------------- #

def verify(checkpoint_dir: str, fmt: str, exclude_patterns: list[str],
           expect_calibrated: bool) -> Findings:
    """Return (problems, notes) for a checkpoint. A non-empty problems list means do not serve it."""
    tensors = collect_tensors(checkpoint_dir)
    weights = weight_tensors(tensors)
    model_config = load_json(checkpoint_dir, "config.json")
    quant_config = load_json(checkpoint_dir, "hf_quant_config.json")
    claims = declares_quantization(model_config, quant_config)

    results = [
        ([], [f"{len(tensors)} tensors across the checkpoint, {len(weights)} of them weights"]),
        ([], ["config.json carries a quantization_config block"] if claims else []),
        check_quant_config(quant_config),
        check_weight_dtypes(weights, fmt.lower(), claims),
        check_exclusions(weights, exclude_patterns),
    ]
    if expect_calibrated:
        results.append(check_calibration(tensors))

    problems = [p for found, _ in results for p in found]
    notes = [n for _, seen in results for n in seen]
    return (problems, notes)


def report(checkpoint_dir: str, problems: list[str], notes: list[str]) -> None:
    """Print the verdict, evidence first so a failure is readable without rerunning."""
    print(f"=== checkpoint verification: {checkpoint_dir} ===")
    for note in notes:
        print(f"  {note}")
    if not problems:
        print("\n  OK: the checkpoint matches what its config claims.")
        return
    print(f"\n  {len(problems)} PROBLEM(S). Do not serve this checkpoint:")
    for problem in problems:
        print(f"    - {problem}")


# ---------------------------------------------------------------------------- #
#                                 ENTRY POINT                                  #
# ---------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Read the checkpoint to verify and how it was meant to be quantized."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", nargs="?", default=os.getenv("CHECKPOINT_DIR"),
                        help="checkpoint directory; defaults to $CHECKPOINT_DIR")
    parser.add_argument("--format", default=os.getenv("QUANTIZATION_FORMAT", "fp8"),
                        help="quantization format the checkpoint should be in")
    parser.add_argument("--exclude", default=os.getenv("QUANT_EXCLUDE_PATTERNS", ""),
                        help="comma-separated patterns that must have stayed high precision")
    parser.add_argument("--no-calibration-expected", action="store_true",
                        help="skip the input_scale check, for a run made with CALIBRATE=0")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("no checkpoint given; pass a path or set CHECKPOINT_DIR")
    return args


def main() -> int:
    """Verify one checkpoint directory and return the exit code a Job should fail on."""
    args = parse_args()
    patterns = [p.strip() for p in args.exclude.split(",") if p.strip()]
    try:
        problems, notes = verify(args.checkpoint, args.format, patterns,
                                 not args.no_calibration_expected)
    except (ValueError, OSError) as exc:
        print(f"=== checkpoint verification: {args.checkpoint} ===")
        print(f"  PROBLEM: {exc}")
        return 1
    report(args.checkpoint, problems, notes)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
