#!/usr/bin/env python3
"""Fetch and pin the GSM8K subset the accuracy gate scores against.

The gate needs a fixed, versioned problem set: if the problems can drift, a score is not
comparable across runs and the gate is worthless. This downloads the official GSM8K test
split, verifies it byte-for-byte against a pinned digest, takes a deterministic seeded subset,
and writes app/bench/data/gsm8k_subset.jsonl, which is committed so the gate needs no network access at
run time. Re-running must reproduce an identical file; if it does not, either upstream changed
(the digest check catches that) or the subset constants were edited, which invalidates every
accuracy reference already captured. Source is openai/grade-school-math, MIT licensed — see
app/bench/data/ATTRIBUTION.md.

    python3 app/bench/src/fetch_eval_data.py            # write the subset
    python3 app/bench/src/fetch_eval_data.py --verify   # check it reproduces, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request

import config

SOURCE_URL = config.SOURCE_URL
SOURCE_SHA256 = config.SOURCE_SHA256
SOURCE_RECORDS = config.SOURCE_RECORDS
SUBSET_SEED = config.SUBSET_SEED
SUBSET_N = config.SUBSET_N
FINGERPRINT_N = config.FINGERPRINT_N


# ---------------------------------------------------------------------------- #
#                              SUBSET BUILDING                                 #
# ---------------------------------------------------------------------------- #

def extract_gold(answer: str) -> str:
    """Pull the final answer from a GSM8K solution, which ends with '#### <answer>'."""
    if "####" not in answer:
        raise ValueError(f"malformed GSM8K answer, no '####': {answer[:80]!r}")
    return answer.rsplit("####", 1)[1].strip().replace(",", "")


def build_subset(raw: bytes) -> list[dict]:
    """Take the deterministic subset, failing if upstream's record count changed.

    A seeded sample rather than the first N: the head of the file is not a representative
    slice, and a biased sample makes the score harder to reason about.
    """
    lines = raw.decode("utf-8").strip().split("\n")
    if len(lines) != SOURCE_RECORDS:
        raise SystemExit(f"expected {SOURCE_RECORDS} records upstream, got {len(lines)}")
    records = [json.loads(line) for line in lines]

    indices = list(range(len(records)))
    random.Random(SUBSET_SEED).shuffle(indices)
    return [{"id": f"gsm8k_test_{i}",
             "question": records[i]["question"],
             "gold": extract_gold(records[i]["answer"])}
            for i in sorted(indices[:SUBSET_N])]


def serialize(subset: list[dict]) -> str:
    """Render the subset with stable key order, so the committed file is byte-reproducible."""
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                   for record in subset)


def fetch_source() -> bytes:
    """Download the upstream split and verify it against the pinned digest."""
    print(f"fetching {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        sys.exit(f"FAIL: upstream SHA256 mismatch.\n"
                 f"  expected {SOURCE_SHA256}\n"
                 f"  got      {digest}\n"
                 f"Upstream data changed. Do not re-pin without regenerating every accuracy\n"
                 f"reference under docs/accuracy/, because scores are tied to this data.")
    print(f"  upstream sha256 OK ({len(raw)} bytes, {SOURCE_RECORDS} records)")
    return raw


# ---------------------------------------------------------------------------- #
#                               ENTRY POINT                                    #
# ---------------------------------------------------------------------------- #

def relative_out() -> str:
    """Render the output path relative to the repository root, for readable messages."""
    return str(config.EVAL_DATA.relative_to(config.PROJECT_ROOT))


def verify(text: str, digest: str) -> int:
    """Check the committed subset matches a fresh build, writing nothing."""
    current = config.EVAL_DATA.read_text(encoding="utf-8")
    if current != text:
        sys.exit(f"FAIL: {relative_out()} does not match a fresh build.\n"
                 f"  committed sha256 {hashlib.sha256(current.encode()).hexdigest()}\n"
                 f"  rebuilt   sha256 {digest}")
    print(f"  VERIFY OK: committed subset reproduces exactly ({digest})")
    return 0


def write(text: str, digest: str) -> int:
    """Write the subset and report what downstream references will be tied to."""
    config.EVAL_DATA.parent.mkdir(parents=True, exist_ok=True)
    config.EVAL_DATA.write_text(text, encoding="utf-8")
    print(f"  wrote {relative_out()}")
    print(f"  records        : {SUBSET_N} (seed {SUBSET_SEED}); "
          f"first {FINGERPRINT_N} used by the fingerprint tier")
    print(f"  subset sha256  : {digest}")
    print("\nCommit this file. Accuracy references are tied to its digest.")
    return 0


def main() -> int:
    """Build the pinned subset, or verify that the committed one still reproduces."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="verify the committed subset reproduces; write nothing")
    args = parser.parse_args()

    if args.verify and not config.EVAL_DATA.exists():
        sys.exit(f"FAIL: {relative_out()} does not exist; run without --verify first")

    text = serialize(build_subset(fetch_source()))
    digest = hashlib.sha256(text.encode()).hexdigest()
    return verify(text, digest) if args.verify else write(text, digest)


if __name__ == "__main__":
    sys.exit(main())
