"""Regression checks for the correctness gate's aggregate comparisons, runnable without a GPU.

Every check here encodes a defect that reached a real run when the gate aborted a
GLM-5.2 NVFP4 sweep on a p6-b200 capacity block.

  truncation counted, not rated
      `compare_aggregates` compared the number of responses hitting the token ceiling against
      the reference with a flat allowance of `reference + max(1, n // 10)`, which is +2 on a
      20-problem fingerprint, and applied it whether or not the server was bit-reproducible. On
      a non-reproducible server the text of 18 of 20 responses changes between identical passes
      at temperature 0, so completion_tokens change too, so whether a response lands exactly on
      the ceiling is a coin flip. The count is a random variable with a standard deviation near
      1.6 at a 15% truncation rate, and a +2 allowance sits inside its noise. The reference
      recorded 2/20, a re-run of the same pod minutes later got 5/20, and the gate called it a
      behaviour change. Every reference in the repo is non-reproducible, so every model was
      exposed; qwen3-235b worst at about one clean run in eight.

      The fix compares the RATE with a percentage-point allowance when reproducible is false,
      mirroring how mean completion_tokens drift was already handled two lines below, and keeps
      the strict absolute count for reproducible servers where it is meaningful.

  the failure it must still catch
      A hybrid-thinking model left in its default mode spends its whole budget on the reasoning
      trace and truncates about 90% of responses. That is the defect the check exists for, it
      was observed on this same model on 2026-08-20, and it must still fail. The checks below
      pin both edges: noise passes, 90% truncation fails.

  what the binomial numbers are worth
      The false-trip figures printed below assume each response truncates independently at the
      reference rate. Live sampling on gemma4-31b-fp8 (g7.24xlarge, 2026-08-20, eight runs of
      the fingerprint set) gave counts [5, 5, 4, 4, 4, 3, 3, 4], a standard deviation of 0.76
      against the model's 1.79, because truncation is mostly decided by the question rather
      than by sampling. Treat them as an upper bound on risk. They are asserted only as
      strictly-better-and-small, never quoted as the measured failure rate.

These run against synthetic item lists and the committed references. No server, no GPU.

Usage:
    python3 main.py verify-accuracy-gate
"""
from __future__ import annotations

import glob
import json
import os
import sys
from math import comb
from pathlib import Path

# Set before importing config, which freezes its environment at import time.
CHECK_ENV = {
    "MODEL_ID": "mock-model",
    "INSTANCE_TYPE": "verify-local",
    "QUANTIZATION": "none",
}

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and print one check result; failures accumulate rather than abort."""
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def items(n: int, truncated: int, tokens: int = 200) -> list[dict]:
    """Build a synthetic item list with a given number of ceiling-truncated responses."""
    return [
        {"id": f"p{i}",
         "finish_reason": "length" if i < truncated else "stop",
         "completion_tokens": tokens}
        for i in range(n)
    ]


def trip_probability(n: int, reference_truncated: int, allowance: int) -> float:
    """Chance a clean re-run breaches `allowance` extra truncations, binomial model.

    Models each response as independently truncating at the reference's observed rate, which is
    the assumption the old absolute allowance implicitly violated.
    """
    p = reference_truncated / n
    ceiling = reference_truncated + allowance
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(ceiling + 1, n + 1))


def static_checks() -> None:
    """Behaviour of compare_aggregates, and the arithmetic that justifies the tolerance."""
    import accuracy
    import config

    print("tolerance behaviour:")
    n = config.FINGERPRINT_N
    reference = items(n, 2)

    # The exact case that aborted the p6-b200 run: reference 2/20, re-run 5/20, unchanged server.
    failures = accuracy.compare_aggregates(reference, items(n, 5), reproducible=False)
    check("the 2/20 -> 5/20 swing that lost a capacity block now passes",
          not failures, "; ".join(failures))

    # The same swing on a bit-reproducible server is still a real signal and must still fail.
    failures = accuracy.compare_aggregates(reference, items(n, 5), reproducible=True)
    check("the same swing still fails when the server IS reproducible",
          any("truncation increased" in f for f in failures), "; ".join(failures))

    # The defect the check exists for: thinking left on, ~90% truncated.
    failures = accuracy.compare_aggregates(reference, items(n, 18), reproducible=False)
    check("a thinking-left-on server (18/20 truncated) still fails",
          any("truncation rate rose" in f for f in failures), "; ".join(failures))

    # The boundary, stated explicitly, including the max(1, n//10) floor, so an edit cannot move it unnoticed.
    floor_items = max(1, n // 10)
    rate_items = int(config.NONREPRO_TRUNCATION_TOLERANCE * n)
    allowance_items = max(rate_items, floor_items)
    edge = 2 + allowance_items
    check(f"at the tolerance edge ({edge}/{n}) a non-reproducible server passes",
          not accuracy.compare_aggregates(reference, items(n, edge), reproducible=False))
    check(f"one past the edge ({edge + 1}/{n}) a non-reproducible server fails",
          any("truncation rate rose" in f
              for f in accuracy.compare_aggregates(reference, items(n, edge + 1),
                                                   reproducible=False)))

    # Small-n floor: on n=5 both the rate (20% = 1 item) and the floor (max(1, 5//10) = 1) agree.
    small_n = 5
    small_ref = items(small_n, 0)
    # Floor is max(1, 5//10) = 1 item, so 0 -> 1 passes, 0 -> 2 fails.
    check(f"small-n floor: 0/5 -> 1/5 passes (floor = 1 item)",
          not accuracy.compare_aggregates(small_ref, items(small_n, 1), reproducible=False))
    check(f"small-n floor: 0/5 -> 2/5 fails (exceeds floor)",
          any("truncation rate rose" in f
              for f in accuracy.compare_aggregates(small_ref, items(small_n, 2),
                                                   reproducible=False)))

    # Mean-token drift must keep working independently of the truncation change.
    failures = accuracy.compare_aggregates(
        items(n, 2, tokens=200), items(n, 2, tokens=400), reproducible=False)
    check("mean completion_tokens drift is still caught",
          any("mean completion_tokens" in f for f in failures), "; ".join(failures))

    # An empty reference must not raise, since a malformed file should fail loudly elsewhere.
    check("empty item lists do not raise",
          accuracy.compare_aggregates([], [], reproducible=False) == [])

    # UPPER BOUND, not a measurement: the binomial model assumes independent truncation, which live sampling contradicts.
    print("\nbinomial UPPER BOUND on false-trip rate, old rule vs new:")
    old_allowance = max(1, n // 10)
    for path in sorted(glob.glob(str(config.ACCURACY_REF_DIR / "*.json"))):
        reference_file = json.loads(Path(path).read_text(encoding="utf-8"))
        fingerprint = reference_file.get("fingerprint")
        if not fingerprint:
            continue
        truncated = sum(1 for i in fingerprint["items"] if i.get("finish_reason") == "length")
        size = len(fingerprint["items"])
        before = trip_probability(size, truncated, old_allowance)
        after = trip_probability(size, truncated, int(config.NONREPRO_TRUNCATION_TOLERANCE * size))
        name = Path(path).name
        check(f"{name}: bound {before:.1%} -> {after:.1%}",
              after < before and after < 0.05,
              f"ref={truncated}/{size} after={after:.1%}")

    print("\nguards on the tolerance itself:")
    check("tolerance is a rate, not a count (0 < value < 1)",
          0 < config.NONREPRO_TRUNCATION_TOLERANCE < 1,
          str(config.NONREPRO_TRUNCATION_TOLERANCE))
    check("tolerance leaves room to catch a thinking-left-on server",
          config.NONREPRO_TRUNCATION_TOLERANCE < 0.5,
          "a tolerance of 50 points or more would admit the defect this check exists for")


def main() -> int:
    """Run all checks; exit nonzero if any FAILED."""
    for key, value in CHECK_ENV.items():
        os.environ.setdefault(key, value)
    static_checks()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
