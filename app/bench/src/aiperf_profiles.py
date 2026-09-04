"""Convert pinned prompt profiles to AIPerf custom-dataset JSONL.

One committed artifact drives both clients: the same profile that feeds `workload.py`
is rendered to the JSONL that `aiperf profile --input-file ... --custom-dataset-type
single_turn` consumes, one `{"text": ...}` object per line (schema confirmed against
aiperf 0.12.0's SingleTurnDatasetLoader).

The conversion cycles the prompt pool in file order to exactly `num_requests` lines,
mirroring `workload.build_prompts` (`pool[i % len(pool)]`), so both clients send the
same prompts in the same proportions. AIPerf is NOT given the pool to sample from,
because its sampling order would differ from the aiohttp harness's cycling and the two
runs would no longer be measuring the same workload.

Every line carries `output_length`, which aiperf's SingleTurnDatasetLoader maps to the
turn's max_tokens and sends as `max_completion_tokens` (loader/single_turn.py:142 in
0.12.0). Without it aiperf sends NO output cap at all and the server generates to EOS:
the 2026-08-10 calibration run decoded ~1,900 tokens per request against the profile's
150 budget, and its numbers were not comparable to throughput.py's on any level.
"""
from __future__ import annotations

import json
from pathlib import Path

import workload


def profile_to_jsonl(profile_name: str, num_requests: int, output_cap: int,
                     dest: Path) -> Path:
    """Render a pinned profile to a single_turn JSONL file with one line per request.

    An unknown profile raises `workload.load_profile`'s own FileNotFoundError, which
    names the available profiles: a run must never silently fall back to generated
    prompts when a pinned profile was asked for.

    `output_cap` is required, not defaulted: a caller that forgets it would silently
    produce the uncapped-generation bug this parameter exists to prevent.
    """
    if output_cap <= 0:
        raise ValueError(f"output_cap must be positive, got {output_cap}: without a "
                         "per-line output_length aiperf sends no token cap at all")
    profile = workload.load_profile(profile_name)
    pool = profile["prompts"]
    if not pool:
        raise ValueError(f"profile {profile_name!r} has no prompts")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for i in range(num_requests):
            fh.write(json.dumps({"text": pool[i % len(pool)],
                                 "output_length": output_cap},
                                ensure_ascii=False) + "\n")
    return dest


def expected_output_cap(profile_name: str) -> int:
    """The profile's pinned output budget, which overrides MAX_TOKENS everywhere."""
    return int(workload.load_profile(profile_name)["output_tokens"])
