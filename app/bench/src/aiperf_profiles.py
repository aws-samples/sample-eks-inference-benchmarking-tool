"""Convert pinned prompt profiles to the AIPerf custom-dataset JSONL that aiperf consumes, one text
object per line, so a single committed artifact drives both clients. The conversion cycles the
prompt pool in file order to exactly num_requests lines, mirroring workload.build_prompts, rather
than handing AIPerf the pool to sample from, because a different sampling order would stop the two
clients from measuring the same workload. Every line carries output_length, which AIPerf maps to
the turn's max_tokens; without it AIPerf sends no output cap and the server generates to EOS."""
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
