"""Every environment variable and measurement protocol the benchmark suite reads, in one place.

Nothing else under src/ calls os.getenv, and the check is mechanical: grep for it. Values are
read once at import so a setting cannot change midway through a run.
"""
import os
from pathlib import Path

# Repository root, derived from this file's location (app/bench/src/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The bench app's own root, which owns the data this harness reads. Resolved one level up rather
# than from PROJECT_ROOT so the paths hold both in the repo (app/bench/src/config.py -> app/bench)
# and inside the container image, where the app is copied to /app with no repo above it.
BENCH_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------- #
#                                 ENV READERS                                  #
# ---------------------------------------------------------------------------- #

def env_str(name: str, default: str) -> str:
    """Read a string variable, treating whitespace-only as unset."""
    return os.getenv(name, default).strip() or default


def env_int(name: str, default: int) -> int:
    """Read an integer variable."""
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    """Read a float variable."""
    return float(os.getenv(name, str(default)))


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean variable, accepting 1/true/yes in any case."""
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes")


def env_int_list(name: str, default: str) -> list[int]:
    """Read a comma-separated integer list, e.g. CONCURRENCY_SWEEP=1,8,32."""
    return [int(x) for x in env_str(name, default).split(",") if x.strip()]


def inherited_environment() -> dict[str, str]:
    """Copy the ambient environment, as the base for a harness subprocess's own settings."""
    return dict(os.environ)


# ---------------------------------------------------------------------------- #
#                             CANONICAL SWEEP                                  #
# ---------------------------------------------------------------------------- #

# The concurrency levels every closed-loop protocol must sweep. Enforced by
# validate_protocols() at import, so a protocol cannot quietly drop a level and produce a
# result that looks comparable to the others but is missing part of the curve.
CANONICAL_SWEEP = [1, 8, 32, 128, 256, 512]
CANONICAL_SWEEP_CSV = ",".join(str(level) for level in CANONICAL_SWEEP)


# ---------------------------------------------------------------------------- #
#                             MODEL IDENTITY                                   #
# ---------------------------------------------------------------------------- #

# Served model name as reported by /v1/models.
MODEL_ID = env_str("MODEL_ID", "/models/quantized/Qwen2.5-7B-Instruct-fp8")


# ---------------------------------------------------------------------------- #
#                                STORAGE PATHS                                 #
# ---------------------------------------------------------------------------- #

PROMPT_PROFILE_DIR = BENCH_ROOT / "data" / "prompt_profiles"
EVAL_DATA = BENCH_ROOT / "data" / "gsm8k_subset.jsonl"
ACCURACY_REF_DIR = PROJECT_ROOT / "docs" / "accuracy"
TOKEN_COUNT_DIR = PROJECT_ROOT / "docs" / "prompt_tokens"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "docs" / "results"


# ---------------------------------------------------------------------------- #
#                           ENDPOINT & WIRE PROTOCOL                           #
# ---------------------------------------------------------------------------- #

# Base URL of the vLLM-compatible endpoint.
VLLM_API_URL = env_str("VLLM_API_URL", "http://vllm-server:8000/v1")

# "chat" or "completions" — which OpenAI-compatible route to drive.
BENCH_ENDPOINT = env_str("BENCH_ENDPOINT", "chat").lower()

# Sampling temperature for the benchmark harnesses (accuracy gate uses 0).
BENCH_TEMPERATURE = env_float("BENCH_TEMPERATURE", 0.7)


# ---------------------------------------------------------------------------- #
#                                 RUN IDENTITY                                 #
# ---------------------------------------------------------------------------- #

# Hardware tag recorded in result file headers.
INSTANCE_TYPE = env_str("INSTANCE_TYPE", "unknown")

# Precision tag recorded in result file headers.
QUANTIZATION = env_str("QUANTIZATION", "fp8")

# Harness git revision, set by the in-cluster launcher (pods have no .git).
HARNESS_GIT_REV = os.getenv("HARNESS_GIT_REV", "").strip()


# ---------------------------------------------------------------------------- #
#                           WORKLOAD & PROMPT SHAPE                            #
# ---------------------------------------------------------------------------- #

# Pinned profile name from app/bench/data/prompt_profiles/ (preferred for published results).
PROMPT_PROFILE = env_str("PROMPT_PROFILE", "")

# Generated-prompt fallback mode: "unique", "legacy", or "shared_prefix".
PROMPT_MODE = env_str("PROMPT_MODE", "unique").lower()

# Target input length in tokens for generated prompts (0 = short natural prompts).
PROMPT_TOKENS = env_int("PROMPT_TOKENS", 0)

# RNG seed for generated prompts (makes runs reproducible).
PROMPT_SEED = env_int("PROMPT_SEED", 1337)

# Approximate tokens-per-word for English prose under a standard BPE tokenizer.
TOKENS_PER_WORD = 1.3

# Token budget reserved for the nonce and task stem before padding fills the rest.
PROMPT_OVERHEAD_TOKENS = 40


# ---------------------------------------------------------------------------- #
#                       THROUGHPUT HARNESS (CLOSED LOOP)                       #
# ---------------------------------------------------------------------------- #

# Concurrency levels to sweep.
THROUGHPUT_CONCURRENCY_SWEEP = env_int_list("CONCURRENCY_SWEEP", CANONICAL_SWEEP_CSV)

# Maximum output tokens per request.
THROUGHPUT_MAX_TOKENS = env_int("MAX_TOKENS", 150)

# Minimum requests per level (floor, not cap).
THROUGHPUT_NUM_REQUESTS = env_int("NUM_REQUESTS", 100)

# Minimum measurement duration in seconds per level.
THROUGHPUT_MIN_DURATION_S = env_float("MIN_DURATION_S", 60)

# Requests issued and discarded before the measurement window opens.
THROUGHPUT_WARMUP_REQUESTS = env_int("WARMUP_REQUESTS", 0)

# Size every level to at least this many whole waves of `concurrency` requests.
THROUGHPUT_MIN_WAVES = env_int("THROUGHPUT_MIN_WAVES", 2)

# Fraction of a level's requests that must succeed for the rep to count toward a median. A rep
# with any level below this is excluded, rather than contributing percentiles computed over the
# survivors of a partial failure.
MIN_LEVEL_SUCCESS = env_float("MIN_LEVEL_SUCCESS", 0.95)

# Share of a level's requests that may fail on TRANSPORT before the sweep is abandoned. Once
# the client's connections are failing at this rate the apparatus is broken, and every
# remaining level and rep produces unusable data — the 2026-08-08 latency run continued for two
# further hours after its tunnel had already collapsed.
TRANSPORT_ABORT_SHARE = env_float("TRANSPORT_ABORT_SHARE", 0.10)

# Guarantee each level reaches its labelled in-flight depth. On by default: a level with fewer
# requests than workers cannot fill, so the row would be labelled with a concurrency the run
# never applied. Set to 0 only to reproduce a pre-2026-08 unfilled run exactly.
THROUGHPUT_FILL_LEVELS = env_flag("THROUGHPUT_FILL_LEVELS", True)


# ---------------------------------------------------------------------------- #
#                        LATENCY HARNESS (CLOSED LOOP)                         #
# ---------------------------------------------------------------------------- #

# Concurrency levels to sweep.
LATENCY_CONCURRENCY_SWEEP = env_int_list("CONCURRENCY_SWEEP", CANONICAL_SWEEP_CSV)

# Maximum output tokens per request.
LATENCY_MAX_TOKENS = env_int("MAX_TOKENS", 256)

# Requests per level.
LATENCY_NUM_REQUESTS = env_int("NUM_REQUESTS", 50)


# ---------------------------------------------------------------------------- #
#                       ARRIVAL-RATE HARNESS (OPEN LOOP)                       #
# ---------------------------------------------------------------------------- #

# Maximum output tokens per request.
ARRIVAL_MAX_TOKENS = env_int("MAX_TOKENS", 256)

# Target Poisson arrival rate (requests/sec).
SERVER_TARGET_QPS = env_float("SERVER_TARGET_QPS", 1.0)

# SLA threshold for time-to-first-token (ms).
TTFT_SLA_MS = env_float("TTFT_SLA_MS", 500)

# SLA threshold for time-per-output-token (ms).
TPOT_SLA_MS = env_float("TPOT_SLA_MS", 50)

# Minimum query count before LoadGen can stop.
# LoadGen's Server-scenario early-stopping rule (loadgen 6.0.17 results.cc +
# early_stopping.cc) certifies a p99 only when queries_issued >= h + t, where t is the
# number of over-SLA samples and h is the smallest value with
# BetaRegularized(0.99, h, 1+t) <= 0.01. That means 459 queries with ZERO over-target
# samples, 662 with one, 1001 with three, 1307 with five, 2010 with ten. The former default
# of 200 could not certify any level no matter how well the server behaved, so every
# 2026-08-12 open-loop level came back UNDERSAMPLED and therefore LoadGen-INVALID.
#
# NO FLOOR CAN GUARANTEE CERTIFICATION, so do not read this number as one. Each additional
# over-target sample adds roughly 130 to the requirement while 100 extra queries buy only one
# more permitted sample, so the requirement always outruns the floor. What decides it is how
# many samples land past the target, and that is a property of the tail, not of the median: on
# 2026-08-14 a g7.24xlarge level whose TTFT p99 was 215 ms against a 500 ms target still failed
# to certify at 1000 queries, because a handful of far-tail samples crossed 500 ms.
#
# 1500 covers a level with about five stragglers and will not cover one with ten. Raising it
# lengthens every level, since a level runs until this count is reached, and at 1 QPS that is
# one second per query. If a level you intend to quote comes back uncertified, raise this for
# that run rather than assuming a bigger default would have been free.
MIN_QUERY_COUNT = env_int("MIN_QUERY_COUNT", 1500)

# Requests issued before LoadGen starts, absorbing one-time costs that would otherwise
# land in the certified percentiles. See arrival_rate.warm_up.
ARRIVAL_WARMUP_REQUESTS = env_int("WARMUP_REQUESTS", 0)

# Consecutive failures after which the open-loop SUT stops sending. Sized well above any
# plausible transient (the worst honest burst observed at 1 QPS was 13 of 200) and far below
# a level's query count, so it trips only when the server is genuinely gone.
ARRIVAL_ABORT_AFTER_FAILURES = env_int("ABORT_AFTER_FAILURES", 50)

# Minimum duration in seconds before LoadGen can stop.
ARRIVAL_MIN_DURATION_S = env_float("MIN_DURATION_S", 120)

# Directory for LoadGen's log output.
LOADGEN_OUTDIR = env_str("LOADGEN_OUTDIR", "loadgen_out")

# Run a single concurrency level (0 = full sweep).
CONCURRENCY = env_int("CONCURRENCY", 0)


# ---------------------------------------------------------------------------- #
#                            MEASUREMENT PROTOCOLS                             #
# ---------------------------------------------------------------------------- #

PROTOCOLS: dict[str, dict] = {
    "corpus": {
        "harness": "throughput",
        # Pinned input by default. output_tokens matches the protocol's own MAX_TOKENS, so
        # adopting it changes where the prompts come from, not the output shape.
        "profile": "legacy_corpus",
        "env": {
            "MAX_TOKENS": "150",
            "NUM_REQUESTS": "100",
            "MIN_DURATION_S": "0",
            "WARMUP_REQUESTS": "0",
            "PROMPT_MODE": "legacy",
            "PROMPT_TOKENS": "0",
            "CONCURRENCY_SWEEP": CANONICAL_SWEEP_CSV,
            "BENCH_ENDPOINT": "chat",
        },
        "note": "Corpus-comparable. For reasoning models this TRUNCATES output.",
    },
    "corpus_aiperf": {
        "harness": "aiperf",
        # The cross-client calibration protocol: the corpus sweep, same pinned profile and
        # canonical levels, driven by NVIDIA AIPerf instead of throughput.py. Exists so two
        # independent clients can measure the same server and their agreement (or gap) is
        # itself a recorded result. Requires: pip install -r app/bench/requirements-aiperf.txt.
        "profile": "legacy_corpus",
        "env": {
            "MAX_TOKENS": "150",
            "NUM_REQUESTS": "100",
            "WARMUP_REQUESTS": "0",
            "CONCURRENCY_SWEEP": CANONICAL_SWEEP_CSV,
            "BENCH_ENDPOINT": "chat",
        },
        "note": "CALIBRATION protocol (aiperf client). Compare against `corpus` on the same "
                "server; do not quote its numbers as corpus results.",
    },
    "reasoning": {
        "harness": "throughput",
        "profile": "legacy_reasoning",
        "env": {
            "MAX_TOKENS": "1024",
            "NUM_REQUESTS": "100",
            "MIN_DURATION_S": "90",
            "WARMUP_REQUESTS": "8",
            "PROMPT_MODE": "unique",
            "PROMPT_TOKENS": "0",
            "CONCURRENCY_SWEEP": CANONICAL_SWEEP_CSV,
            "BENCH_ENDPOINT": "chat",
        },
        "note": "Reasoning-model protocol. Reported tokens INCLUDE reasoning tokens.",
    },
    "latency": {
        "harness": "latency",
        "per_level": True,
        "levels": CANONICAL_SWEEP,
        "min_samples": 100,
        "waves": 2,
        "profile": "coding_completion",
        "env": {
            "MAX_TOKENS": "256",
            "PROMPT_MODE": "unique",
            "PROMPT_TOKENS": "0",
            "BENCH_ENDPOINT": "chat",
        },
        "note": "Streaming TTFT/TPOT/ITL across the full sweep.",
    },
    "latency_sharedprefix": {
        "harness": "latency",
        "per_level": True,
        "levels": CANONICAL_SWEEP,
        "min_samples": 100,
        "waves": 2,
        # No profile: this protocol requires a byte-identical shared prefix across prompts,
        # which a pinned profile of distinct prompts cannot express.
        "profile": None,
        "env": {
            "MAX_TOKENS": "256",
            "PROMPT_MODE": "shared_prefix",
            "PROMPT_TOKENS": "4000",
            "BENCH_ENDPOINT": "chat",
        },
        "note": "CACHE-HIT EXPERIMENT. Prefix caching must be ON server-side.",
    },
    "latency_longctx": {
        "harness": "latency",
        "ctx_levels": [4000, 8000, 16000, 28000],
        "concurrency": 8,
        "level_label": "ctx_tok",
        "min_samples": 100,
        "waves": 2,
        # No profile: this protocol sweeps input length, and a profile pins it to one value.
        "profile": None,
        "env": {
            "MAX_TOKENS": "256",
            "PROMPT_MODE": "unique",
            "BENCH_ENDPOINT": "chat",
        },
        "note": "Long-context TTFT at fixed concurrency=8.",
    },
    "server_qps": {
        "harness": "arrival_rate",
        # Levels bracket the SLA boundary rather than spanning orders of magnitude. The
        # 2026-08-12 run located the interactive limit between 2 and 8 QPS (p99 TTFT
        # 386 ms at 2, 499 ms at 4, 10 s at 8), so 3, 5 and 6 are where a capacity answer
        # is actually decided; 16 is kept as the one saturated level, which is what
        # establishes the throughput ceiling (~13.9 QPS achieved). 32 is dropped: it
        # measured the same ceiling with a 144 s p99, telling us nothing 16 did not.
        "qps_levels": [1, 2, 3, 4, 5, 6, 8, 16],
        # Seconds to let the server drain between levels; see suite.sweep_qps_levels.
        "settle_s": 60,
        "profile": "coding_completion",
        "env": {
            "MAX_TOKENS": "256",
            "PROMPT_MODE": "unique",
            "PROMPT_TOKENS": "0",
            "BENCH_ENDPOINT": "chat",
            "TTFT_SLA_MS": "500",
            "TPOT_SLA_MS": "50",
            # Covers a level with a few far-tail samples past the target. Not a guarantee: see
            # MIN_QUERY_COUNT in this file for why no floor is one.
            "MIN_QUERY_COUNT": "1500",
            "MIN_DURATION_S": "120",
            # Absorbs first-request costs that flipped an SLA verdict on 2026-08-12.
            "WARMUP_REQUESTS": "20",
        },
        "note": "OPEN LOOP (Poisson arrivals). NOT comparable to the concurrency sweep.",
    },
    "agentx": {
        "harness": "aiperf_trace",
        # The SemiAnalysis AgentX agentic-coding scenario: 40 concurrent multi-turn
        # sessions with think-time delays and KV-cache-faithful prefix structure, from
        # the committed trace (see app/bench/data/traces/README.md for the pin and the
        # 32k scaling rationale). The workload the aiohttp harness structurally cannot
        # express; supersedes the shared-prefix latency probe for the coding-assistant
        # question. Streaming metrics are aiperf's definitions — never comparable to
        # latency.py columns (docs/reports/aiperf-calibration/latency-findings.md).
        "profile": None,
        "env": {
            "AIPERF_MODE": "trace",
            "TRACE_FILE": "app/bench/data/traces/agentx_32k_seed42.jsonl",
            "TRACE_TYPE": "mooncake_trace",
            "BENCH_ENDPOINT": "chat",
        },
        "note": "AGENTIC REPLAY (aiperf, fixed schedule). Output budgets are per-record "
                "in the trace. NOT comparable to any concurrency-sweep protocol.",
    },
    "trace_replay": {
        "harness": "aiperf_trace",
        # Replays a user-supplied mooncake-format trace with its real arrival timing.
        # No committed trace: pass --trace <repo-relative-path>, with the file under
        # app/bench/data/traces/ so the launcher payload carries it in-cluster.
        "profile": None,
        "env": {
            "AIPERF_MODE": "trace",
            "TRACE_TYPE": "mooncake_trace",
            "BENCH_ENDPOINT": "chat",
        },
        "note": "TRACE REPLAY (aiperf, fixed schedule). The workload is whatever trace "
                "was passed; results are comparable only to replays of the SAME trace.",
    },
}

# Harness name -> module path, relative to the repository root.
HARNESS_PATHS = {
    "throughput": "app/bench/src/throughput.py",
    "latency": "app/bench/src/latency.py",
    "arrival_rate": "app/bench/src/arrival_rate.py",
    # Second closed-loop client, behind the same seam. Never hosts the canonical sweep:
    # corpus comparability and the fill-verification pedigree stay on throughput.py.
    "aiperf": "app/bench/src/aiperf_harness.py",
    # Same adapter in trace-replay mode (AIPERF_MODE=trace in the protocol env): one
    # streaming fixed-schedule replay of a committed trace, reported as a single row.
    # A distinct harness name because the suite's row regex, columns and level label
    # are keyed by harness, and a replay's columns are not the sweep's.
    "aiperf_trace": "app/bench/src/aiperf_harness.py",
}

# Extra dependency sets a harness needs beyond the suite's own. The launcher installs the
# named file inside the Job, so the pin recorded in the repo is the one that runs in-cluster.
HARNESS_EXTRA_REQUIREMENTS = {
    "aiperf": "app/bench/requirements-aiperf.txt",
    "aiperf_trace": "app/bench/requirements-aiperf.txt",
}

# Protocols that sweep an axis other than in-flight concurrency, and so are exempt from the
# canonical sweep. Named explicitly rather than inferred, so adding an exemption is a
# deliberate edit visible in review.
NON_CONCURRENCY_PROTOCOLS = {"latency_longctx", "server_qps", "agentx", "trace_replay"}


def swept_concurrency(protocol: dict) -> list[int] | None:
    """Return a protocol's concurrency levels, or None if it does not sweep concurrency."""
    if protocol.get("levels"):
        return list(protocol["levels"])
    sweep = protocol["env"].get("CONCURRENCY_SWEEP")
    return [int(x) for x in sweep.split(",") if x.strip()] if sweep else None


def validate_protocols() -> None:
    """Refuse to load if any concurrency-swept protocol deviates from the canonical sweep.

    A protocol that quietly drops a level yields a curve that looks comparable to the others
    while missing part of it — `corpus` shipped without 256 and its peak was read from 512.
    Raised at import so no code path can run against an invalid protocol table.
    """
    problems = []
    for name, protocol in PROTOCOLS.items():
        if name in NON_CONCURRENCY_PROTOCOLS:
            continue
        levels = swept_concurrency(protocol)
        if levels is None:
            problems.append(f"  {name}: sweeps concurrency but declares no levels")
        elif levels != CANONICAL_SWEEP:
            problems.append(f"  {name}: sweep {levels} != canonical {CANONICAL_SWEEP}")
        if "profile" not in protocol:
            problems.append(f"  {name}: no 'profile' key; declare a pinned profile or None")
    if problems:
        raise ValueError(
            "PROTOCOLS failed validation:\n" + "\n".join(problems)
            + f"\n\nEvery closed-loop protocol must sweep {CANONICAL_SWEEP}. Protocols that "
              "sweep a\ndifferent axis must be listed in NON_CONCURRENCY_PROTOCOLS.")


validate_protocols()


# ---------------------------------------------------------------------------- #
#                              ACCURACY GATE                                   #
# ---------------------------------------------------------------------------- #

# Schema version for accuracy reference files.
ACCURACY_SCHEMA = 1

# Deterministic generation settings for the accuracy gate (not the benchmark harnesses).
ACCURACY_GEN = {"temperature": 0, "top_p": 1, "seed": 1337}

# Number of problems used by the fingerprint tier.
FINGERPRINT_N = 20

# Max output tokens for the fingerprint tier.
FINGERPRINT_MAX_TOKENS = 256

# Max output tokens for the score tier.
SCORE_MAX_TOKENS = 512

# Per-item completion_tokens drift allowed for a reproducible reference.
TOKEN_DRIFT_TOLERANCE = 0.02

# Aggregate mean token drift allowed for a non-reproducible (MoE) reference.
NONREPRO_MEAN_TOKEN_TOLERANCE = 0.20

# Extra truncation RATE, in percentage points, allowed for a non-reproducible reference.
NONREPRO_TRUNCATION_TOLERANCE = 0.20

# Distinct-n-gram ratio below which output is treated as collapsed repetition.
MIN_DISTINCT_NGRAM_RATIO = 0.30

# N-gram size for the repetition collapse check.
NGRAM_N = 8

# Fraction of responses that may hit the token ceiling before a score is unsafe.
MAX_TRUNCATED_FRACTION = 0.20


# ---------------------------------------------------------------------------- #
#                              ARRIVAL RATE                                    #
# ---------------------------------------------------------------------------- #

# Sample pool size for LoadGen (must exceed MIN_QUERY_COUNT to avoid repeats).
QSL_SIZE = max(MIN_QUERY_COUNT * 2, 512)

# Hosts that indicate a port-forward tunnel (adds ~150-200ms to TTFT).
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")


# ---------------------------------------------------------------------------- #
#                             SUITE RUNNER                                     #
# ---------------------------------------------------------------------------- #

# Client CPU fraction above which a level may be measuring the harness.
CLIENT_CPU_SUSPECT = 0.85

# Max run-to-run spread (fraction of median) before a cell is distrusted.
MAX_SPREAD = 0.10


# ---------------------------------------------------------------------------- #
#                            EVAL DATA PINNING                                 #
# ---------------------------------------------------------------------------- #

# Upstream GSM8K test split URL.
SOURCE_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
              "master/grade_school_math/data/test.jsonl")

# Pinned SHA256 of the upstream test split.
SOURCE_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"

# Expected number of records in the upstream split.
SOURCE_RECORDS = 1319

# Seed for the deterministic subset selection.
SUBSET_SEED = 1337

# Number of problems in the pinned subset.
SUBSET_N = 200


# ---------------------------------------------------------------------------- #
#                           PROFILE GENERATION                                 #
# ---------------------------------------------------------------------------- #

# Fraction of prompts deliberately duplicated for cache-effect measurement.
DUPLICATE_FRACTION = 0.08

# Characters per token by content type (measured against GLM-5.2's tokenizer).
CHARS_PER_TOKEN = {"prose": 4.35, "code": 3.20}


# ---------------------------------------------------------------------------- #
#                           PROFILE VERIFICATION                               #
# ---------------------------------------------------------------------------- #

# How far a measured median may sit from the profile's stated target, for the reference tokenizer.
PROFILE_TOLERANCE = 0.15

# The tokenizer the generator's CHARS_PER_TOKEN ratios are calibrated against, and therefore the
# only one whose deviation means the generator is wrong. Chosen on measurement, not preference:
# across all eight profiles this tokenizer lands within 3.3% of every stated target, while Qwen
# runs +2% to +12% and Gemma 4 up to +26% on the code profiles.
#
# The tolerance cannot be applied to every tokenizer, because the same pinned text differs by 27
# percentage points between these families and no single nominal target can sit within 15% of all
# of them. Deviation for a non-reference tokenizer is recorded in docs/prompt_tokens/ and reported
# by each run's protocol lock as a measured value, which is where it actually matters.
PROFILE_REFERENCE_TOKENIZER = "nvidia/GLM-5.2-NVFP4"


# ---------------------------------------------------------------------------- #
#                          MANIFEST VALIDATION                                 #
# ---------------------------------------------------------------------------- #

# Glob for the K8s manifests to validate. Every recipe, not a subset: all of them build their
# vllm serve invocation as an inline shell script, so all of them carry the truncation risk.
MANIFEST_GLOB = "vllm-recipes/*/*.yaml"

# Flags each manifest MUST pass through to vllm serve, keyed by path relative to vllm-recipes/.
# Two models share a basename (fp8-g7e.4xlarge.yaml, fp8-p5en.48xlarge.yaml), so the model
# directory is part of the key.
# Every manifest MANIFEST_GLOB matches must appear here; an unlisted one fails rather than
# being skipped, so renaming a recipe cannot silently disable its flag assertions.
REQUIRED_FLAGS = {
    "kimi-k3/mxfp4-p6-b300.48xlarge-baseline.yaml": [
        "--max-cudagraph-capture-size", "512", "--kv-cache-dtype", "fp8",
        "--attention-config", "--reasoning-parser", "--port", "8000",
    ],
    "kimi-k3/mxfp4-p6-b300.48xlarge-dspark.yaml": [
        "--speculative-config", "--max-cudagraph-capture-size", "512",
        "--kv-cache-dtype", "fp8", "--reasoning-parser", "--port", "8000",
    ],
    "kimi-k3/mxfp4-p6-b300.48xlarge-ep.yaml": [
        "--enable-expert-parallel", "--all2all-backend", "--moe-backend",
        "--max-cudagraph-capture-size", "512", "--port", "8000",
    ],
}
