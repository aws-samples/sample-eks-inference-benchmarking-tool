"""Every environment variable and measurement protocol the benchmark suite reads, in one place, so
nothing else under src/ calls os.getenv and the check stays mechanical. Values are read once at
import so a setting cannot change midway through a run."""
import os
from pathlib import Path

# Repository root, derived from this file's location (app/bench/src/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Resolved one level up rather than from PROJECT_ROOT so the paths hold in the repo and in the image.
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

# Levels every closed-loop protocol must sweep; validate_protocols() enforces it so none can drop one.
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

# Success fraction a level needs for its rep to count, so a partial failure excludes the rep entirely.
MIN_LEVEL_SUCCESS = env_float("MIN_LEVEL_SUCCESS", 0.95)

# Transport-failure share that abandons the sweep, since a failing client makes every later rep unusable.
TRANSPORT_ABORT_SHARE = env_float("TRANSPORT_ABORT_SHARE", 0.10)

# Guarantees a level reaches its labelled depth, so a row cannot claim a concurrency never applied.
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

# LoadGen floor, not a certification guarantee: the far tail sets the requirement, so raise it per run.
MIN_QUERY_COUNT = env_int("MIN_QUERY_COUNT", 1500)

# Requests issued before LoadGen starts, keeping one-time costs out of the certified percentiles.
ARRIVAL_WARMUP_REQUESTS = env_int("WARMUP_REQUESTS", 0)

# Consecutive failures that stop the open-loop SUT, above any transient burst and below a query count.
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
        # Pinned input by default; output_tokens matches MAX_TOKENS, so only the prompt source changes.
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
        # Cross-client calibration: the corpus sweep driven by NVIDIA AIPerf instead of throughput.py.
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
        # No profile, because a byte-identical shared prefix cannot come from a set of distinct prompts.
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
        # Levels bracket the SLA boundary, plus one saturated level to establish the throughput ceiling.
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
            # Covers a few far-tail samples past the target; see MIN_QUERY_COUNT for why no floor guarantees this.
            "MIN_QUERY_COUNT": "1500",
            "MIN_DURATION_S": "120",
            # Absorbs first-request costs that flipped an SLA verdict on 2026-08-12.
            "WARMUP_REQUESTS": "20",
        },
        "note": "OPEN LOOP (Poisson arrivals). NOT comparable to the concurrency sweep.",
    },
    "agentx": {
        "harness": "aiperf_trace",
        # AgentX agentic-coding trace: multi-turn sessions with think-time gaps and cache-faithful prefixes.
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
        # Replays a user-supplied mooncake trace, so --trace is required and the file must sit under traces/.
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
    # Second closed-loop client; never hosts the canonical sweep, which stays on throughput.py.
    "aiperf": "app/bench/src/aiperf_harness.py",
    # Same adapter replaying a trace as one row, named apart because the suite keys columns by harness.
    "aiperf_trace": "app/bench/src/aiperf_harness.py",
}

# Extra requirements a harness needs, installed by the launcher so the repo's pin runs in-cluster.
HARNESS_EXTRA_REQUIREMENTS = {
    "aiperf": "app/bench/requirements-aiperf.txt",
    "aiperf_trace": "app/bench/requirements-aiperf.txt",
}

# Protocols sweeping another axis, listed explicitly so an exemption is a visible deliberate edit.
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

# Reference tokenizer the CHARS_PER_TOKEN ratios are calibrated against, chosen on measured deviation.
PROFILE_REFERENCE_TOKENIZER = "nvidia/GLM-5.2-NVFP4"


# ---------------------------------------------------------------------------- #
#                          MANIFEST VALIDATION                                 #
# ---------------------------------------------------------------------------- #

# Every recipe, not a subset: all build their vllm serve line in shell, so all carry truncation risk.
MANIFEST_GLOB = "vllm-recipes/*/*.yaml"

# Flags each manifest must reach vllm serve with, keyed by path since two models share a basename.
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
