"""Prompt supply for the harnesses, preferring a pinned profile and falling back to generated prompts.
A profile named in PROMPT_PROFILE serves committed text and is preferred for anything published,
because what was sent is answered by opening a file, every model receives byte-identical input so a
difference between two models belongs to the models, and the input size stops being implicit, which
is how a 41-token input once came to be published as though it were 1,024. Without a profile,
unique gives every request a distinct leading nonce so prefix caching cannot make prefill free and
overstate tokens per second, shared_prefix deliberately measures cache benefit, and legacy
reproduces the original five fixed prompts for continuity with the earlier published corpus only."""
import json
import random
import uuid

import config

# The original fixed set, retained for PROMPT_MODE=legacy reproducibility only.
LEGACY_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to reverse a linked list.",
    "What are the main differences between Kubernetes and Docker Swarm?",
    "Summarize the plot of The Matrix.",
    "How does a transformer neural network work?",
]

# Stems asking for open-ended generation, so max_tokens governs length rather than an early stop.
TASK_STEMS = [
    "Explain in detail how {topic} works, covering the underlying mechanics.",
    "Write a thorough technical overview of {topic}, including tradeoffs.",
    "Compare and contrast the main approaches to {topic} and justify a recommendation.",
    "Describe the failure modes of {topic} and how to mitigate each one.",
    "Walk through the design of a production system for {topic}, step by step.",
]

TOPICS = [
    "distributed consensus", "GPU memory hierarchies", "columnar storage engines",
    "mixture-of-experts routing", "TLS session resumption", "lock-free queues",
    "query planner cost models", "container network interfaces", "write-ahead logging",
    "vector index construction", "speculative decoding", "CRDT merge semantics",
    "cache coherence protocols", "gradient checkpointing", "service mesh sidecars",
    "copy-on-write filesystems", "branch prediction", "erasure coding",
]

# Domain-flavoured padding, since tokenizers behave differently on gibberish and skew prefill.
FILLER_VOCAB = (
    "system latency throughput request buffer kernel scheduler memory bandwidth cache "
    "partition replica commit offset segment compaction index shard cluster node worker "
    "queue backlog checkpoint snapshot lease quorum heartbeat timeout retry backoff "
    "pipeline batch tensor gradient parameter activation embedding attention routing "
    "expert token sequence context window prefill decode allocation fragmentation"
).split()

# Rough tokens-per-word and overhead budget come from config.
from config import TOKENS_PER_WORD, PROMPT_OVERHEAD_TOKENS


# ---------------------------------------------------------------------------- #
#                               PINNED PROFILES                                #
# ---------------------------------------------------------------------------- #

def load_profile(name: str | None = None) -> dict | None:
    """Load the named pinned profile, or None when no profile is selected.

    Raises rather than falling back to generated prompts if a named profile is missing or
    empty: a silent fallback would produce a run labelled with a profile it never used, which
    is worse than failing.
    """
    name = config.PROMPT_PROFILE if name is None else name
    if not name:
        return None
    path = config.PROMPT_PROFILE_DIR / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in config.PROMPT_PROFILE_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"prompt profile {name!r} not found at {path}. Available: {available}")
    doc = json.loads(path.read_text())
    if not doc.get("prompts"):
        raise ValueError(f"prompt profile {name!r} contains no prompts")
    return doc


# ---------------------------------------------------------------------------- #
#                              GENERATED PROMPTS                               #
# ---------------------------------------------------------------------------- #

def filler(num_tokens: int, rng: random.Random) -> str:
    """Generate roughly num_tokens worth of varied prose padding."""
    if num_tokens <= 0:
        return ""
    num_words = max(1, int(num_tokens / TOKENS_PER_WORD))
    return " ".join(rng.choice(FILLER_VOCAB) for _ in range(num_words))


def task_prompt(rng: random.Random) -> str:
    """Draw one open-ended task stem with a topic substituted in."""
    return rng.choice(TASK_STEMS).format(topic=rng.choice(TOPICS))


def unique_prompt(rng: random.Random, target_tokens: int) -> str:
    """Build one prompt whose leading tokens differ from every other prompt's.

    The nonce comes FIRST because prefix caching matches on shared leading tokens, so a
    leading nonce defeats it even when the rest of the text is similar.
    """
    head = f"[req:{uuid.uuid4().hex}] {task_prompt(rng)}"
    if target_tokens <= 0:
        return head
    body = filler(max(target_tokens - PROMPT_OVERHEAD_TOKENS, 0), rng)
    return f"{head}\n\nBackground material:\n{body}"


def shared_prefix_prompt(prefix: str, rng: random.Random) -> str:
    """Build one prompt sharing a byte-identical cacheable prefix with a unique tail."""
    return f"{prefix}\n\n{task_prompt(rng)} (ref {uuid.uuid4().hex[:12]})"


def resolve_generation(mode: str | None, target_tokens: int | None, seed: int | None):
    """Resolve the generated-prompt settings, falling back to the configured defaults."""
    mode = (mode or config.PROMPT_MODE).lower()
    if mode not in ("unique", "legacy", "shared_prefix"):
        raise ValueError(f"unknown PROMPT_MODE {mode!r}; expected unique|legacy|shared_prefix")
    tokens = config.PROMPT_TOKENS if target_tokens is None else target_tokens
    rng = random.Random(config.PROMPT_SEED if seed is None else seed)
    return mode, tokens, rng


# ---------------------------------------------------------------------------- #
#                               PUBLIC INTERFACE                               #
# ---------------------------------------------------------------------------- #

def build_prompts(num_requests: int, mode: str | None = None,
                  target_tokens: int | None = None, seed: int | None = None) -> list[str]:
    """Return num_requests prompts as a list, for harnesses that need them up front.

    A profile takes precedence over every generated mode and is cycled in file order, so a run
    of N requests draws deterministically from the pinned pool.
    """
    profile = load_profile()
    if profile is not None:
        pool = profile["prompts"]
        return [pool[i % len(pool)] for i in range(num_requests)]

    mode, tokens, rng = resolve_generation(mode, target_tokens, seed)
    if mode == "legacy":
        repeats = num_requests // len(LEGACY_PROMPTS) + 1
        return (LEGACY_PROMPTS * repeats)[:num_requests]
    if mode == "shared_prefix":
        # Built once, so every request shares byte-identical leading tokens.
        prefix = filler(max(tokens, 1024), rng)
        return [shared_prefix_prompt(prefix, rng) for _ in range(num_requests)]
    return [unique_prompt(rng, tokens) for _ in range(num_requests)]


def prompt_stream(mode: str | None = None, target_tokens: int | None = None,
                  seed: int | None = None):
    """Yield prompts indefinitely, for the sustained-load path that cannot know its count.

    A long run may issue far more requests than are known up front, and at PROMPT_TOKENS in
    the 10k-100k range materialising a list would cost hundreds of MB of client memory.
    Single-threaded asyncio only — do not share one generator across threads.
    """
    profile = load_profile()
    if profile is not None:
        pool = profile["prompts"]
        index = 0
        while True:
            yield pool[index % len(pool)]
            index += 1

    mode, tokens, rng = resolve_generation(mode, target_tokens, seed)
    if mode == "legacy":
        index = 0
        while True:
            yield LEGACY_PROMPTS[index % len(LEGACY_PROMPTS)]
            index += 1
    if mode == "shared_prefix":
        prefix = filler(max(tokens, 1024), rng)
        while True:
            yield shared_prefix_prompt(prefix, rng)
    while True:
        yield unique_prompt(rng, tokens)


def describe_workload() -> str:
    """One-line provenance string naming the prompt source, for the run header."""
    profile = load_profile()
    if profile is not None:
        return describe_profile(profile)
    length = f"~{config.PROMPT_TOKENS} tok (approx)" if config.PROMPT_TOKENS > 0 else "short"
    warning = ""
    if config.PROMPT_MODE == "legacy":
        warning = "  [WARNING: repeating prompts — invalid for throughput claims if caching is on]"
    elif config.PROMPT_MODE == "shared_prefix":
        warning = "  [cache-hit experiment — not a baseline throughput number]"
    return (f"generated {config.PROMPT_MODE}, input {length}, "
            f"seed {config.PROMPT_SEED}{warning}")


def describe_profile(profile: dict) -> str:
    """Describe a pinned profile, flagging shapes that must not be quoted as representative."""
    duplicates = len(profile.get("intentional_duplicate_indices", []))
    warning = ""
    if profile.get("legacy"):
        warning = "  [LEGACY SHAPE — historical continuity only, not representative]"
    if profile.get("fixed_prompt_set"):
        warning += "  [fixed prompt set: repeats, so invalid as throughput if caching is on]"
    return (f"profile {profile['name']}, {profile['prompt_count']} pinned prompts "
            f"({profile['distinct_prompt_count']} distinct, {duplicates} declared duplicates), "
            f"nominal input {profile['target_input_tokens']} tok, "
            f"output cap {profile['output_tokens']}{warning}")
