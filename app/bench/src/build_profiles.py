#!/usr/bin/env python3
"""Generate the pinned prompt-profile JSON files that every benchmark run draws its input from,
storing prompts as literal text rather than a target length plus a per-model padding knob. Pinning
the text means a throughput or latency difference between two models is a property of the models
rather than of their prompt streams, where the earlier design served every model different text.
Token counts still differ slightly per tokenizer for identical text, which is expected and recorded
per model by verify_profiles.py rather than fed back into generation. Content is composed from
fixed building blocks under a fixed seed, so re-running reproduces the files byte for byte; the
JSON is the artifact, so run this only to add a profile or change a shape."""
from __future__ import annotations

import argparse
import json
import random
import re

import config

OUT = config.PROMPT_PROFILE_DIR
SCHEMA = "prompt-profile/v2"

DUPLICATE_FRACTION = config.DUPLICATE_FRACTION
CHARS_PER_TOKEN = config.CHARS_PER_TOKEN

# ---------------------------------------------------------------------------- #
#                               BUILDING BLOCKS                                #
# ---------------------------------------------------------------------------- #

# Synthesised technical content, not copied from any source.

CODE_UNITS = [
    '''def merge_windows(windows, tolerance=0):
    """Collapse overlapping intervals, treating gaps <= tolerance as contiguous."""
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(w) for w in merged]
''',
    '''class RetryBudget:
    """Token-bucket retry limiter. Rejects retries once the budget is exhausted."""

    def __init__(self, capacity, refill_per_second):
        self._capacity = capacity
        self._refill = refill_per_second
        self._tokens = float(capacity)
        self._last = None

    def allow(self, now):
        if self._last is None:
            self._last = now
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._refill)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False
''',
    '''func (s *Store) Compact(ctx context.Context, before time.Time) (int, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin: %w", err)
	}
	defer tx.Rollback()

	res, err := tx.ExecContext(ctx, `DELETE FROM events WHERE created_at < ?`, before)
	if err != nil {
		return 0, fmt.Errorf("delete: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("rows: %w", err)
	}
	return int(n), tx.Commit()
}
''',
    '''export function useDebounced<T>(value: T, delayMs: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const handle = window.setTimeout(() => setSettled(value), delayMs);
    return () => window.clearTimeout(handle);
  }, [value, delayMs]);
  return settled;
}
''',
    '''def shard_for(key: str, shards: int, *, replicas: int = 1) -> list[int]:
    """Rendezvous hashing: pick `replicas` shards for a key, stable under resize."""
    scored = []
    for shard in range(shards):
        digest = hashlib.blake2b(f"{shard}:{key}".encode(), digest_size=8).digest()
        scored.append((int.from_bytes(digest, "big"), shard))
    scored.sort(reverse=True)
    return [shard for _, shard in scored[:replicas]]
''',
    '''CREATE TABLE ledger_entry (
    id           BIGSERIAL PRIMARY KEY,
    account_id   BIGINT      NOT NULL REFERENCES account (id),
    amount_minor BIGINT      NOT NULL,
    currency     CHAR(3)     NOT NULL,
    posted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    idem_key     TEXT        NOT NULL,
    CONSTRAINT ledger_entry_idem UNIQUE (account_id, idem_key)
);

CREATE INDEX ledger_entry_account_posted
    ON ledger_entry (account_id, posted_at DESC)
    INCLUDE (amount_minor);
''',

    '''def coalesce_batches(items, max_bytes, max_items):
    """Group items into batches bounded by both byte size and count."""
    batch, size = [], 0
    for item in items:
        encoded = len(item.payload)
        if batch and (size + encoded > max_bytes or len(batch) >= max_items):
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += encoded
    if batch:
        yield batch
''',
    '''type Limiter struct {
	mu       sync.Mutex
	windows  map[string][]time.Time
	limit    int
	interval time.Duration
}

func (l *Limiter) Permit(key string, now time.Time) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	cutoff := now.Add(-l.interval)
	kept := l.windows[key][:0]
	for _, t := range l.windows[key] {
		if t.After(cutoff) {
			kept = append(kept, t)
		}
	}
	if len(kept) >= l.limit {
		l.windows[key] = kept
		return false
	}
	l.windows[key] = append(kept, now)
	return true
}
''',
    '''async function drainQueue(queue, handler, { concurrency = 4 } = {}) {
  const inFlight = new Set();
  for await (const job of queue) {
    const task = handler(job).finally(() => inFlight.delete(task));
    inFlight.add(task);
    if (inFlight.size >= concurrency) {
      await Promise.race(inFlight);
    }
  }
  await Promise.all(inFlight);
}
''',
    '''def reconcile(desired, observed, *, prune=True):
    """Return the operations needed to move observed toward desired."""
    ops = []
    for name, spec in desired.items():
        current = observed.get(name)
        if current is None:
            ops.append(("create", name, spec))
        elif current != spec:
            ops.append(("update", name, spec))
    if prune:
        for name in observed.keys() - desired.keys():
            ops.append(("delete", name, None))
    return sorted(ops, key=lambda op: (op[0], op[1]))
''',
    '''WITH ranked AS (
    SELECT
        session_id,
        event_name,
        occurred_at,
        ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY occurred_at) AS step
    FROM analytics_event
    WHERE occurred_at >= now() - INTERVAL \'7 days\'
)
SELECT step, event_name, COUNT(*) AS sessions
FROM ranked
WHERE step <= 5
GROUP BY step, event_name
ORDER BY step, sessions DESC;
''',
    '''impl Backoff {
    fn next(&mut self) -> Duration {
        let exp = self.base.as_millis() as u64 * (1 << self.attempt.min(10));
        let capped = exp.min(self.ceiling.as_millis() as u64);
        self.attempt = self.attempt.saturating_add(1);
        let jitter = self.rng.gen_range(0..=capped / 2);
        Duration::from_millis(capped / 2 + jitter)
    }
}
''',
]

PROSE_UNITS = [
    "Write-ahead logging decouples durability from data-file layout. A committed "
    "transaction is durable once its log records reach stable storage, which lets the "
    "buffer manager defer page writes and batch them by locality rather than by commit "
    "order. Recovery replays the log forward from the last checkpoint, then rolls back "
    "transactions that never committed.",

    "A service mesh sidecar terminates connections on behalf of the workload, which moves "
    "retry, timeout and circuit-breaking policy out of application code. The cost is a "
    "second network hop and a second failure domain: a sidecar that has not finished "
    "loading its configuration will reject traffic that the workload is perfectly able to "
    "serve, so readiness must gate on the sidecar rather than the application.",

    "Erasure coding stores k data fragments plus m parity fragments, tolerating m losses "
    "at a storage overhead of (k+m)/k rather than the 3x of triple replication. The "
    "tradeoff is reconstruction cost: recovering one lost fragment requires reading k "
    "surviving fragments, so a rebuild consumes far more network bandwidth than copying a "
    "single replica would.",

    "Branch predictors trade table area for accuracy. A two-level adaptive predictor keys "
    "a pattern-history table on recent global branch outcomes, capturing correlations that "
    "a per-branch counter cannot see. Mispredictions cost the full pipeline depth, so on a "
    "deep out-of-order core a one percent accuracy improvement can outweigh a substantial "
    "increase in predictor storage.",

    "Consistent hashing maps both keys and nodes onto a ring so that adding a node "
    "displaces only the keys between it and its predecessor. Virtual nodes smooth the "
    "resulting distribution: without them, a small cluster develops hot ranges purely from "
    "the accident of where node hashes land on the ring.",

    "Copy-on-write snapshots let a filesystem present a stable view while writes continue. "
    "A block referenced by a snapshot is never overwritten in place; the writer allocates "
    "elsewhere and updates the pointer. The consequence is write amplification and "
    "fragmentation over time, which is why long-lived snapshot chains degrade sequential "
    "read throughput even when the live data has not moved.",

    "Backpressure has to propagate to the point where work is admitted, not merely to the "
    "point where it is queued. A component that accepts requests into an unbounded buffer "
    "converts an overload into unbounded latency, which is harder to diagnose than an "
    "outright rejection and defeats every downstream timeout budget.",

    "Vector clocks detect concurrent updates without a global ordering, but they grow with "
    "the number of writers and cannot by themselves decide which concurrent value should "
    "win. Systems that expose them push that decision to the application; systems that "
    "hide them usually resolve by last-writer-wins on a physical timestamp and accept the "
    "silent loss of one update.",

    "Read repair fixes divergence lazily, during a read that already had to contact several "
    "replicas. It costs nothing extra in the common case where replicas agree, but it also "
    "cannot bound how long a rarely-read key stays stale, which is why systems that rely on "
    "it alone pair it with a periodic anti-entropy sweep.",

    "Admission control belongs at the edge of a system, before a request has consumed any "
    "expensive resource. Rejecting a request after it has already acquired a database "
    "connection and a worker thread converts an overload into a resource leak, because the "
    "cost was paid and the work discarded.",

    "Column-oriented storage wins on analytic scans because a query touching three of forty "
    "columns reads only those three, and because values of one type compress far better "
    "adjacent to each other than interleaved with unrelated fields. It loses on point "
    "updates, where a single logical row change touches every column file.",

    "A leader lease bounds how long a partitioned-away leader can keep acting as one. The "
    "lease has to be shorter than the time it takes a new leader to be elected and start "
    "serving, otherwise two nodes believe they hold write authority simultaneously and the "
    "system has no ordering guarantee at all.",

    "Speculative execution hides tail latency by issuing duplicate work and taking the first "
    "answer. It is only sound when the work is idempotent and when the cluster has spare "
    "capacity: under saturation the duplicates compete with the originals and make the tail "
    "worse rather than better.",

    "Schema evolution in a columnar format is cheap for additive changes and expensive for "
    "type changes. Adding a nullable column is a metadata edit; narrowing an integer width "
    "requires rewriting every file that contains it, which on a large table is a migration "
    "with its own failure and rollback story.",
]

QUESTIONS = [
    "Given the code above, identify the concurrency bug and propose a fix that does not "
    "introduce a lock ordering hazard.",
    "Refactor the code above to make the error paths testable without a live database, and "
    "explain what you changed and why.",
    "Review the code above for correctness under retries. Which operations are not "
    "idempotent, and what would you change?",
    "Write unit tests for the code above, covering the boundary conditions you think are "
    "most likely to be wrong in production.",
    "Explain what the code above does, then describe how you would extend it to support "
    "multi-region deployment.",
    "The code above is a hot path under load. Where is the allocation pressure coming from, "
    "and how would you reduce it without hurting readability?",
    "Summarise the material above, then answer: which approach would you choose for a "
    "system with strict tail-latency requirements, and what would you give up?",
    "Based on the material above, draft the design review section covering failure modes "
    "and their mitigations.",
]

FILE_HEADERS = [
    "# ---- src/scheduler/windows.py ----",
    "# ---- src/net/retry.go ----",
    "# ---- internal/store/compact.go ----",
    "# ---- web/src/hooks/useDebounced.ts ----",
    "# ---- src/routing/shard.py ----",
    "# ---- migrations/0007_ledger.sql ----",
]



# ---------------------------------------------------------------------------- #
#                             PER-PROMPT MUTATION                              #
# ---------------------------------------------------------------------------- #

# Cycling a shared pool made prompts overlap heavily, which prefix caching turns into a cache result.

_IDENTS = [
    "merge_windows", "RetryBudget", "Compact", "useDebounced", "shard_for", "ledger_entry",
    "coalesce_batches", "Limiter", "drainQueue", "reconcile", "ranked", "Backoff",
    "capacity", "refill_per_second", "tolerance", "windows", "merged", "scored", "digest",
    "batch", "cutoff", "kept", "inFlight", "concurrency", "desired", "observed", "ops",
    "ceiling", "jitter", "attempt", "session_id", "event_name", "occurred_at", "step",
]


def _mutate(text: str, salt: str, rng: random.Random) -> str:
    """Deterministically rename identifiers and jitter numeric literals."""
    out = text
    for ident in _IDENTS:
        if ident in out:
            out = re.sub(rf"\b{re.escape(ident)}\b", f"{ident}_{salt}", out)
    # Jitter standalone integers so constants differ, skipping 0, 1 and anything bound to an identifier.
    def _jit(m):
        """Return the matched integer, nudged upward unless it is 0 or 1."""
        v = int(m.group(0))
        return str(v) if v < 2 else str(v + rng.randint(1, 9))
    out = re.sub(r"(?<![\w.])\d{1,4}(?![\w.])", _jit, out)
    return out




def _trim_to(text: str, target_chars: int, boundary: str = " ") -> str:
    """Trim to a character budget at a boundary.

    Applied AFTER mutation, not before. Mutation inflates text — salted identifiers and
    per-paragraph reference tags added about 20% — so a budget enforced beforehand
    overshoots. Trimming last makes the final length the controlled quantity.
    """
    if len(text) <= target_chars:
        return text
    cut = text.rfind(boundary, 0, target_chars)
    return text[: cut if cut > 0 else target_chars].rstrip()


def _mutate_prose(text: str, salt: str, idx: int, rng: random.Random) -> str:
    """Vary prose per prompt. Prose has no identifiers to rename, so distinctness comes from
    per-paragraph reference markers and jittered figures. Without this, a long prose prompt
    cycles the whole unit pool and every prompt ends up with the same vocabulary — measured
    at 0.96 pairwise word overlap on rag_document before this was added."""
    paras = [seg for seg in text.split("\n\n") if seg.strip()]
    out = []
    for k, seg in enumerate(paras):
        tag = f"[{salt}:{idx:03d}.{k:02d}]"
        seg = re.sub(r"(?<![\w.])\d{1,4}(?![\w.])",
                     lambda m: str(int(m.group(0)) + rng.randint(1, 9))
                     if int(m.group(0)) >= 2 else m.group(0), seg)
        out.append(f"{tag} {seg}")
    return "\n\n".join(out)


def _lead(kind: str, idx: int, salt: str) -> str:
    """A unique opening line per prompt. Differing at position zero is what defeats
    prefix caching; everything after it only reduces total overlap."""
    if kind.startswith("code"):
        return f"# review-id {salt} / file group {idx:03d} / revision {1000 + idx * 7}"
    if kind == "rag":
        return f"[retrieval set {salt} | query {idx:03d} | top-k reranked]"
    return f"(context ref {salt}-{idx:03d})"


def _fill_to(rng: random.Random, target_chars: int, units: list[str], joiner: str) -> str:
    """Assemble units to about target_chars, cycling deterministically, then trim.

    Trimming matters. Adding whole units until the budget is exceeded overshoots by up to a
    full unit, which on the smaller profiles is a large fraction of the total — that is what
    put chat_typical 20% under its stated size. Trimming at a word boundary lands within a
    few characters, leaving only tokenizer variance.
    """
    parts: list[str] = []
    total = 0
    idx = 0
    order = list(range(len(units)))
    rng.shuffle(order)
    while total < target_chars:
        unit = units[order[idx % len(order)]]
        parts.append(unit)
        total += len(unit) + len(joiner)
        idx += 1
    text = joiner.join(parts)
    if len(text) > target_chars:
        cut = text.rfind(" ", 0, target_chars)
        text = text[: cut if cut > 0 else target_chars].rstrip()
    return text


def make_chat(rng: random.Random, target_tokens: int, n: int,
              cpt: float | None = None) -> list[str]:
    """Build n conversational prompts of roughly target_tokens, each mutated to be distinct."""
    out = []
    for i in range(n):
        salt = f"{rng.randrange(16**6):06x}"
        # Per-prompt rng, without which neighbouring prompts open with the same paragraph.
        prng = random.Random(f"chat::{i}::{salt}")
        units = list(PROSE_UNITS)
        prng.shuffle(units)
        lead = _lead("chat", i, salt)
        q = QUESTIONS[(i + 6) % len(QUESTIONS)]
        # Reserve exactly what the wrapper costs, since a fixed guess is ruinous on a short prompt.
        budget = (int(target_tokens * (cpt or CHARS_PER_TOKEN["prose"]))
                  - len(lead) - len(q) - 4)
        body = _fill_to(prng, int(budget * 1.6), units, "\n\n")
        body = _trim_to(_mutate_prose(body, salt, i, prng), max(budget, 40))
        out.append(f"{lead}\n\n{body}\n\n{q}")
    return out


def make_code(rng: random.Random, target_tokens: int, n: int, multifile: bool) -> list[str]:
    """Build n coding prompts, optionally spanning several files for repo-context shapes."""
    out = []
    for i in range(n):
        salt = f"{rng.randrange(16**6):06x}"
        prng = random.Random(f"code::{i}::{salt}")
        budget = int(target_tokens * CHARS_PER_TOKEN["code"]) - 210
        chunks: list[str] = []
        total = 0
        j = 0
        order = list(range(len(CODE_UNITS)))
        prng.shuffle(order)
        while total < int(budget * 1.4):
            k = order[j % len(order)]
            header = FILE_HEADERS[k % len(FILE_HEADERS)] if multifile else ""
            unit = _mutate(CODE_UNITS[k], f"{salt}{j:02d}", prng)
            piece = (header + "\n" + unit) if header else unit
            chunks.append(piece)
            total += len(piece)
            j += 1
        body = _trim_to("\n\n".join(chunks), budget, boundary="\n")
        q = QUESTIONS[i % len(QUESTIONS)]
        out.append(f"{_lead('code', i, salt)}\n{body}\n\n{q}")
    return out


def make_rag(rng: random.Random, target_tokens: int, n: int) -> list[str]:
    """Build n retrieval-style prompts: passages of context followed by a question."""
    out = []
    for i in range(n):
        salt = f"{rng.randrange(16**6):06x}"
        prng = random.Random(f"rag::{i}::{salt}")
        units = list(PROSE_UNITS)
        prng.shuffle(units)
        lead = _lead("rag", i, salt)
        q = QUESTIONS[(i + 3) % len(QUESTIONS)]
        instr = "Use only the passages below to answer."
        budget = (int(target_tokens * CHARS_PER_TOKEN["prose"])
                  - len(lead) - len(q) - len(instr) - 60)
        body = _fill_to(prng, int(budget * 1.6), units, "\n\n")
        body = _trim_to(_mutate_prose(body, salt, i, prng), max(budget, 60))
        passages = "\n\n".join(
            f"[passage {k + 1}] {seg.strip()}" for k, seg in enumerate(body.split("\n\n"))
        )
        out.append(f"{lead}\n{instr}\n\n{passages}\n\nQuestion: {q}")
    return out


LEGACY_FIXED = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to reverse a linked list.",
    "What are the main differences between Kubernetes and Docker Swarm?",
    "Summarize the plot of The Matrix.",
    "How does a transformer neural network work?",
]

# Pool size falls as prompts grow, keeping each profile shippable inside a 1 MiB ConfigMap.
SPECS = [
    dict(name="chat_short", input_tokens=128, output_tokens=512, n=32, kind="chat",
         use_case="Single-turn question with no conversation history.",
         rationale="Floor case for interactive chat. Prefill is negligible, so this "
                   "isolates decode cost with a plausible prompt rather than a 41-token one."),
    dict(name="chat_typical", input_tokens=512, output_tokens=1024, n=32, kind="chat",
         # Mutation tags are token-dense and shift with length, so this profile needs its own ratio.
         chars_per_token=4.84,
         use_case="Conversational turn carrying a short history and a system prompt.",
         rationale="Representative of a general assistant. Output budget matches the "
                   "published reasoning protocol, so the only difference from the legacy "
                   "headline shape is the input."),
    dict(name="coding_completion", input_tokens=2048, output_tokens=256, n=24, kind="code",
         use_case="Inline code completion with the surrounding file supplied as context.",
         rationale="Short output by design: completions are a few lines. Latency-sensitive, "
                   "so pair with the latency or server_qps protocols, not throughput."),
    dict(name="rag_document", input_tokens=4096, output_tokens=512, n=16, kind="rag",
         use_case="Retrieved passages plus a question.",
         rationale="Moderate context, short grounded answer. Prefix caching usually does "
                   "NOT help here, because retrieved passages differ per query — unlike a "
                   "coding assistant's stable file context."),
    dict(name="coding_assistant", input_tokens=8192, output_tokens=1024, n=12,
         kind="code_multifile",
         use_case="Coding chat: several open files plus conversation history.",
         rationale="The profile to quote for a coding-assistant deployment. Prefill "
                   "dominates time-to-first-token at this size, and KV per request rises to "
                   "~9.2k tokens, which materially lowers achievable concurrency."),
    dict(name="coding_repo_context", input_tokens=16384, output_tokens=1024, n=8,
         kind="code_multifile",
         use_case="Large repository or multi-file context, agentic tool output.",
         rationale="Upper end of what a coding assistant sends. ~17.4k tokens of KV per "
                   "request, so concurrency is KV-bound well before the compute limit."),
    dict(name="legacy_corpus", input_tokens=9, output_tokens=150, n=5, kind="legacy",
         use_case="HISTORICAL ONLY. Cross-instance continuity with the g7e/g6e/p5en corpus. "
                  "Five FIXED prompts that REPEAT across requests.",
         rationale="Do not use for new claims, for two reasons. A 150-token output budget "
                   "truncates a reasoning model mid-thought: the budget is spent on chain of "
                   "thought and the request returns finish_reason=length with zero visible "
                   "content, so it measures truncation rather than serving. And the five "
                   "prompts repeat, so with prefix caching enabled server-side it becomes a "
                   "cache-hit measurement."),
    dict(name="legacy_reasoning", input_tokens=41, output_tokens=1024, n=32, kind="legacy41",
         use_case="HISTORICAL ONLY. The shape behind the headline throughput figures in the "
                  "published reports.",
         rationale="A 41-token input represents no real workload. Retained so the shape of "
                   "the published figures stays explicit instead of being reconstructed from "
                   "source. Treat those numbers as node capacity under a synthetic best "
                   "case: on one measured configuration, raising input to 13k tokens moved "
                   "time-to-first-token from 71 ms to 2.58 s on the same server."),
]


def build(spec: dict) -> dict:
    """Build one profile document from its spec, seeded by name so it reproduces exactly."""
    rng = random.Random(f"prompt-profile::{spec['name']}")
    kind = spec["kind"]
    if kind == "chat":
        prompts = make_chat(rng, spec["input_tokens"], spec["n"], spec.get("chars_per_token"))
    elif kind == "code":
        prompts = make_code(rng, spec["input_tokens"], spec["n"], multifile=False)
    elif kind == "code_multifile":
        prompts = make_code(rng, spec["input_tokens"], spec["n"], multifile=True)
    elif kind == "rag":
        prompts = make_rag(rng, spec["input_tokens"], spec["n"])
    elif kind == "legacy":
        prompts = list(LEGACY_FIXED)
    elif kind == "legacy41":
        # Reproduces the historical unique shape from fixed inspectable strings rather than a live uuid4.
        stems = [
            "Explain in detail how {t} works, covering the underlying mechanics.",
            "Compare and contrast the main approaches to {t} and justify a recommendation.",
            "Describe the failure modes of {t} and how to mitigate each one.",
            "Walk through the design of a production system for {t}, step by step.",
        ]
        topics = ["write-ahead logging", "service mesh sidecars", "erasure coding",
                  "branch prediction", "consistent hashing", "copy-on-write snapshots",
                  "backpressure propagation", "vector clocks"]
        prompts = [
            f"[req:{i:04d}{'ab3f9c2d' * 3}] " + stems[i % len(stems)].format(t=topics[i % len(topics)])
            for i in range(spec["n"])
        ]
    else:
        raise ValueError(kind)

    # A small declared fraction repeats an earlier prompt, isolating cache effect without inflating it.
    dup_idx: list[int] = []
    if spec["kind"] not in ("legacy", "legacy41") and len(prompts) >= 8:
        n_dup = max(1, round(len(prompts) * DUPLICATE_FRACTION))
        for d in range(n_dup):
            src = d
            dst = len(prompts) - 1 - d
            if dst > src:
                prompts[dst] = prompts[src]
                dup_idx.append(dst)

    return {
        "schema": SCHEMA,
        "name": spec["name"],
        "legacy": spec["name"].startswith("legacy"),
        "use_case": spec["use_case"],
        "rationale": spec["rationale"],
        "target_input_tokens": spec["input_tokens"],
        "output_tokens": spec["output_tokens"],
        "fixed_prompt_set": spec["kind"] == "legacy",
        "prompt_count": len(prompts),
        "distinct_prompt_count": len(set(prompts)),
        # Indices that are exact repeats of an earlier prompt, and of which one.
        "intentional_duplicate_indices": sorted(dup_idx),
        "duplicate_note": (
            "These indices repeat an earlier prompt verbatim, on purpose. With prefix "
            "caching OFF they are indistinguishable from any other prompt. With it ON they "
            "are the only prompts able to hit the cache, which is how to measure the caching "
            "effect without contaminating the uncached figure."
        ),
        # Model-agnostic literal strings, since byte-identical input is what makes a comparison meaningful.
        "prompts": prompts,
        # No model-keyed data here, so per-model counts live under docs/prompt_tokens/ instead.
    }


def main() -> int:
    """Write every profile, or verify the committed files match a fresh build."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify committed files match a fresh build; do not write")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    mismatched: list[str] = []
    print(f"{'profile':<24} {'prompts':>8} {'target tok':>11} {'out tok':>8} {'KiB':>7}")
    print("-" * 64)

    for spec in SPECS:
        doc = build(spec)
        path = OUT / f"{spec['name']}.json"

        if args.check:
            if not path.exists():
                mismatched.append(f"  {spec['name']}: missing")
                continue
            existing = json.loads(path.read_text())
            if existing.get("prompts") != doc["prompts"]:
                mismatched.append(f"  {spec['name']}: prompts differ from a fresh build")
        else:
            path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

        kib = (len(json.dumps(doc)) / 1024) if not path.exists() else path.stat().st_size / 1024
        print(f"{spec['name']:<24} {doc['prompt_count']:>8} {spec['input_tokens']:>11} "
              f"{spec['output_tokens']:>8} {kib:>7.0f}")

    if args.check:
        if mismatched:
            print("\nNOT REPRODUCIBLE:")
            print("\n".join(mismatched))
            return 1
        print("\nall profiles reproducible from this generator")
    else:
        print(f"\n{len(SPECS)} profiles written to {OUT.relative_to(config.PROJECT_ROOT)}")
        print("Next: app/bench/src/verify_profiles.py --model <repo-or-path> to record "
              "measured token counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
