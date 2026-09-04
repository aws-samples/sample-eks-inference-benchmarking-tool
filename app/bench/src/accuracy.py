#!/usr/bin/env python3
"""Correctness gate: proves a server is producing right answers before its speed is measured.

Every result this harness writes measures throughput and latency, and until this gate existed
nothing verified the server under test was configured the way the result claimed. That is not
a theoretical gap: a truncated `vllm serve` argv once silently dropped --kv-cache-dtype fp8
and both parsers, and the server came up healthy and reported clean throughput at the wrong
precision. It has two tiers — `fingerprint` compares a small pinned prompt set against a
stored reference in about 30 seconds and is cheap enough to run before every benchmark, while
`score` measures GSM8K exact match against a reference with a tolerance. Both use their own
request path at temperature 0 with a fixed seed, because the benchmark protocols deliberately
run at 0.7 and sampling at 0.7 is not reproducible. Exit codes: 0 pass, 1 fail, 2 environment
or usage error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from pathlib import Path

import aiohttp

import config
from stats import (distinct_ngram_ratio, git_revision, sha256_file, sha256_text,
                  strip_trailing_whitespace, utc_stamp)

SCHEMA = config.ACCURACY_SCHEMA

# ---------------------------------------------------------------------------- #
#                             GENERATION SETTINGS                              #
# ---------------------------------------------------------------------------- #

GEN = config.ACCURACY_GEN
FINGERPRINT_N = config.FINGERPRINT_N
FINGERPRINT_MAX_TOKENS = config.FINGERPRINT_MAX_TOKENS
SCORE_MAX_TOKENS = config.SCORE_MAX_TOKENS

# ---------------------------------------------------------------------------- #
#                               CHECK TOLERANCES                               #
# ---------------------------------------------------------------------------- #

TOKEN_DRIFT_TOLERANCE = config.TOKEN_DRIFT_TOLERANCE
NONREPRO_MEAN_TOKEN_TOLERANCE = config.NONREPRO_MEAN_TOKEN_TOLERANCE
NONREPRO_TRUNCATION_TOLERANCE = config.NONREPRO_TRUNCATION_TOLERANCE
MIN_DISTINCT_NGRAM_RATIO = config.MIN_DISTINCT_NGRAM_RATIO
NGRAM_N = config.NGRAM_N
MAX_TRUNCATED_FRACTION = config.MAX_TRUNCATED_FRACTION

# ---------------------------------------------------------------------------- #
#                               ANSWER PATTERNS                                #
# ---------------------------------------------------------------------------- #

NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")
MONEY = re.compile(r"-?\$\s?\d[\d,]*\.?\d*")
# Where a chat model restates its result. Anchoring here beats scanning the whole reply, which
# is full of intermediate arithmetic.
ANCHOR = re.compile(r"(?i)final answer|answer\s*[:*]|answer\s+is")
# The GSM8K gold marker is '#### 42' alone on a line. It must NOT match a markdown level-4
# heading like '#### **Remaining time:**', which these models emit routinely — 3 of 200 Qwen3
# replies did, and a naive '####' split silently returned a number from the heading.
GSM_MARK = re.compile(r"####\s*(-?\$?[\d][\d,]*\.?\d*)\s*$", re.M)
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)


# ---------------------------------------------------------------------------- #
#                                   DATASET                                    #
# ---------------------------------------------------------------------------- #

def load_problems() -> list[dict]:
    """Load the pinned GSM8K subset, exiting with an environment error if it is missing."""
    if not config.EVAL_DATA.exists():
        sys.exit(f"ENV ERROR: {relative(config.EVAL_DATA)} missing. "
                 f"Run: python3 app/bench/src/fetch_eval_data.py")
    lines = config.EVAL_DATA.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(line) for line in lines]


def relative(path: Path) -> str:
    """Render a path relative to the repository root, for readable messages."""
    return str(path.relative_to(config.PROJECT_ROOT))


# ---------------------------------------------------------------------------- #
#                              ANSWER EXTRACTION                               #
# ---------------------------------------------------------------------------- #

def normalise_latex(text: str) -> str:
    """Normalise LaTeX numeric spellings so '180{,}000' parses as one number."""
    return (text.replace("{,}", ",").replace("\\,", "").replace("{.}", ".")
                .replace("\\!", "").replace("\\ ", " "))


def strip_number(text: str) -> str:
    """Reduce a matched numeric span to bare digits for comparison."""
    return (text.replace(",", "").replace("$", "").replace("{", "").replace("}", "")
                .replace(" ", "").rstrip("."))


def boxed_spans(text: str) -> list[tuple[int, str]]:
    r"""Find every \boxed{...} with its offset, matching braces by hand.

    A regex of [^}]* truncates \boxed{45 \text{ miles}} at the inner brace, which is why this
    is counted rather than pattern-matched.
    """
    spans, cursor = [], 0
    while True:
        start = text.find("\\boxed{", cursor)
        if start < 0:
            return spans
        end, depth = start + 7, 1
        while end < len(text) and depth:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        spans.append((start, text[start + 7:end - 1]))
        cursor = end


def best_number(span: str) -> str | None:
    """Pick the answer number from a short span: a currency amount if present, else the first.

    Currency wins because a money answer is written '$220', and the sentence around it often
    restates the question first ('balance after 12 months is $220').
    """
    span = normalise_latex(span)
    money = MONEY.search(span)
    if money:
        return strip_number(money.group(0))
    number = NUMBER.search(span)
    return strip_number(number.group(0)) if number else None


def extract_answer(text: str) -> str | None:
    r"""Pull the model's final numeric answer, trying the most explicit conventions first.

    Order: the strict '#### n' marker, the first \boxed after an answer anchor, the last
    \boxed anywhere, the first bold span after an anchor, then a last resort. Tuned against
    200 real Qwen3-235B replies where "last number in the text" scored 179/200 against gold
    and this scores 194/200; the 15 recovered cases were all extraction artefacts, so the
    naive version would have understated accuracy by 7.5 points and baked that into a reference.
    """
    marks = GSM_MARK.findall(text)
    if marks:
        return strip_number(normalise_latex(marks[-1]))

    boxes = boxed_spans(text)
    anchors = [m.end() for m in ANCHOR.finditer(text)]

    # First boxed after the last anchor: when a model offers alternate units
    # ('\boxed{180,000} meters or \boxed{180} kilometers') the first is the unit asked for.
    if anchors and boxes:
        after = [body for pos, body in boxes if pos >= anchors[-1]]
        if after and best_number(after[0]):
            return best_number(after[0])
    if boxes and best_number(boxes[-1][1]):
        return best_number(boxes[-1][1])
    if anchors:
        tail = text[anchors[-1]:]
        bold = BOLD.search(tail)
        if bold and best_number(bold.group(1)):
            return best_number(bold.group(1))
        if best_number(tail):
            return best_number(tail)

    bolds = list(BOLD.finditer(text))
    if bolds and best_number(bolds[-1].group(1)):
        return best_number(bolds[-1].group(1))
    numbers = NUMBER.findall(normalise_latex(text))
    return strip_number(numbers[-1]) if numbers else None


def answers_match(got: str | None, gold: str) -> bool:
    """Compare an extracted answer against gold numerically, falling back to text equality."""
    if got is None:
        return False
    try:
        return abs(float(got) - float(gold)) < 1e-6
    except ValueError:
        return got.strip() == gold.strip()


# ---------------------------------------------------------------------------- #
#                                 REQUEST PATH                                 #
# ---------------------------------------------------------------------------- #

def build_payload(model: str, question: str, max_tokens: int, thinking: str) -> dict:
    """Build one deterministic chat-completion body, optionally suppressing the think trace.

    Hybrid-thinking models (Qwen3, Kimi K3) emit a <think> trace by default, and on a maths
    word problem that trace routinely exceeds the token budget, so the response never reaches
    its final answer and the score measures TRUNCATION rather than correctness. 'off'
    suppresses it via the chat template. Recorded in the reference, because a score taken with
    thinking on is not comparable to one taken with it off.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        **GEN,
    }
    if thinking != "auto":
        payload["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
    return payload


async def one_request(session, endpoint: str, model: str, question: str,
                      max_tokens: int, thinking: str) -> dict:
    """Issue one deterministic completion and return the fields the checks compare.

    Uses /chat/completions because every model in this repo is instruction-tuned and needs its
    chat template applied server-side, matching BENCH_ENDPOINT=chat in the benchmark protocols.
    A transport failure is returned as data rather than raised, so one dropped connection is
    reported as a failed item instead of a stack trace.
    """
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = build_payload(model, question, max_tokens, thinking)
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                return {"error": f"HTTP {response.status}: {(await response.text())[:400]}"}
            data = await response.json()
    except aiohttp.ClientError as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "transport_error": True}
    return summarise_response(data)


def summarise_response(data: dict) -> dict:
    """Reduce a completion to the fields the gate compares.

    reasoning_content is captured because its presence is itself a config fingerprint: vLLM
    only separates chain-of-thought when a reasoning parser is active, and the truncated-argv
    incident dropped the parser, which moves that text into content instead.
    """
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (data.get("usage") or {}).get("completion_tokens", 0),
    }


async def run_set(endpoint: str, model: str, problems: list[dict], max_tokens: int,
                  concurrency: int, label: str, thinking: str) -> list[dict]:
    """Run the whole problem set at bounded concurrency, preserving input order."""
    limiter = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(problems)
    progress = {"done": 0}

    async def worker(index: int, problem: dict):
        """Answer one problem, store it at its original index, and report progress."""
        async with limiter:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
                result = await one_request(s, endpoint, model, problem["question"],
                                           max_tokens, thinking)
        result["id"], result["gold"] = problem["id"], problem["gold"]
        results[index] = result
        progress["done"] += 1
        if progress["done"] % 20 == 0 or progress["done"] == len(problems):
            print(f"    {label}: {progress['done']}/{len(problems)}", flush=True)

    await asyncio.gather(*[worker(i, p) for i, p in enumerate(problems)])
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------- #
#                            REFERENCE-FREE CHECKS                             #
# ---------------------------------------------------------------------------- #

def structural_failures(items: list[dict]) -> list[str]:
    """Check what must hold for any healthy deployment, with no reference needed."""
    failures = []
    for item in items:
        if item.get("error"):
            failures.append(f"{item['id']}: request failed: {item['error']}")
            continue
        text = item["content"] or ""
        if not text.strip():
            failures.append(f"{item['id']}: empty content "
                            f"(finish_reason={item.get('finish_reason')})")
            continue
        if item["completion_tokens"] <= 0:
            failures.append(f"{item['id']}: usage.completion_tokens={item['completion_tokens']}")
        ratio = distinct_ngram_ratio(text, NGRAM_N)
        if ratio < MIN_DISTINCT_NGRAM_RATIO:
            failures.append(f"{item['id']}: output collapsed into repetition "
                            f"(distinct-{NGRAM_N}-gram ratio {ratio:.2f} "
                            f"< {MIN_DISTINCT_NGRAM_RATIO})")
    return failures


def truncation_failure(items: list[dict], max_tokens: int, thinking: str) -> list[str]:
    """Refuse to report a score that is really a truncation rate.

    A hybrid-thinking model left in its default mode spends the whole budget on <think> and
    never emits a final answer. extract_answer then picks the last number out of a partial
    reasoning trace, which is usually wrong, so the result looks like catastrophic accuracy
    loss when nothing is actually broken.
    """
    if not items:
        return []
    truncated = sum(1 for item in items if item.get("finish_reason") == "length")
    fraction = truncated / len(items)
    if fraction <= MAX_TRUNCATED_FRACTION:
        return []
    hint = ("Pass --thinking off to suppress the <think> trace." if thinking != "off"
            else f"Raise the token budget above {max_tokens}.")
    return [f"{truncated}/{len(items)} responses ({fraction:.0%}) hit the {max_tokens}-token "
            f"ceiling without finishing, above the {MAX_TRUNCATED_FRACTION:.0%} limit. This "
            f"score would measure the token budget, not accuracy. {hint}"]


# ---------------------------------------------------------------------------- #
#                               FINGERPRINT TIER                               #
# ---------------------------------------------------------------------------- #

def build_fingerprint(items: list[dict], reproducible: bool) -> dict:
    """Capture the per-item hashes and metadata that later runs are compared against."""
    return {
        "reproducible": reproducible,
        "items": [
            {
                "id": item["id"],
                "sha256": sha256_text(item["content"]),
                # Preview is for human debugging on mismatch; the hash is the actual check.
                "preview": strip_trailing_whitespace(item["content"])[:80],
                "completion_tokens": item["completion_tokens"],
                "has_reasoning_content": item.get("reasoning_content") is not None,
                "finish_reason": item.get("finish_reason"),
            }
            for item in items
        ],
    }


def compare_settings(ref: dict, max_tokens: int, thinking: str) -> list[str]:
    """Check the generation settings match, since different settings are not comparable."""
    failures = []
    reference_max = ref["fingerprint"].get("max_tokens")
    if reference_max is not None and reference_max != max_tokens:
        failures.append(f"max_tokens mismatch: reference captured at {reference_max}, "
                        f"this run used {max_tokens}")
    reference_gen = ref.get("generation", {})
    for key, value in {**GEN, "thinking": thinking}.items():
        if key in reference_gen and reference_gen[key] != value:
            failures.append(f"generation.{key} mismatch: reference {reference_gen[key]}, "
                            f"now {value}")
    return failures


def compare_items(ref_items: list[dict], by_id: dict[str, dict],
                  reproducible: bool) -> list[str]:
    """Compare each reference item against this run's, honouring text reproducibility."""
    failures = []
    for reference in ref_items:
        got = by_id.get(reference["id"])
        if got is None:
            failures.append(f"{reference['id']}: missing from this run")
            continue

        # Parser configuration check, run regardless of text reproducibility: losing
        # --reasoning-parser is a config defect, not sampling noise.
        got_has_reasoning = got.get("reasoning_content") is not None
        if got_has_reasoning != reference["has_reasoning_content"]:
            failures.append(
                f"{reference['id']}: reasoning_content presence changed "
                f"(reference {reference['has_reasoning_content']}, got {got_has_reasoning}). "
                f"A missing --reasoning-parser moves chain-of-thought into content.")
        if reproducible:
            failures += compare_text(reference, got)
    return failures


def compare_text(reference: dict, got: dict) -> list[str]:
    """Compare one item's exact text and token count against a reproducible reference."""
    failures = []
    expected, actual = reference["completion_tokens"], got["completion_tokens"]
    if expected > 0 and abs(actual - expected) / expected > TOKEN_DRIFT_TOLERANCE:
        failures.append(f"{reference['id']}: completion_tokens {actual} vs reference "
                        f"{expected} (>{TOKEN_DRIFT_TOLERANCE:.0%} drift)")
    if sha256_text(got["content"]) != reference["sha256"]:
        failures.append(f"{reference['id']}: output text differs from reference\n"
                        f"      reference: {reference['preview']!r}\n"
                        f"      got      : "
                        f"{strip_trailing_whitespace(got['content'])[:80]!r}")
    return failures


def compare_aggregates(ref_items: list[dict], items: list[dict],
                       reproducible: bool) -> list[str]:
    """Check the whole-set signals that stay meaningful even when text is not reproducible."""
    failures = []
    # How tightly a truncation increase can be judged depends on reproducibility: see NONREPRO_TRUNCATION_TOLERANCE.
    reference_truncated = sum(1 for r in ref_items if r.get("finish_reason") == "length")
    got_truncated = sum(1 for i in items if i.get("finish_reason") == "length")

    if reproducible:
        # Bit-identical text makes the count deterministic, so any real increase is a behaviour change.
        if got_truncated > reference_truncated + max(1, len(items) // 10):
            failures.append(f"truncation increased: {got_truncated}/{len(items)} responses hit "
                            f"the token ceiling vs {reference_truncated}/{len(ref_items)} in the "
                            f"reference")
        return failures

    # Non-reproducible: compare the RATE with a percentage-point allowance, floored like the reproducible case.
    reference_rate = reference_truncated / len(ref_items) if ref_items else 0.0
    got_rate = got_truncated / len(items) if items else 0.0
    allowance_rate = max(NONREPRO_TRUNCATION_TOLERANCE, max(1, len(items) // 10) / len(items)) if items else NONREPRO_TRUNCATION_TOLERANCE
    if got_rate > reference_rate + allowance_rate:
        failures.append(f"truncation rate rose to {got_rate:.0%} ({got_truncated}/{len(items)}) "
                        f"from {reference_rate:.0%} ({reference_truncated}/{len(ref_items)}) in "
                        f"the reference, beyond the "
                        f"{allowance_rate:.0%}-point allowance for a "
                        f"non-reproducible server")

    reference_mean = statistics.mean([r["completion_tokens"] for r in ref_items]) if ref_items else 0
    got_mean = statistics.mean([i["completion_tokens"] for i in items]) if items else 0
    if reference_mean > 0:
        drift = abs(got_mean - reference_mean) / reference_mean
        if drift > NONREPRO_MEAN_TOKEN_TOLERANCE:
            failures.append(f"mean completion_tokens {got_mean:.1f} vs reference "
                            f"{reference_mean:.1f} ({drift:.0%} drift, limit "
                            f"{NONREPRO_MEAN_TOKEN_TOLERANCE:.0%})")
    return failures


def compare_fingerprint(ref: dict, items: list[dict], max_tokens: int,
                        thinking: str) -> list[str]:
    """Compare a run against a stored fingerprint, returning every problem found."""
    fingerprint = ref["fingerprint"]
    reproducible = fingerprint.get("reproducible", True)
    ref_items = fingerprint["items"]
    failures = compare_settings(ref, max_tokens, thinking)
    if len(ref_items) != len(items):
        failures.append(f"item count differs: reference has {len(ref_items)}, got {len(items)}")
    failures += compare_items(ref_items, {i["id"]: i for i in items}, reproducible)
    failures += compare_aggregates(ref_items, items, reproducible)
    return failures


# ---------------------------------------------------------------------------- #
#                                  SCORE TIER                                  #
# ---------------------------------------------------------------------------- #

def score_items(items: list[dict]) -> tuple[float, int, int, list[str]]:
    """Score exact-match accuracy, returning (fraction, correct, total, sample misses)."""
    correct, misses = 0, []
    for item in items:
        if item.get("error"):
            misses.append(f"{item['id']}: {item['error']}")
            continue
        got = extract_answer(item["content"] or "")
        if answers_match(got, item["gold"]):
            correct += 1
        elif len(misses) < 10:
            truncated = " [TRUNCATED]" if item.get("finish_reason") == "length" else ""
            misses.append(f"{item['id']}: got {got!r}, gold {item['gold']!r}{truncated}")
    total = len(items)
    return (correct / total if total else 0.0), correct, total, misses


def compare_score(ref: dict, accuracy: float, count: int, args) -> list[str]:
    """Compare a score against its reference, checking the comparison is even valid."""
    failures = []
    reference_value = ref["score"]["value"]
    floor = reference_value * args.tolerance
    read_config = args.reference_config or args.config
    cross_config = read_config != args.config

    print(f"  reference        : {reference_value:.4f} "
          f"(config {ref['config']}, captured {ref['captured_utc']})")
    if cross_config:
        print(f"  comparison       : serving {args.config}, scored against "
              f"{read_config} reference")
    print(f"  tolerance floor  : {floor:.4f} ({args.tolerance:.0%} of reference)")

    # Problem count and thinking mode must match or the two scores are not comparable: a
    # 60-problem score has much wider error bars than a 200-problem one, and a reasoning trace
    # changes both answers and token counts.
    reference_count = ref["score"].get("n")
    if reference_count is not None and reference_count != count:
        failures.append(f"problem count differs: reference scored {reference_count}, this run "
                        f"{count}. Use --n {reference_count} to compare like with like.")
    reference_thinking = ref.get("generation", {}).get("thinking")
    if reference_thinking is not None and reference_thinking != args.thinking:
        failures.append(f"thinking mode differs: reference captured with --thinking "
                        f"{reference_thinking}, this run used {args.thinking}. Reasoning traces "
                        f"change both answers and token counts, so the two scores are not "
                        f"comparable.")
    if accuracy < floor:
        degraded = (f" — quantization to {args.config} appears to have degraded accuracy"
                    if cross_config else "")
        failures.append(f"gsm8k exact match {accuracy:.4f} is below {floor:.4f} "
                        f"({args.tolerance:.0%} of {read_config} reference "
                        f"{reference_value:.4f}){degraded}")
    return failures


# ---------------------------------------------------------------------------- #
#                               REFERENCE FILES                                #
# ---------------------------------------------------------------------------- #

def reference_path(model_tag: str, server_config: str) -> Path:
    """Locate the reference file for one model and server config."""
    return config.ACCURACY_REF_DIR / f"{model_tag}-{server_config}.json"


def load_reference(path: Path, mode: str) -> dict:
    """Load and validate a stored reference, failing closed if it is missing or stale."""
    if not path.exists():
        sys.exit(
            f"FAIL: no accuracy reference at {relative(path)}\n\n"
            f"This gate fails closed. A missing reference is not a pass, because silently\n"
            f"proceeding is the exact failure this gate exists to prevent.\n\n"
            f"Capture one against a server you have verified independently:\n"
            f"  python3 app/bench/src/accuracy.py --mode {mode} --write-reference \\\n"
            f"      --model-tag <tag> --config <config> --endpoint <url> --model-id <id>\n\n"
            f"References are gitignored and stay local to your clone, by design: one is the\n"
            f"output of YOUR server on YOUR hardware, so inheriting someone else's would gate\n"
            f"your runs against a configuration you never inspected. See docs/accuracy/README.md.\n\n"
            f"To run a benchmark without this gate (and without its protection):\n"
            f"  app/bench/src/suite.py --skip-accuracy")
    ref = json.loads(path.read_text(encoding="utf-8"))
    if ref.get("schema") != SCHEMA:
        sys.exit(f"FAIL: {relative(path)} schema {ref.get('schema')} != expected {SCHEMA}")
    if ref.get("dataset", {}).get("sha256") != sha256_file(config.EVAL_DATA):
        sys.exit(
            f"FAIL: dataset digest mismatch for {relative(path)}\n"
            f"  reference captured against {ref.get('dataset', {}).get('sha256')}\n"
            f"  app/bench/data/gsm8k_subset.jsonl is  {sha256_file(config.EVAL_DATA)}\n"
            f"The problem set changed, so the reference is not comparable. Either restore\n"
            f"the pinned subset (app/bench/src/fetch_eval_data.py) or recapture references.")
    return ref


def reference_skeleton(args, max_tokens: int) -> dict:
    """Build the provenance block every written reference shares."""
    return {
        "schema": SCHEMA,
        "model_tag": args.model_tag,
        "config": args.config,
        "served_model_id": args.model_id,
        "captured_utc": utc_stamp(),
        "harness_rev": git_revision(config.PROJECT_ROOT),
        # Shared deterministic settings only: max_tokens differs per tier, so it lives inside
        # each tier's own block rather than here.
        "generation": {**GEN, "thinking": args.thinking},
        "dataset": {"file": "app/bench/data/gsm8k_subset.jsonl", "sha256": sha256_file(config.EVAL_DATA)},
    }


def write_reference(path: Path, ref: dict, preserve_key: str) -> None:
    """Write a reference, carrying over the other tier's block if one already exists."""
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if preserve_key in previous:
            ref[preserve_key] = previous[preserve_key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Deliberately does not say "commit this file": docs/accuracy/*.json is gitignored, see docs/accuracy/README.md.
    print(f"\nwrote {relative(path)} (gitignored, local to this clone)")


# ---------------------------------------------------------------------------- #
#                                 TIER DRIVERS                                 #
# ---------------------------------------------------------------------------- #

def measure_reproducibility(args, problems: list[dict], max_tokens: int,
                            first_pass: list[dict]) -> bool:
    """Run the set a second time to measure, rather than assume, text reproducibility.

    Large MoE models with expert routing are not guaranteed bit-reproducible even at
    temperature 0. When two identical passes disagree the reference records
    `reproducible: false` and text matching is disabled for that config, so later runs are
    checked structurally instead. Recording it beats a silent behavior change.
    """
    print("  second pass (reproducibility probe)...")
    second_pass = asyncio.run(run_set(args.endpoint, args.model_id, problems, max_tokens,
                                      args.concurrency, "pass2", args.thinking))
    first = {item["id"]: sha256_text(item["content"]) for item in first_pass}
    second = {item["id"]: sha256_text(item["content"]) for item in second_pass}
    differing = sorted(key for key in first if first.get(key) != second.get(key))
    if not differing:
        print(f"  reproducible: YES ({len(first_pass)}/{len(first_pass)} identical across passes)")
        return True
    print(f"  reproducible: NO — {len(differing)}/{len(first)} differ between identical "
          f"passes at temperature 0")
    print("    text matching will be DISABLED for this config; the gate will use structural "
          "checks only")
    print(f"    differing: {differing[:5]}{' ...' if len(differing) > 5 else ''}")
    return False


def run_fingerprint(args, ref, items, problems, max_tokens, failures) -> tuple[int, list[str]]:
    """Run the fingerprint tier, either writing a reference or checking against one."""
    if args.write_reference:
        reproducible = measure_reproducibility(args, problems, max_tokens, items)
        if failures:
            return report_write_refusal(failures), failures
        ref = reference_skeleton(args, max_tokens)
        ref["fingerprint"] = {"n": len(problems), "max_tokens": max_tokens,
                              **build_fingerprint(items, reproducible)}
        write_reference(reference_path(args.model_tag, args.config), ref, "score")
        return 0, failures

    failures += compare_fingerprint(ref, items, max_tokens, args.thinking)
    if not ref["fingerprint"].get("reproducible", True):
        print("  note: reference marked non-reproducible; text matching skipped, "
              "structural + parser + token checks still enforced")
    return None, failures


def run_score(args, ref, items, max_tokens, failures) -> tuple[int, list[str]]:
    """Run the score tier, either writing a reference or comparing against one."""
    accuracy, correct, count, misses = score_items(items)
    failures += truncation_failure(items, max_tokens, args.thinking)
    print(f"\n  gsm8k exact match: {accuracy:.4f} ({correct}/{count})")
    if misses:
        print("  sample misses:")
        for miss in misses:
            print(f"    - {miss}")

    if args.write_reference:
        if failures:
            return report_write_refusal(failures), failures
        ref = reference_skeleton(args, max_tokens)
        ref["score"] = {"metric": "gsm8k_exact_match", "value": round(accuracy, 4),
                        "n": count, "max_tokens": max_tokens}
        write_reference(reference_path(args.model_tag, args.config), ref, "fingerprint")
        return 0, failures

    failures += compare_score(ref, accuracy, count, args)
    return None, failures


# ---------------------------------------------------------------------------- #
#                                  REPORTING                                   #
# ---------------------------------------------------------------------------- #

def report_write_refusal(failures: list[str]) -> int:
    """Refuse to capture a reference from a server that already fails structural checks."""
    print("\nFAIL: structural checks failed; refusing to write a reference from a bad server:")
    for failure in failures:
        print(f"  - {failure}")
    return 1


def report_transport_failure(endpoint: str, first_error: str) -> int:
    """Report an unreachable endpoint as an environment error, not an accuracy verdict.

    Distinguished from a genuine failure (exit 1) so a broken port-forward is never mistaken
    for a bad deployment.
    """
    print(f"\nENV ERROR: every request failed at the transport layer.\n"
          f"  endpoint: {endpoint}\n"
          f"  first    : {first_error}\n"
          f"The server is unreachable, so no accuracy verdict was reached. Check the\n"
          f"port-forward or service address before retrying.")
    return 2


def report_verdict(failures: list[str]) -> int:
    """Print the final verdict and return the process exit code."""
    if not failures:
        print("\nACCURACY GATE PASSED")
        return 0
    print(f"\nACCURACY GATE FAILED ({len(failures)} problem(s)):")
    for failure in failures:
        print(f"  - {failure}")
    print("\nThe served model does not match its reference. Do not publish numbers from\n"
          "this server until the cause is understood. Check the resolved argv first:\n"
          "  python3 app/bench/src/validate_manifests.py")
    return 1


def print_gate_header(args, problems: list[dict], max_tokens: int, path: Path) -> None:
    """Print what is being checked, against what, with which settings."""
    print(f"accuracy gate: mode={args.mode} model_tag={args.model_tag} config={args.config}")
    print(f"  endpoint  : {args.endpoint}")
    print(f"  model id  : {args.model_id}")
    print(f"  problems  : {len(problems)} from app/bench/data/gsm8k_subset.jsonl "
          f"({sha256_file(config.EVAL_DATA)[:12]})")
    print(f"  generation: temperature={GEN['temperature']} seed={GEN['seed']} "
          f"top_p={GEN['top_p']} max_tokens={max_tokens} thinking={args.thinking}")
    print(f"  reference : {relative(path)}"
          f"{' (writing)' if args.write_reference else ''}")


# ---------------------------------------------------------------------------- #
#                              ARGUMENT HANDLING                               #
# ---------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["fingerprint", "score"])
    parser.add_argument("--model-tag", required=True, help="e.g. qwen7b, gemma4-31b, kimik3")
    parser.add_argument("--config", required=True,
                        help="server config tag, e.g. bf16|fp8|baseline")
    parser.add_argument("--endpoint", default=config.VLLM_API_URL)
    parser.add_argument("--model-id", default=config.MODEL_ID,
                        help="served-model-name as reported by /v1/models")
    parser.add_argument("--write-reference", action="store_true",
                        help="capture this run as the reference instead of checking against one")
    parser.add_argument("--tolerance", type=float, default=0.99,
                        help="score mode: minimum fraction of the reference score (default 0.99)")
    parser.add_argument("--reference-config", default=None,
                        help="score mode: compare against ANOTHER config's reference score, "
                             "e.g. --config fp8 --reference-config bf16 to check that "
                             "quantization did not degrade accuracy. Defaults to --config.")
    parser.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                        help="hybrid-thinking models (Qwen3, Kimi K3): 'off' suppresses the "
                             "<think> trace so a bounded token budget still reaches a final "
                             "answer. 'auto' uses the model default. Recorded in the reference.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--n", type=int, default=None,
                        help="override problem count (score mode); default is the whole subset")
    return parser.parse_args()


def check_usage(args) -> None:
    """Reject argument combinations that cannot produce a meaningful verdict."""
    if not args.model_id:
        sys.exit("ENV ERROR: --model-id required (or set MODEL_ID)")
    if args.reference_config and args.mode == "fingerprint":
        sys.exit("USAGE ERROR: --reference-config applies to --mode score only.\n"
                 "Fingerprints are exact text comparisons, so comparing one config's text\n"
                 "against another's would fail for the wrong reason: different weights\n"
                 "legitimately produce different text.")
    if args.reference_config and args.write_reference:
        sys.exit("USAGE ERROR: --reference-config and --write-reference are mutually "
                 "exclusive.\nA reference is captured from the server in front of it.")


def select_problems(args, problems: list[dict]) -> tuple[list[dict], int]:
    """Choose the problem slice and token budget for the requested tier."""
    if args.mode == "fingerprint":
        return problems[:FINGERPRINT_N], FINGERPRINT_MAX_TOKENS
    return (problems[:args.n] if args.n else problems), SCORE_MAX_TOKENS


def resolve_reference(args) -> tuple[dict | None, Path]:
    """Load the reference this run compares against, before any request is issued.

    A missing or unusable reference is knowable up front, and finding out after 20 generations
    wastes GPU time and buries the real error underneath request output.
    """
    write_path = reference_path(args.model_tag, args.config)
    if args.write_reference:
        return None, write_path

    # Where a run READS can differ from where it WRITES: an FP8 server is scored against the
    # BF16 reference, because that is the comparison that means something. A standalone FP8
    # score has nothing to be wrong relative to.
    read_path = reference_path(args.model_tag, args.reference_config or args.config)
    ref = load_reference(read_path, args.mode)
    require_tier(ref, args, read_path)
    return ref, read_path


def require_tier(ref: dict, args, path: Path) -> None:
    """Exit if the reference exists but lacks the tier this run needs."""
    if args.mode == "fingerprint" and "fingerprint" not in ref:
        present = ", ".join(sorted(k for k in ref if k in ("score", "fingerprint")))
        sys.exit(
            f"FAIL: {relative(path)} exists but holds no fingerprint block\n"
            f"  (contains: {present or 'neither tier'})\n\n"
            f"A score reference does not gate a benchmark run; the fingerprint tier does.\n"
            f"Capture one against a verified server:\n"
            f"  python3 app/bench/src/accuracy.py --mode fingerprint --write-reference \\\n"
            f"      --model-tag {args.model_tag} --config {args.config} "
            f"--endpoint {args.endpoint} --model-id {args.model_id}")
    if args.mode == "score" and "score" not in ref:
        sys.exit(f"FAIL: {relative(path)} has no score reference. "
                 f"Capture one with --mode score --write-reference")


# ---------------------------------------------------------------------------- #
#                                 ENTRY POINT                                  #
# ---------------------------------------------------------------------------- #

def main() -> int:
    """Run the requested tier and return the process exit code."""
    args = parse_args()
    check_usage(args)
    problems, max_tokens = select_problems(args, load_problems())
    ref, path = resolve_reference(args)
    print_gate_header(args, problems, max_tokens, path)

    items = asyncio.run(run_set(args.endpoint, args.model_id, problems, max_tokens,
                                args.concurrency, "pass1", args.thinking))
    transport = [item for item in items if item.get("transport_error")]
    if transport and len(transport) == len(items):
        return report_transport_failure(args.endpoint, transport[0]["error"])

    # Structural checks always run, reference or not.
    failures = structural_failures(items)
    if args.mode == "fingerprint":
        early, failures = run_fingerprint(args, ref, items, problems, max_tokens, failures)
    else:
        early, failures = run_score(args, ref, items, max_tokens, failures)
    return early if early is not None else report_verdict(failures)


if __name__ == "__main__":
    sys.exit(main())
