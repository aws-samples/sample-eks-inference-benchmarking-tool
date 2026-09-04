# Accuracy references

This directory holds only this README in a fresh clone. `docs/accuracy/*.json` is gitignored. A reference is the output of your server on your hardware, and it is only meaningful as ground truth for a server you have verified yourself, so inheriting a reference from another environment would gate your runs against a configuration you have not inspected.

The gate fails closed, so read this before your first benchmark. With no reference for the `(model_tag, config)` you are running, `suite` aborts before measuring anything:

```
FAIL: no accuracy reference at docs/accuracy/<model-tag>-<config>.json

This gate fails closed. A missing reference is not a pass, because silently
proceeding is the exact failure this gate exists to prevent.
```

There are two ways forward. Capturing a reference is the intended path: it takes one command against a running server and is step 5 of every runbook in [`docs/runbooks/`](../runbooks/README.md). Alternatively, `--skip-accuracy` proceeds and records `accuracy gate : SKIPPED (--skip-accuracy) — results unverified` in the protocol lock, so the result states that nothing verified it.

Capture is a manual step and cannot be automated. A reference is captured from a server, so if the harness generated a missing reference on your behalf, it would capture from the server under test and then compare that server against itself, which always passes. The truncated-argv failure this gate exists for — a server serving at the wrong precision with both parsers dropped, reporting clean numbers — would pass, because the wrong-precision output would have become the reference. Prompt token counts are generated automatically, because they involve no server and so cannot manufacture agreement; see [`docs/prompt_tokens/README.md`](../prompt_tokens/README.md).

There is one JSON file per `(model_tag, config)` pair, at `docs/accuracy/<model-tag>-<config>.json`, flat, matching how recipes and results are named. `app/bench/src/accuracy.py` checks a live server against them.

These files are generated rather than hand-written. Capture them against a server you have independently verified.

## Why they exist

Throughput and latency numbers do not indicate whether a server is serving the configuration it claims to. A truncated `vllm serve` argv dropped `--kv-cache-dtype fp8` and both parsers, and the server came up healthy and served traffic at the wrong precision with no error. Comparing output against a reference catches that; `validate-manifests` catches the truncation itself.

## Two tiers

| Tier | Cost | When | Catches |
| ---- | ---- | ---- | ------- |
| `fingerprint` | ~30 s | before every benchmark, automatically | precision drift, dropped parsers, wrong weights, repetition collapse, empty output |
| `score` | minutes | once per model+config, manually | quantization that degrades answer quality without breaking anything visibly |

## The problem set

A 200-problem GSM8K subset, pinned by digest and committed at `app/bench/data/gsm8k_subset.jsonl`, so no network access is needed at benchmark time. Regenerate or verify it with:

```bash
python3 app/bench/src/main.py fetch-eval-data --verify
```

## Capturing a reference

Run against a known-good server. Both tiers write to the same file and preserve each other, so the order does not matter. Capture runs in-cluster over cluster DNS, like a measurement, so it needs no port-forward:

```bash
# tier 1: fingerprint (fast, gates every benchmark run)
python3 app/bench/src/main.py launch --capture-accuracy-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <service> \
    --mode fingerprint \
    --model-tag qwen7b --config fp8 \
    --model-id <served-model-name> \
    --out-dir docs/results

# tier 2: scored accuracy (slower, run once per model+config)
python3 app/bench/src/main.py launch --capture-accuracy-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <service> \
    --mode score \
    --model-tag qwen7b --config fp8 \
    --model-id <served-model-name> \
    --out-dir docs/results
```

The reference lands here, in `docs/accuracy/`. For a server you can already reach without a tunnel — a local vLLM, or a run from inside the cluster — `main.py accuracy` runs the same checks directly against any endpoint via `--endpoint` and `--write-reference`.

### Checking quantization with `--reference-config`

A standalone FP8 score has no baseline to be measured against. Score the BF16 build first, then check the quantized build against that reference. This involves two servers, so two runs:

```bash
# 1. Capture against the BF16 server.
python3 app/bench/src/main.py launch --capture-accuracy-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <bf16-service> \
    --mode score --model-tag qwen7b --config bf16 \
    --model-id <served-model-name> --out-dir docs/results

# 2. Check the FP8 server against the BF16 result.
python3 app/bench/src/main.py launch \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <fp8-service> \
    --mode score --model-tag qwen7b --config fp8 --reference-config bf16 \
    --model-id <served-model-name> --out-dir docs/results
```

This reads `qwen7b/bf16.json` while serving the FP8 build, and fails if the FP8 score falls below 99% of the reference (adjustable with `--tolerance`). Both runs must use the same `--n`; a 60-problem score has much wider error bars than a 200-problem one, so the gate refuses to compare across counts.

`--reference-config` is score-only. Fingerprints are exact text comparisons, and different weights legitimately produce different text, so cross-config fingerprinting would fail for an unrelated reason.

## Schema

`schema: 1`.

| Field | Meaning |
| ----- | ------- |
| `model_tag`, `config` | identify the pair; must match the `--model-tag` / `--config` passed to the runner |
| `served_model_id` | what `/v1/models` reported when the reference was taken |
| `captured_utc`, `harness_rev` | provenance |
| `generation` | deterministic settings shared by both tiers: `temperature: 0`, `top_p: 1`, `seed: 1337` |
| `dataset.sha256` | digest of `app/bench/data/gsm8k_subset.jsonl`. A mismatch invalidates the reference and the gate aborts. |
| `fingerprint.n`, `fingerprint.max_tokens` | tier 1 problem count and token budget |
| `fingerprint.reproducible` | whether two identical passes agreed. See below. |
| `fingerprint.items[]` | per problem: `sha256` of normalised output, an 80-char `preview` for debugging, `completion_tokens`, `has_reasoning_content`, `finish_reason` |
| `score.value`, `score.n`, `score.max_tokens` | tier 2 GSM8K exact-match result |

`max_tokens` is stored per tier rather than globally, because the two tiers use different budgets (256 and 512). A reference captured at a different budget produces different text and different token counts, so the gate refuses to compare across them.

## `reproducible: false`

`--write-reference` runs the fingerprint set twice and compares. If the two passes disagree at `temperature: 0`, the model is not bit-reproducible.

This is the expected outcome for a batched server, and it is not confined to MoE models. Two measurements, both at FP8 across 4 GPUs, illustrate the range. A 235B-class MoE model with 128 experts and 8 active per token produced different text on 18 of 20 prompts between two identical passes: requests run at concurrency 8, so batch composition varies between passes, which changes expert routing and therefore the numerics. A 31B-class dense model with tensor parallelism and chunked prefill differed on 16 of 20, with no experts involved. The shared cause is that batch composition sets the order of floating-point reductions, so a request batched differently is not guaranteed the same logits, and at FP8 a difference far below the noise floor is enough to move an argmax. Expert routing amplifies this rather than causing it. Treat `reproducible: false` as information about the serving configuration rather than as a fault to investigate.

When a reference records `reproducible: false` the gate switches checks:

| Check | `reproducible: true` | `reproducible: false` |
| ----- | -------------------- | --------------------- |
| Output text hash | enforced per item | **skipped** |
| `completion_tokens` per item | within 2% | **skipped** |
| Mean `completion_tokens` | — | within 20% of the reference |
| Truncation rate | enforced | enforced |
| Non-empty, no repetition collapse | enforced | enforced |
| `reasoning_content` presence | enforced | enforced |

The per-item token band is dropped rather than widened because on that same run individual items moved by 10% between identical passes, so no per-item threshold separates jitter from a real fault. The aggregate mean does separate them: a negative control with the reasoning trace re-enabled failed on mean tokens (256.0 vs 202.4, 26% drift) and on truncation rate (20/20 vs 8/20), so the weaker path still catches genuine drift.

The reproducibility verdict is stored in the file rather than inferred at run time, so a weaker check is visible rather than silent.

## Scores are stable even when text is not

Text-level non-reproducibility does not make the score tier unusable. On that same deployment the reference scored 0.9600 and two subsequent independent checks both scored 0.9700, above the 0.9504 floor. Aggregate accuracy averages out routing jitter.

Two cautions apply on a 200-problem set. A 99% tolerance is about two problems of slack, so for a non-reproducible model, prefer capturing the reference from the lowest of a few runs rather than from a single run. And scores taken at different `--n` are not comparable; the gate refuses to compare them.

## When to recapture

Recapture, and note why alongside the run, when any of these change intentionally:

* the served model weights or quantization
* any serving flag that affects output: precision, parsers, sampling defaults, chat template
* `app/bench/data/gsm8k_subset.jsonl` (the gate will abort on the digest mismatch first)

Recapturing to clear a failing gate defeats the purpose of the check. A mismatch means the server changed, so establish which side is wrong before overwriting the reference.

## Hybrid-thinking models

Some reasoning models emit a `<think>` trace by default. On a maths word problem that trace routinely
exceeds the token budget, so the response never reaches its final answer and the score measures
truncation rather than correctness. Pass `--thinking off` for those models:

```bash
python3 app/bench/src/main.py launch --capture-accuracy-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <service> \
    --mode score --thinking off \
    --model-tag qwen3-235b --config fp8 \
    --model-id <served-model-name> --out-dir docs/results
```

The gate refuses to report a score when more than 20% of responses hit the token ceiling
unfinished, and it records the thinking mode in the reference, so a later run cannot be compared
against a differently configured one.

## Why the gate and the benchmarks use different temperatures

The gate uses its own request path at `temperature: 0`, `top_p: 1`, `seed: 1337`, because a
reference has to be comparable against itself. The benchmark protocols stay at `temperature: 0.7`,
because that is representative traffic and changing it would break comparability with every
run already taken. The two settings are deliberate and independent.
