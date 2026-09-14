# Accuracy references

Before the first benchmark of a model and server config, capture an accuracy reference. Every `measure` run then checks the live server against it and aborts if they disagree.

Every runbook in [`docs/runbooks/`](../runbooks/README.md) includes the capture as a numbered step before its first benchmark, so following a runbook covers this. Capture it once per `(model_tag, config)` pair, against a server you have checked yourself.

With no reference for the pair you are running, `measure` stops before measuring:

```
FAIL: no accuracy reference at docs/accuracy/<model-tag>-<config>.json

This gate fails closed. A missing reference is not a pass, because silently
proceeding is the exact failure this gate exists to prevent.
```

Two ways forward:

- Capture a reference, which is one command against a running server. See [Capturing a reference](#capturing-a-reference).
- Pass `--skip-accuracy`. The run proceeds and the protocol lock records `accuracy gate : SKIPPED (--skip-accuracy) — results unverified`.

This directory holds only this README in a fresh clone. `docs/accuracy/*.json` is gitignored, because a reference describes your server on your hardware. A reference from another environment would gate your runs against a configuration you have not seen.

Files are named `docs/accuracy/<model-tag>-<config>.json`, flat, matching how recipes and results are named. `app/bench/src/accuracy.py` checks a live server against them.

## Why capture is manual

A reference is captured from a server. If the harness generated a missing one on demand, it would capture from the server under test and compare that server against itself, which always passes. The failure this gate exists for would pass too: a server serving at the wrong precision, with both parsers dropped, would have its wrong-precision output recorded as the reference.

Prompt token counts are generated automatically, because they involve no server and cannot manufacture agreement. See [`docs/prompt_tokens/README.md`](../prompt_tokens/README.md).

## What the gate catches

Throughput and latency numbers do not show whether a server is serving the configuration it claims to. A truncated `vllm serve` argv once dropped `--kv-cache-dtype fp8` and both parsers; the server came up healthy and served traffic at the wrong precision with no error. Comparing output against a reference catches that. `validate-manifests` catches the truncation itself.

## Two tiers

| Tier | Cost | When | Catches |
| ---- | ---- | ---- | ------- |
| `fingerprint` | ~30 s | before every benchmark, automatically | precision drift, dropped parsers, wrong weights, repetition collapse, empty output |
| `score` | minutes | once per model+config, manually | quantization that degrades answer quality without breaking anything visibly |

## The problem set

A 200-problem GSM8K subset, pinned by digest and committed at `app/bench/data/gsm8k_subset.jsonl`, so no network access is needed at benchmark time. Regenerate or verify it with:

```bash
eks-bench fetch-eval-data --verify
```

## Capturing a reference

Run against a server you have checked. Both tiers write to the same file and preserve each other, so order does not matter. Capture runs in-cluster over cluster DNS, so it needs no port-forward:

```bash
# tier 1: fingerprint (fast, gates every benchmark run)
eks-bench capture-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <service> \
    --mode fingerprint \
    --model-tag qwen7b --config fp8 \
    --model-id <served-model-name> \
    --out-dir docs/results

# tier 2: scored accuracy (slower, run once per model+config)
eks-bench capture-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <service> \
    --mode score \
    --model-tag qwen7b --config fp8 \
    --model-id <served-model-name> \
    --out-dir docs/results
```

The reference lands here, in `docs/accuracy/`. For a server you can already reach without a tunnel, such as a local vLLM or a run from inside the cluster, `eks-bench capture-reference-local` runs the same checks against any `--endpoint`.

### Checking quantization with `--reference-config`

A standalone FP8 score has no baseline. Score the BF16 build first, then check the quantized build against that reference. Two servers, so two runs:

```bash
# 1. Capture against the BF16 server.
eks-bench capture-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <bf16-service> \
    --mode score --model-tag qwen7b --config bf16 \
    --model-id <served-model-name> --out-dir docs/results

# 2. Check the FP8 server against the BF16 result. This runs where you type it, so give it an
#    endpoint you can reach: from inside the cluster, or a port-forward to the FP8 Service.
eks-bench check-accuracy \
    --mode score --model-tag qwen7b --config fp8 --reference-config bf16 \
    --endpoint http://<fp8-service>.<namespace>.svc.cluster.local:8000/v1 \
    --model-id <served-model-name>
```

This reads `qwen7b-bf16.json` while serving the FP8 build, and fails if the FP8 score falls below 99% of the reference. Adjust with `--tolerance`. Both runs must use the same `--n`: a 60-problem score has much wider error bars than a 200-problem one, so the gate refuses to compare across counts.

`--reference-config` belongs to `check-accuracy` and `capture-reference`, not to `measure`. The gate a `measure` run invokes is always `--mode fingerprint` against `<model-tag>-<config>.json`.

`--reference-config` is score-only. Fingerprints are exact text comparisons, and different weights produce different text, so cross-config fingerprinting would fail for an unrelated reason.

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

`max_tokens` is stored per tier, because the two tiers use different budgets, 256 and 512. A reference captured at a different budget produces different text and different token counts, so the gate refuses to compare across them.

## `reproducible: false`

Capture runs the fingerprint set twice and compares. If the two passes disagree at `temperature: 0`, the model is not bit-reproducible. This is the expected outcome for a batched server, and it is not confined to MoE models.

Batch composition sets the order of floating-point reductions, so a request batched differently is not guaranteed the same logits, and at FP8 a difference far below the noise floor can move an argmax. Expert routing amplifies this; it is not the cause. Two measurements, both FP8 across 4 GPUs, show the range:

| Model | Prompts differing between two identical passes |
| ----- | --- |
| 235B-class MoE, 128 experts, 8 active per token, concurrency 8 | 18 of 20 |
| 31B-class dense, tensor parallel, chunked prefill | 16 of 20 |

`reproducible: false` describes the serving configuration. It is not a fault to investigate.

When a reference records it, the gate switches checks:

| Check | `reproducible: true` | `reproducible: false` |
| ----- | -------------------- | --------------------- |
| Output text hash | enforced per item | **skipped** |
| `completion_tokens` per item | within 2% | **skipped** |
| Mean `completion_tokens` | — | within 20% of the reference |
| Truncation rate | enforced | enforced |
| Non-empty, no repetition collapse | enforced | enforced |
| `reasoning_content` presence | enforced | enforced |

The per-item token band is dropped instead of widened because individual items moved by 10% between identical passes on that run, so no per-item threshold separates jitter from a real fault. The aggregate mean does separate them: a negative control with the reasoning trace re-enabled failed on mean tokens, 256.0 against 202.4, and on truncation rate, 20/20 against 8/20.

The verdict is stored in the file, not inferred at run time, so the weaker check is visible.

## Scores stay usable when text does not

Text-level non-reproducibility does not make the score tier unusable. On the same deployment the reference scored 0.9600 and two later independent checks scored 0.9700, above the 0.9504 floor. Aggregate accuracy averages out routing jitter.

Two cautions on a 200-problem set. A 99% tolerance is about two problems of slack, so for a non-reproducible model capture the reference from the lowest of a few runs. And scores taken at different `--n` are not comparable; the gate refuses to compare them.

## When to recapture

Recapture, and note why alongside the run, when any of these change intentionally:

* the served model weights or quantization
* any serving flag that affects output: precision, parsers, sampling defaults, chat template
* `app/bench/data/gsm8k_subset.jsonl`, though the gate aborts on the digest mismatch first

A mismatch means the server changed. Establish which side is wrong before overwriting the reference.

## Hybrid-thinking models

Some reasoning models emit a `<think>` trace by default. On a maths word problem that trace routinely exceeds the token budget, so the response never reaches its final answer and the score measures truncation. Pass `--thinking off` for those models:

```bash
eks-bench capture-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service <service> \
    --mode score --thinking off \
    --model-tag qwen3-235b --config fp8 \
    --model-id <served-model-name> --out-dir docs/results
```

The gate refuses to report a score when more than 20% of responses hit the token ceiling unfinished, and it records the thinking mode in the reference, so a later run cannot be compared against a differently configured one.

## Why the gate and the benchmarks use different temperatures

The gate uses `temperature: 0`, `top_p: 1`, `seed: 1337`, because a reference has to be comparable against itself. The benchmark protocols use `temperature: 0.7`, because that is representative traffic and changing it would break comparability with every run already taken.
