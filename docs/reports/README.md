# Reports

This directory is empty in a fresh clone. `docs/reports/*.md` is gitignored. A report interprets measurements taken on your own infrastructure: it quotes numbers from `docs/results/`, which is gitignored for the same reason, and carries cluster names, namespaces, weight paths, and account-specific details. This README is the only tracked file in this directory.

The harness produces a `.txt` per run recording what was measured. A report is the accompanying document that a reader who was not present can act on. The two have different audiences, so this file documents the report format.

## Report format

| Section | Answers |
|---|---|
| **Title** and bold subtitle | What was measured, on what hardware, at what precision, and whether this is a baseline or a tuned run |
| **Caveat blockquote** | Any condition that changes how the numbers below should be read. State it before the numbers, not after |
| **Executive Summary** | The headline figure in context, and what the shape of the curve means |
| **Key numbers** | Peak throughput and its concurrency, a recommended operating point, and single-stream latency |
| **Benchmark Results** | A provenance line (model, GPU, vLLM version, precision, parallelism), then the sweep table: concurrency, tokens/sec, req/sec, p50, p99 |
| **Key insight** | An interpretation of the table: where it saturates, and what doubling concurrency buys |
| **Batch offline vs real-time** | Which settings to change for throughput-first versus latency-first, since one server supports both objectives |
| **vLLM Configuration** | Every flag as Parameter / Value / Note, with ✅ and ❌ marking what must and must not be set |
| **Key parameters explained** | Why each non-obvious flag is set the way it is, and what changes otherwise |
| **Why this hardware** | Memory arithmetic, interconnect, whether pipeline parallelism is needed, weight staging — then an instance comparison table with a verdict per candidate |
| **Deployment notes** | What operators encounter in practice: image tag, staging time, node-local storage, startup timings, concurrency planning |
| **Configuration Stack** | The manifest inline, with `<ACCOUNT>`, `<REGION>`, `<NAMESPACE>` and `<WEIGHTS_BUCKET>` left as placeholders |

### Conventions

- **Quote numbers inline rather than citing a result file, a report, or a section of one.** Both directories are gitignored, so a reader in another environment does not have them, and a citation by path, by filename, or as "see section 6 of the b300 report" cannot be followed. Name the recipe, protocol, and profile instead, which is provenance a reader can reproduce.
- **State the output token budget wherever throughput appears.** The same server on the same node recorded 14,992 tokens/sec at a 1,024-token budget and 3,785 at 150, a 4.0x difference attributable to truncation alone.
- **State what was not measured.** An absent number is not a passing one. Streaming latency, accuracy, and multi-node scaling are the three most commonly assumed.
- **Use placeholders in the Deployment manifest**, matching [`k8s/examples/`](../../k8s/examples/): `012345678901`, `my-namespace`, `my-cluster`.

## Template

Below is a filled-out report with the model name replaced by a label and the figures replaced by illustrative ones. Copy it, replace every value, and delete any section you cannot fill rather than leaving an empty heading.

````markdown
# vLLM Inference on p6-b200 (Blackwell)

*Benchmark Results & Configuration Guide - MoE Model A (FP8) - Single-Node Baseline*

**Note:** Tokens/sec is not a like-for-like measure across models. A mixture-of-experts model activates a fraction of its parameters per token, so a larger total parameter count can produce more tokens/sec than a smaller one while doing more work per token. Compare tokens/sec only between runs at the same output budget, and treat a figure from a model with a different active parameter count as a different measurement rather than a slower or faster one.

## Executive Summary

A single p6-b200.48xlarge serves MoE Model A at 12,288 tokens/sec at a 1,024-token output budget, on one node with no multi-node networking, no pipeline parallelism and no shared filesystem. Throughput scales steeply to concurrency 128 and then flattens, so the peak at 512 is a ceiling rather than a recommendation. FP8 weights are 700 GiB against 1,440 GiB of HBM, which leaves 740 GiB for KV cache and is what keeps this configuration on one node.

## Key numbers

- **p6-b200 peak:** 12,288 tokens/sec at concurrency=512 (single node, 8 GPUs)
- **Recommended concurrency:** 8,192 tokens/sec at concurrency=128, p50 16.0 s
- **Single stream:** 154 tokens/sec

## Benchmark Results - p6-b200.48xlarge (vLLM 0.21.0 + TP=8)

**Model:** MoE Model A (FP8, 700 GiB checkpoint) | **GPU:** 8 x 180 GiB Blackwell | **vLLM:** 0.21.0 | **Precision:** FP8 weights, FP8 KV cache | **Parallelism:** TP=8 over NVLink

**Baseline - 1,024 max output tokens**

| Concurrency | Tokens/sec | Req/sec | p50 Latency | p99 Latency |
|---|---|---|---|---|
| 1 | 154 | 0.15 | 6.7 s | 7.1 s |
| 8 | 1,024 | 1.00 | 8.0 s | 8.4 s |
| 32 | 3,072 | 3.00 | 10.7 s | 11.5 s |
| 128 | 8,192 | 8.00 | 16.0 s | 17.8 s |
| 256 | 10,240 | 10.00 | 25.6 s | 28.1 s |
| 512 | 12,288 | 12.00 | 42.7 s | 47.9 s |

**Key insight:** throughput scales steeply to concurrency 128 and then flattens. Doubling from 128 to 256 buys 25% more throughput for 60% more p50 latency; doubling again buys 20% for 67%. The knee is only visible because the sweep includes both 128 and 256. Prefix caching was off, so these figures include full prefill on every request.

## Batch Offline vs. Real-Time Inference

One server supports both objectives, and the settings for each differ.

* For batch work, run at concurrency 512 and leave the batch-token budget high. Total job time is the objective and per-request latency is not constrained.
* For real-time work, run at concurrency 128 or below, drop `--max-num-batched-tokens` to 4096 or 8192 and `--max-num-seqs` to 128 or 256, then re-measure with a latency protocol at a realistic input size. A throughput sweep does not report latency metrics, and validating a latency-tuned configuration with short prompts does not exercise the shape it was tuned for.

## vLLM Configuration

| Parameter | Value | Note |
|---|---|---|
| `--tensor-parallel-size` | 8 | single node, NVLink |
| `--max-num-seqs` | 512 | highest concurrency served |
| `--max-num-batched-tokens` | 32768 | fixes the KV budget; not auto |
| `--max-model-len` | 32768 (pinned) | not auto, or deployments stop being comparable |
| `--gpu-memory-utilization` | 0.90 | leaves KV cache headroom |
| `--quantization` | fp8 | halves memory traffic against BF16 |
| `--kv-cache-dtype` | fp8_e4m3 | doubles KV capacity per GiB |
| `--enable-chunked-prefill` | ✅ On | prevents head-of-line blocking |
| `--enable-prefix-caching` | ❌ Off | required off for an uncached baseline |
| `--enforce-eager` | ❌ Not set | would disable CUDA graphs |
| `--speculative-config` | ❌ Not tested | clean baseline |

## Key parameters explained

- **`--enable-prefix-caching` off.** A cache hit skips prefill, so any throughput figure taken with it on is inflated. The provenance block records the observed hit rate; confirm it reads 0% before quoting the figure as uncached.
- **`--max-model-len` pinned, not auto.** Left to vLLM it is inferred from the model and moves between deployments, which makes two runs non-comparable. It must also stay at or below `--max-num-batched-tokens` or the server refuses to start.
- **`--max-num-batched-tokens` set explicitly.** This is the token budget for one forward pass and the strongest control on first-token latency. At concurrency 512 with 2,048-token inputs the batch owes 32 passes before every request has started prefilling.
- **`--kv-cache-dtype` fp8_e4m3.** Doubles the KV cache per GiB, which is what allows concurrency 512 at this context length. Some attention backends reject FP8 KV; confirm yours accepts it before assuming the flag took effect.
- **`--enforce-eager` unset.** Eager mode costs the CUDA-graph speedup. Set it only if graph capture fails, and record that you did.

## Why FP8 on p6-b200?

- **GPU memory:** 8 x 180 GiB is 1,440 GiB. The 700 GiB FP8 checkpoint takes 49%, leaving 740 GiB where KV cache is the largest consumer.
- **Single-node NVLink:** all 8 GPUs sit on one mesh, so expert traffic never leaves the node. No EFA dependency and no NCCL version matching.
- **No pipeline parallelism needed:** the checkpoint fits with room to spare, so TP=8 is sufficient. The remaining scaling lever is context length, not node count.
- **Node-local NVMe:** the checkpoint stages in roughly 40 minutes, and later pods skip the transfer.
- **Tensor parallelism:** TP=8 is the maximum on this node and it already spans all 8 GPUs.

| Instance | GPU | GPU memory | FP8 (700 GiB) | Verdict |
|---|---|---|---|---|
| p6-b200.48xlarge | 8 x B200 | 1,440 GiB | 1 node | measured here, 12,288 tokens/sec |
| 8-GPU Hopper node | 8 x H200 | 1,128 GiB | 1 node | viable, untested here |
| 8-GPU prior-gen node | 8 x H100 | 640 GiB | 2 nodes | needs pipeline parallelism and EFA |
| 8-GPU L40S node | 8 x L40S | 358 GiB | does not fit | insufficient HBM |

## p6-b200 Deployment

* Pin the image to a specific tag. A floating tag makes two runs non-comparable.
* Stage weights once to shared storage, roughly 40 minutes. An init container check skips the download when the shards are already present.
* Stage inside a GPU-holding pod. Staging as a separate job risks another team claiming the GPUs while the transfer runs.
* Verify the checkpoint before serving. Assert the shard count matches the index and that no partial files remain.
* Startup is roughly 12 minutes cold, most of it weight load and graph capture. Set a startup probe long enough to cover it.
* Plan replica count from the recommended concurrency, not the peak.

## What was not measured

Open-loop capacity at a target request rate, behaviour with prefix caching enabled, multi-node scaling, and accuracy beyond the gate. An absent number is not a passing one.

## Configuration Stack

Replace `<ACCOUNT>`, `<REGION>`, `<NAMESPACE>` and `<WEIGHTS_BUCKET>` for your environment.

```yaml
# MoE Model A, FP8, on one p6-b200.48xlarge, TP=8.
# Measured 12,288 tokens/sec at concurrency 512, 1,024-token output budget.
# FP8 weights are 700 GiB against 1,440 GiB of HBM, which leaves 740 GiB for KV
# cache and is why this fits one node without pipeline parallelism.
# Prefix caching is OFF deliberately: a cache hit skips prefill and inflates throughput.
# The download runs in an initContainer so the GPU allocation is held from scheduling.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: moe-model-a-fp8
  namespace: <NAMESPACE>
  labels: { model: moe-model-a, config: fp8 }
spec:
  replicas: 1
  strategy: { type: Recreate }   # one pod claims all 8 GPUs; a rolling update deadlocks
  selector:
    matchLabels: { app: moe-model-a-fp8 }
# ... remainder of the manifest inline, with account, region, namespace and bucket as placeholders
```
````

### Adapting it

Split the configuration commentary under **Settings specific to this model** and **Rules that apply regardless of model** whenever a reader might copy a flag into a different recipe.

## Naming

```
<model>-<instance>-report.md               # the full report format above, written for a reader
<model>-<instance>-benchmark-report.md     # working notes, same evidence, less surrounding context
<topic>-findings.md                        # a focused study, e.g. nccl-p2p-findings.md
<topic>-plan.md                            # method agreed before the run
integration-friction-report-<model>.md     # what failed and what it cost, kept for the next run
```

Use the full instance type rather than the family. Reports predating this convention may exist locally and are not renamed.
