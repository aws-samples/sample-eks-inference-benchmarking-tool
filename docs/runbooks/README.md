# Runbooks

Each model directory contains a `README.md` overview and the runbooks for that model: the resources to create, where the weights come from, prerequisite checks, deployment, accuracy gating, measurement, and teardown. The [root README](../../README.md) links to each model overview.

## Per-recipe prerequisites

Runbooks are grouped by model under `docs/runbooks/<model>/`; each directory's `README.md` is the model overview, and each runbook filename starts with the precision.

| Model | Configuration | Runbook | Weights come from | Also needs |
|---|---|---|---|---|
| GLM-5.2 | p5en.48xlarge — FP8 | [`fp8-p5en.48xlarge-runbook.md`](./glm-5.2/fp8-p5en.48xlarge-runbook.md) | node directory | none |
| GLM-5.2 | p6-b200.48xlarge — FP8 | [`fp8-p6-b200.48xlarge-runbook.md`](./glm-5.2/fp8-p6-b200.48xlarge-runbook.md) | published FP8 staged to FSx | none |
| GLM-5.2 | p6-b300.48xlarge — NVFP4 | [`nvfp4-p6-b300.48xlarge-runbook.md`](./glm-5.2/nvfp4-p6-b300.48xlarge-runbook.md) | Hugging Face to emptyDir | none |
| Gemma 4 31B | g6e.12xlarge — BF16 | [`bf16-g6e.12xlarge-runbook.md`](./gemma4-31b/bf16-g6e.12xlarge-runbook.md) | published BF16 staged to FSx | none |
| Gemma 4 31B | g7.24xlarge — BF16 | [`bf16-g7.24xlarge-runbook.md`](./gemma4-31b/bf16-g7.24xlarge-runbook.md) | published BF16 staged to FSx | none |
| Gemma 4 31B | g7e.2xlarge — BF16 | [`bf16-g7e.2xlarge-runbook.md`](./gemma4-31b/bf16-g7e.2xlarge-runbook.md) | published BF16 downloaded in pod | none |
| Gemma 4 31B | g7e.4xlarge — BF16 | [`bf16-g7e.4xlarge-runbook.md`](./gemma4-31b/bf16-g7e.4xlarge-runbook.md) | published BF16 staged to FSx | none |
| Gemma 4 31B | g6e.12xlarge — FP8 | [`fp8-g6e.12xlarge-runbook.md`](./gemma4-31b/fp8-g6e.12xlarge-runbook.md) | locally quantized FP8 on FSx | none |
| Gemma 4 31B | g7.24xlarge — FP8 | [`fp8-g7.24xlarge-runbook.md`](./gemma4-31b/fp8-g7.24xlarge-runbook.md) | locally quantized FP8 on FSx | none |
| Gemma 4 31B | g7.24xlarge — FP8, TP=4, NCCL P2P study | [`fp8-g7.24xlarge-p2p-runbook.md`](./gemma4-31b/fp8-g7.24xlarge-p2p-runbook.md) | locally quantized FP8 staged on FSx | none |
| Gemma 4 31B | g7e.4xlarge — FP8 | [`fp8-g7e.4xlarge-runbook.md`](./gemma4-31b/fp8-g7e.4xlarge-runbook.md) | locally quantized FP8 on FSx | none |
| Llama 3.3 70B Instruct | g7e.12xlarge — FP8, TP2 | [`fp8-g7e.12xlarge-runbook.md`](./llama3.3-70b/fp8-g7e.12xlarge-runbook.md) | Hugging Face to emptyDir | benchmark-reserved node |
| Kimi K3 (2.8T MoE) | p6-b200.48xlarge — MXFP4, lws | [`mxfp4-p6-b200.48xlarge-lws-runbook.md`](./kimi-k3/mxfp4-p6-b200.48xlarge-lws-runbook.md) | S3 to node-local NVMe | LWS, EFA, custom image |
| Kimi K3 (2.8T MoE) | p6-b200.48xlarge — MXFP4, lws-dspark | [`mxfp4-p6-b200.48xlarge-lws-dspark-runbook.md`](./kimi-k3/mxfp4-p6-b200.48xlarge-lws-dspark-runbook.md) | FSx PVC, weights must be staged | LWS, EFA |
| Kimi K3 (2.8T MoE) | p6-b200.48xlarge — MXFP4, sts | [`mxfp4-p6-b200.48xlarge-sts-runbook.md`](./kimi-k3/mxfp4-p6-b200.48xlarge-sts-runbook.md) | S3 to node-local NVMe | EFA, custom image |
| Kimi K3 (2.8T MoE) | p6-b200.48xlarge — MXFP4, sts-latency | [`mxfp4-p6-b200.48xlarge-sts-latency-runbook.md`](./kimi-k3/mxfp4-p6-b200.48xlarge-sts-latency-runbook.md) | S3 to node-local NVMe | EFA, custom image |
| Kimi K3 (2.8T MoE) | p6-b300.48xlarge — MXFP4, baseline | [`mxfp4-p6-b300.48xlarge-baseline-runbook.md`](./kimi-k3/mxfp4-p6-b300.48xlarge-baseline-runbook.md) | S3 to node-local NVMe | custom image |
| Kimi K3 (2.8T MoE) | p6-b300.48xlarge — MXFP4, dspark | [`mxfp4-p6-b300.48xlarge-dspark-runbook.md`](./kimi-k3/mxfp4-p6-b300.48xlarge-dspark-runbook.md) | S3 to node-local NVMe | custom image |
| Kimi K3 (2.8T MoE) | p6-b300.48xlarge — MXFP4, ep | [`mxfp4-p6-b300.48xlarge-ep-runbook.md`](./kimi-k3/mxfp4-p6-b300.48xlarge-ep-runbook.md) | S3 to node-local NVMe | custom image |
| Kimi K3 (2.8T MoE) | u-p6e-gb200x72 — MXFP4 | [`mxfp4-u-p6e-gb200x72-runbook.md`](./kimi-k3/mxfp4-u-p6e-gb200x72-runbook.md) | FSx PVC, weights must be staged | none |
| Qwen2.5-32B-Instruct | g7e.4xlarge — FP8 | [`fp8-g7e.4xlarge-runbook.md`](./qwen2.5-32b/fp8-g7e.4xlarge-runbook.md) | locally quantized FP8 on FSx | none |
| Qwen2.5-7B-Instruct | g6e.4xlarge — BF16 | [`bf16-g6e.4xlarge-runbook.md`](./qwen2.5-7b/bf16-g6e.4xlarge-runbook.md) | published BF16 staged to FSx | none |
| Qwen2.5-7B-Instruct | g7e.12xlarge — BF16 | [`bf16-g7e.12xlarge-runbook.md`](./qwen2.5-7b/bf16-g7e.12xlarge-runbook.md) | published BF16 staged to FSx | none |
| Qwen2.5-7B-Instruct | p5.4xlarge — BF16 | [`bf16-p5.4xlarge-runbook.md`](./qwen2.5-7b/bf16-p5.4xlarge-runbook.md) | published BF16 staged to FSx | none |
| Qwen2.5-7B-Instruct | p5en.48xlarge — BF16 | [`bf16-p5en.48xlarge-runbook.md`](./qwen2.5-7b/bf16-p5en.48xlarge-runbook.md) | published BF16 staged to FSx | none |
| Qwen2.5-7B-Instruct | g6e.4xlarge — FP8 | [`fp8-g6e.4xlarge-runbook.md`](./qwen2.5-7b/fp8-g6e.4xlarge-runbook.md) | published or locally quantized FP8 pre-staged on FSx | none |
| Qwen2.5-7B-Instruct | g7e.12xlarge — FP8 | [`fp8-g7e.12xlarge-runbook.md`](./qwen2.5-7b/fp8-g7e.12xlarge-runbook.md) | published FP8 staged to FSx; local output also accepted | none |
| Qwen2.5-7B-Instruct | g7e.12xlarge — FP8, kvscales | [`fp8-g7e.12xlarge-kvscales-runbook.md`](./qwen2.5-7b/fp8-g7e.12xlarge-kvscales-runbook.md) | published FP8 staged to FSx; local output also accepted | none |
| Qwen2.5-7B-Instruct | p5.4xlarge — FP8 | [`fp8-p5.4xlarge-runbook.md`](./qwen2.5-7b/fp8-p5.4xlarge-runbook.md) | published or locally quantized FP8 pre-staged on FSx | none |
| Qwen2.5-7B-Instruct | p5en.48xlarge — FP8 | [`fp8-p5en.48xlarge-runbook.md`](./qwen2.5-7b/fp8-p5en.48xlarge-runbook.md) | published FP8 staged to FSx; local output also accepted | none |
| Qwen3-235B-A22B | g7e.12xlarge — FP8, lws | [`fp8-g7e.12xlarge-lws-runbook.md`](./qwen3-235b/fp8-g7e.12xlarge-lws-runbook.md) | published FP8 staged to FSx | LWS, EFA |
| Qwen3-235B-A22B | 2x g7e.12xlarge — FP8, multi-node EFA | [`fp8-g7e.12xlarge-multi-node-efa-runbook.md`](./qwen3-235b/fp8-g7e.12xlarge-multi-node-efa-runbook.md) | published FP8 staged to FSx | LWS, EFA, capacity reservation |

## Quantization

Published checkpoints are staged by the model-specific Jobs indexed in [`model-downloads/`](../../model-downloads/README.md#fp8-weights-downloaded-or-quantized). When a recipe requires local conversion, follow the [`job-quantize-model.yaml` workflow](../../k8s/examples/README.md#job-quantize-modelyaml); the model overview and runbook identify those cases.

## Studies

Runbooks that are not scoped to a single model live in [`studies/`](./studies/README.md). A model-specific investigation stays in its model directory; a study that spans models, or a technique that applies to any recipe, belongs here.

- [`g7e.12xlarge-pix-vs-phb-proxy-runbook.md`](./studies/g7e.12xlarge-pix-vs-phb-proxy-runbook.md) — two-model, three-arm default/PIX/forced-SHM study with a balanced NCCL matrix, isolated benchmark client, durable provenance, and UID-guarded cleanup.
- [`g7e.12xlarge-profile-gpus-nsight-systems-runbook.md`](./studies/g7e.12xlarge-profile-gpus-nsight-systems-runbook.md) — capture an Nsight Systems trace from a serving pod.
