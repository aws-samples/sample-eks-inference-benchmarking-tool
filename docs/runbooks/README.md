# Runbooks

Each model directory under `docs/runbooks/<model>/` holds a `README.md` overview and one runbook per recipe. A runbook covers the resources to create, how the weights are staged, deployment, the accuracy gate, measurement and teardown. Runbook filenames start with the precision. The [root README](../../README.md) links to each model overview.

## Recipes

| Model | Configuration | Runbook | Weights come from | Also needs |
|---|---|---|---|---|
| GLM-5.2 | p6-b300.48xlarge — NVFP4, TP8 | [`nvfp4-p6-b300.48xlarge-runbook.md`](./glm-5.2/nvfp4-p6-b300.48xlarge-runbook.md) | published NVFP4, Hugging Face to node-local NVMe | node pinned by hostname |
| Kimi K3 | p6-b300.48xlarge — MXFP4, FP8 KV, TP8 | [`mxfp4-p6-b300.48xlarge-baseline-runbook.md`](./kimi-k3/mxfp4-p6-b300.48xlarge-baseline-runbook.md) | published MXFP4, Hugging Face to S3 to node-local NVMe | Pod Identity |
| Llama 4 Maverick | p6-b200.48xlarge — FP8, FP8 KV, TP8, FlashInfer attention | [`fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md`](./llama4-maverick/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) | published FP8, Hugging Face to S3 to node-local NVMe | Pod Identity |
| Mistral Large 3 | p5en.48xlarge — FP8, TP8, FlashAttention MLA | [`fp8-p5en.48xlarge-runbook.md`](./mistral-large-3/fp8-p5en.48xlarge-runbook.md) | published FP8, Hugging Face to S3 to node-local NVMe | Pod Identity; CoreDNS |
| Nemotron 3 Ultra | p6-b200.48xlarge — NVFP4, FP8 KV, DP2×TP4, expert parallel, FlashInfer, bfloat16 Mamba cache (throughput) | [`nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-bf16seq512-flashinfer-ep-runbook.md`](./nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-bf16seq512-flashinfer-ep-runbook.md) | published NVFP4, Hugging Face to S3 to node-local NVMe | Pod Identity |
| Nemotron 3 Ultra | p6-b200.48xlarge — NVFP4, FP8 KV, DP2×TP4, expert parallel, FlashInfer, MTP speculative decoding (latency) | [`nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-mtp5-flashinfer-ep-runbook.md`](./nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-mtp5-flashinfer-ep-runbook.md) | published NVFP4, Hugging Face to S3 to node-local NVMe | Pod Identity |
| Muse-Glimmer 30B | g7e.48xlarge — FP8, FP8 KV, 8 replicas, FlashInfer attention | [`fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer-runbook.md`](./muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer-runbook.md) | locally quantized FP8, S3 to node-local NVMe | Pod Identity |
| Muse-Glimmer 30B | p6-b200.48xlarge — FP8, FP8 KV, 1 GPU, FlashInfer attention | [`fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md`](./muse-glimmer-30b/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) | locally quantized FP8, S3 to node-local NVMe | Pod Identity |
| Qwen2.5-7B-Instruct | g6e.12xlarge — FP8, 1 GPU | [`fp8-g6e.12xlarge-runbook.md`](./qwen2.5-7b/fp8-g6e.12xlarge-runbook.md) | quantized in an init container onto node-local NVMe | none |
| Qwen2.5-7B-Instruct | g7e.12xlarge — FP8, 1 GPU | [`fp8-g7e.12xlarge-runbook.md`](./qwen2.5-7b/fp8-g7e.12xlarge-runbook.md) | quantized in an init container onto node-local NVMe | none |
| Gemma 4 31B | p6-b200.48xlarge — FP8, FP8 KV, 8 replicas, text only, FlashInfer attention | [`fp8-p6-b200.48xlarge-fp8kv-32k-dp8-textonly-flashinfer-runbook.md`](./gemma4-31b/fp8-p6-b200.48xlarge-fp8kv-32k-dp8-textonly-flashinfer-runbook.md) | locally quantized FP8, S3 to node-local NVMe | Pod Identity |
| Gemma 4 E4B | g7e.2xlarge — FP8, FP8 KV, Triton attention | [`fp8-g7e.2xlarge-fp8kv-attn-triton-runbook.md`](./gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-triton-runbook.md) | locally quantized FP8 on FSx | none |
| Gemma 4 E4B | p5.4xlarge — FP8, FP8 KV, Triton attention | [`fp8-p5.4xlarge-fp8kv-attn-triton-runbook.md`](./gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton-runbook.md) | locally quantized FP8, S3 to node-local NVMe | Pod Identity; Capacity Block node pool |

## Checkpoints

Every checkpoint is staged by a Job in [`model-jobs/`](../../model-jobs/README.md): a download Job for published weights, or a quantize Job for weights this repository produces. Each runbook names the Job it runs.

## Studies

Runbooks that are not scoped to a single model live in [`studies/`](./studies/README.md).

- [`g7e.12xlarge-pix-vs-phb-proxy-runbook.md`](./studies/g7e.12xlarge-pix-vs-phb-proxy-runbook.md): what inference loses when two GPUs lose direct peer access.
- [`g7e.12xlarge-profile-gpus-nsight-systems-runbook.md`](./studies/g7e.12xlarge-profile-gpus-nsight-systems-runbook.md): capture an Nsight Systems trace from a serving pod.
