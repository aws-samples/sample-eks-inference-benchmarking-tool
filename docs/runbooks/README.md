# Runbooks

Each model directory under `docs/runbooks/<model>/` holds a `README.md` overview and one runbook per recipe. A runbook covers the resources to create, how the weights are staged, deployment, the accuracy gate, measurement and teardown. Runbook filenames start with the precision. The [root README](../../README.md) links to each model overview.

## Recipes

| Model | Configuration | Runbook | Weights come from | Also needs |
|---|---|---|---|---|
| GLM-5.2 | p6-b200.48xlarge — FP8, FP8 KV, TP8 | [`fp8-p6-b200.48xlarge-runbook.md`](./glm-5.2/fp8-p6-b200.48xlarge-runbook.md) | published FP8, Hugging Face to FSx | none |
| GLM-5.2 | p6-b300.48xlarge — NVFP4, TP8 | [`nvfp4-p6-b300.48xlarge-runbook.md`](./glm-5.2/nvfp4-p6-b300.48xlarge-runbook.md) | published NVFP4, Hugging Face to node-local NVMe | node pinned by hostname |
| Kimi K3 | p6-b200.48xlarge — MXFP4, FP8 KV, TP8 x PP2 across 2 nodes | [`mxfp4-p6-b200.48xlarge-sts-runbook.md`](./kimi-k3/mxfp4-p6-b200.48xlarge-sts-runbook.md) | published MXFP4, Hugging Face to S3 to node-local NVMe | IRSA; EFA; custom image |
| Kimi K3 | p6-b300.48xlarge — MXFP4, FP8 KV, TP8 | [`mxfp4-p6-b300.48xlarge-baseline-runbook.md`](./kimi-k3/mxfp4-p6-b300.48xlarge-baseline-runbook.md) | published MXFP4, Hugging Face to S3 to node-local NVMe | IRSA; custom image |
| Kimi K3 | p6-b300.48xlarge — MXFP4, FP8 KV, TP8, DSpark speculative decoding | [`mxfp4-p6-b300.48xlarge-dspark-runbook.md`](./kimi-k3/mxfp4-p6-b300.48xlarge-dspark-runbook.md) | published MXFP4 and DSpark draft, Hugging Face to S3 to node-local NVMe | IRSA; custom image |
| Llama 4 Maverick | p6-b200.48xlarge — FP8, FP8 KV, TP8, Triton attention | [`fp8-p6-b200.48xlarge-fp8kv-32k-runbook.md`](./llama4-maverick/fp8-p6-b200.48xlarge-fp8kv-32k-runbook.md) | published FP8, Hugging Face to S3 to node-local NVMe | IRSA; benchmark-client node group |
| Llama 4 Maverick | p6-b200.48xlarge — FP8, FP8 KV, TP8, FlashAttention | [`fp8-p6-b200.48xlarge-fp8kv-32k-flashattn-runbook.md`](./llama4-maverick/fp8-p6-b200.48xlarge-fp8kv-32k-flashattn-runbook.md) | published FP8, Hugging Face to S3 to node-local NVMe | IRSA; benchmark-client node group |
| Llama 4 Maverick | p6-b200.48xlarge — FP8, FP8 KV, TP8, FlashInfer attention | [`fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md`](./llama4-maverick/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) | published FP8, Hugging Face to S3 to node-local NVMe | IRSA; benchmark-client node group |
| Nemotron 3 Ultra | p6-b200.48xlarge — NVFP4, FP8 KV, TP8, Triton attention | [`nvfp4-p6-b200.48xlarge-fp8kv-32k-runbook.md`](./nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-runbook.md) | published NVFP4, Hugging Face to S3 to node-local NVMe | IRSA; benchmark-client node group |
| Nemotron 3 Ultra | p6-b200.48xlarge — NVFP4, FP8 KV, TP8, FlashAttention | [`nvfp4-p6-b200.48xlarge-fp8kv-32k-flashattn-runbook.md`](./nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashattn-runbook.md) | published NVFP4, Hugging Face to S3 to node-local NVMe | IRSA; benchmark-client node group |
| Nemotron 3 Ultra | p6-b200.48xlarge — NVFP4, FP8 KV, TP8, FlashInfer attention | [`nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md`](./nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) | published NVFP4, Hugging Face to S3 to node-local NVMe | IRSA; benchmark-client node group |
| Muse-Glimmer 30B | g7e.48xlarge — FP8, FP8 KV, 8 replicas, Triton attention | [`fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-runbook.md`](./muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-runbook.md) | locally quantized FP8, S3 to node-local NVMe | Pod Identity; benchmark-client node group |
| Muse-Glimmer 30B | g7e.48xlarge — FP8, FP8 KV, 8 replicas, FlashInfer attention | [`fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer-runbook.md`](./muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer-runbook.md) | locally quantized FP8, S3 to node-local NVMe | Pod Identity; benchmark-client node group |
| Muse-Glimmer 30B | p6-b200.48xlarge — FP8, FP8 KV, 1 GPU, FlashInfer attention | [`fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md`](./muse-glimmer-30b/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) | locally quantized FP8, S3 to node-local NVMe | Pod Identity; benchmark-client node group |
| Qwen2.5-7B-Instruct | g6e.12xlarge — FP8, 1 GPU | [`fp8-g6e.12xlarge-runbook.md`](./qwen2.5-7b/fp8-g6e.12xlarge-runbook.md) | locally quantized FP8 on FSx | none |
| Qwen2.5-7B-Instruct | g7e.12xlarge — FP8, 1 GPU | [`fp8-g7e.12xlarge-runbook.md`](./qwen2.5-7b/fp8-g7e.12xlarge-runbook.md) | locally quantized FP8 on FSx | none |
| Qwen3-30B-A3B | g7e.12xlarge — FP8, TP2+EP, NCCL default | [`fp8-g7e.12xlarge-switch-default-runbook.md`](./qwen3-30b-a3b/fp8-g7e.12xlarge-switch-default-runbook.md) | published FP8, Hugging Face to a gp3 EBS claim | EBS CSI driver; node pinned by hostname; benchmark-client node group |
| Qwen3-30B-A3B | g7e.12xlarge — FP8, TP2+EP, NCCL `PIX` | [`fp8-g7e.12xlarge-switch-pix-runbook.md`](./qwen3-30b-a3b/fp8-g7e.12xlarge-switch-pix-runbook.md) | published FP8, Hugging Face to a gp3 EBS claim | EBS CSI driver; node pinned by hostname; benchmark-client node group |
| Gemma 4 31B | g7.24xlarge — FP8, TP4 | [`fp8-g7.24xlarge-runbook.md`](./gemma4-31b/fp8-g7.24xlarge-runbook.md) | locally quantized FP8 on FSx | `vllm-sitecustomize` ConfigMap |
| Gemma 4 31B | p6-b200.48xlarge — FP8, FP8 KV, 8 replicas, text only, FlashInfer attention | [`fp8-p6-b200.48xlarge-fp8kv-32k-dp8-textonly-flashinfer-runbook.md`](./gemma4-31b/fp8-p6-b200.48xlarge-fp8kv-32k-dp8-textonly-flashinfer-runbook.md) | locally quantized FP8, S3 to node-local NVMe | Pod Identity; benchmark-client node group |
| Gemma 4 E4B | g7e.2xlarge — FP8, FP8 KV, Triton attention | [`fp8-g7e.2xlarge-fp8kv-attn-triton-runbook.md`](./gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-triton-runbook.md) | locally quantized FP8 on FSx | client node with more than 8 CPU |
| Gemma 4 E4B | p5.4xlarge — FP8, FP8 KV, Triton attention | [`fp8-p5.4xlarge-fp8kv-attn-triton-runbook.md`](./gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton-runbook.md) | locally quantized FP8, S3 to node-local NVMe | Pod Identity; Capacity Block node pool; benchmark-client node group |

## Checkpoints

Every checkpoint is staged by a Job in [`model-jobs/`](../../model-jobs/README.md): a download Job for published weights, or a quantize Job for weights this repository produces. Each runbook names the Job it runs.

## Studies

Runbooks that are not scoped to a single model live in [`studies/`](./studies/README.md).

- [`g7e.12xlarge-pix-vs-phb-proxy-runbook.md`](./studies/g7e.12xlarge-pix-vs-phb-proxy-runbook.md): what inference loses when two GPUs lose direct peer access.
- [`g7e.12xlarge-profile-gpus-nsight-systems-runbook.md`](./studies/g7e.12xlarge-profile-gpus-nsight-systems-runbook.md): capture an Nsight Systems trace from a serving pod.
