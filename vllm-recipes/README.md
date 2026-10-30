# vLLM recipes

One manifest per row. Each serves exactly one model on one instance type at one precision, and pairs with the runbook that runs it. Each model lists up to three recipes: the configurations with the best combination of latency and throughput. Attention backend is `Auto` where the recipe pins none and vLLM chooses.

Every recipe passes `--no-enable-prefix-caching`, so its benchmarks measure uncached prefill. Each recipe's deploy script runs `eks-bench validate-manifests` on it after deploying; run `eks-bench validate-manifests <model>/<recipe>.yaml` yourself to check a recipe without deploying it.

## Recipes

### GLM-5.2

| Runbook | Instance | Weights Precision | KV Precision | Attention backend | GPUs | Parallelism | Kubernetes resources | Image |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Runbook](../docs/runbooks/glm-5.2/nvfp4-p6-b300.48xlarge-runbook.md) | `p6-b300.48xlarge` | NVFP4 | 16-bit | Auto | 8x B300 | TP8 | Deployment, Service | Upstream vLLM |

### Kimi K3

| Runbook | Instance | Weights Precision | KV Precision | Attention backend | GPUs | Parallelism | Kubernetes resources | Image |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Runbook](../docs/runbooks/kimi-k3/mxfp4-p6-b300.48xlarge-baseline-runbook.md) | `p6-b300.48xlarge` | MXFP4 | FP8 | Auto; `FLASHINFER` for MLA prefill | 8x B300 | TP8 | Deployment, Service | Upstream vLLM |

### Llama 4 Maverick

| Runbook | Instance | Weights Precision | KV Precision | Attention backend | GPUs | Parallelism | Kubernetes resources | Image |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Runbook](../docs/runbooks/llama4-maverick/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) | `p6-b200.48xlarge` | FP8 | FP8 | `FLASHINFER` | 8x B200 | TP8 | Deployment, Service | Upstream vLLM |

### Mistral Large 3

| Runbook | Instance | Weights Precision | KV Precision | Attention backend | GPUs | Parallelism | Kubernetes resources | Image |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Runbook](../docs/runbooks/mistral-large-3/fp8-p5en.48xlarge-runbook.md) | `p5en.48xlarge` | FP8 | 16-bit | `FLASH_ATTN_MLA` | 8x H200 | TP8 | Deployment, Service | Upstream vLLM |

### Nemotron 3 Ultra

| Runbook | Instance | Weights Precision | KV Precision | Attention backend | GPUs | Parallelism | Kubernetes resources | Image |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Runbook](../docs/runbooks/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-bf16seq512-flashinfer-ep-runbook.md) | `p6-b200.48xlarge` | NVFP4 | FP8 | `FLASHINFER` | 8x B200 | DP2 × TP4, expert parallel | Deployment, Service | Upstream vLLM |
| [Runbook](../docs/runbooks/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-mtp5-flashinfer-ep-runbook.md) | `p6-b200.48xlarge` | NVFP4 | FP8 | `FLASHINFER` | 8x B200 | DP2 × TP4, expert parallel, MTP | Deployment, Service | Upstream vLLM |

### Muse-Glimmer 30B

| Runbook | Instance | Weights Precision | KV Precision | Attention backend | GPUs | Parallelism | Kubernetes resources | Image |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Runbook](../docs/runbooks/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer-runbook.md) | `g7e.48xlarge` | FP8 | FP8 | `FLASHINFER` | 8x RTX PRO 6000 | N/A | Deployment, Service | Upstream vLLM |
| [Runbook](../docs/runbooks/muse-glimmer-30b/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) | `p6-b200.48xlarge` | FP8 | FP8 | `FLASHINFER` | 1x B200 | N/A | Deployment, Service | Upstream vLLM |

### Qwen2.5-7B-Instruct

| Runbook | Instance | Weights Precision | KV Precision | Attention backend | GPUs | Parallelism | Kubernetes resources | Image |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Runbook](../docs/runbooks/qwen2.5-7b/fp8-g6e.12xlarge-runbook.md) | `g6e.12xlarge` | FP8 | 16-bit | Auto | 1x L40S | TP1 | Deployment, Service | AWS DLC |
| [Runbook](../docs/runbooks/qwen2.5-7b/fp8-g7e.12xlarge-runbook.md) | `g7e.12xlarge` | FP8 | 16-bit | Auto | 1x RTX PRO 6000 | TP1 | Deployment, Service | AWS DLC |

### Gemma 4 31B

| Runbook | Instance | Weights Precision | KV Precision | Attention backend | GPUs | Parallelism | Kubernetes resources | Image |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Runbook](../docs/runbooks/gemma4-31b/fp8-p6-b200.48xlarge-fp8kv-32k-dp8-textonly-flashinfer-runbook.md) | `p6-b200.48xlarge` | FP8 | FP8 | `FLASHINFER` | 8x B200 | N/A | Deployment, Service | Upstream vLLM |

### Gemma 4 E4B

| Runbook | Instance | Weights Precision | KV Precision | Attention backend | GPUs | Parallelism | Kubernetes resources | Image |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| [Runbook](../docs/runbooks/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-triton-runbook.md) | `g7e.2xlarge` | FP8 | FP8 | `TRITON_ATTN` | 1x RTX PRO 6000 | N/A | Deployment, Service | Upstream vLLM |
| [Runbook](../docs/runbooks/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton-runbook.md) | `p5.4xlarge` | FP8 | FP8 | `TRITON_ATTN` | 1x H100 | N/A | Deployment, Service | Upstream vLLM |

## Naming

`<model>/<precision>-<instance>[-<config>].yaml`. The instance is the full type, never the family tag, because the same model runs on different sizes within a family and the family alone hides the GPU count. A `<config>` suffix appears only to break a genuine collision, and names the axis that differs.

## What a recipe is not

A recipe runs `vllm serve`. Weight staging Jobs live in [`model-jobs/`](../model-jobs/README.md), quantization is a Job from [`k8s/examples/`](../k8s/examples/README.md), and infrastructure examples live there too. None of those belong here.
