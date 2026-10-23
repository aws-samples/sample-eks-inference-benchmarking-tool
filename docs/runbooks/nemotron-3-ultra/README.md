# Nemotron 3 Ultra

The runbooks listed in this section are for [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4): NVIDIA's published NVFP4 build of a hybrid Mamba-2 and mixture-of-experts model with about 550 billion parameters, of which about 55 billion are active per token. Only 12 of its 108 layers use attention, so its KV cache is small, about 6 KiB per token at FP8, plus a fixed Mamba state of about 0.38 GiB per sequence. The ~328 GiB checkpoint is served as two data-parallel replicas of four GPUs each across the eight GPUs of a p6-b200.48xlarge. Two configurations are kept: a throughput-maximizing recipe and a latency-minimizing one.

## Recipes

Pick the recipe for your goal: the bfloat16-Mamba-cache recipe for the highest aggregate throughput, or the MTP recipe for the lowest single-request latency. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Attention backend | Node | GPUs | Parallelism | Optimized for | Runbook |
| -------- | --------- | ----------------- | ---- | ---- | ----------- | ------------- | ------- |
| `p6-b200.48xlarge` | NVFP4, FP8 KV cache | `FLASHINFER`, expert parallel | Single node, 2 replicas | 8 | DP2 × TP4 | Throughput (~4,276 tok/s at concurrency 512) | [Runbook](./nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-bf16seq512-flashinfer-ep-runbook.md) |
| `p6-b200.48xlarge` | NVFP4, FP8 KV cache | `FLASHINFER`, expert parallel, MTP | Single node, 2 replicas | 8 | DP2 × TP4 | Latency (~3.5 ms/token at concurrency 1) | [Runbook](./nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-mtp5-flashinfer-ep-runbook.md) |

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `nemotron-3-ultra-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for the recipes here. Open-loop runs add an `openloop` suffix.
