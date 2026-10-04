# Nemotron 3 Ultra

The runbooks listed in this section are for [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4): NVIDIA's published NVFP4 build of a hybrid Mamba-2 and mixture-of-experts model with about 550 billion parameters, of which about 55 billion are active per token. Only 12 of its 108 layers use attention, so its KV cache is small, about 6 KiB per token at FP8, plus a fixed Mamba state of about 0.38 GiB per sequence. The ~328 GiB checkpoint is served as one tensor-parallel copy across all eight GPUs of a p6-b200.48xlarge.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Attention backend | Node | GPUs | TP | Runbook |
| -------- | --------- | ----------------- | ---- | ---- | -- | ------- |
| `p6-b200.48xlarge` | NVFP4, FP8 KV cache | `TRITON_ATTN` | Single node, 1 replica | 8 | 8 | [Runbook](./nvfp4-p6-b200.48xlarge-fp8kv-32k-runbook.md) |
| `p6-b200.48xlarge` | NVFP4, FP8 KV cache | `FLASH_ATTN` | Single node, 1 replica | 8 | 8 | [Runbook](./nvfp4-p6-b200.48xlarge-fp8kv-32k-flashattn-runbook.md) |
| `p6-b200.48xlarge` | NVFP4, FP8 KV cache | `FLASHINFER` | Single node, 1 replica | 8 | 8 | [Runbook](./nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) |

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `nemotron-3-ultra-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for the recipes here. Open-loop runs add an `openloop` suffix.
