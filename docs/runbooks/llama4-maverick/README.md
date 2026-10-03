# Llama 4 Maverick

The runbooks listed in this section are for [meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8): a mixture-of-experts model of roughly 400 billion parameters, about 17 billion of them active per token, with 128 routed experts on every second layer. Every expert stays resident, so the ~388 GiB FP8 checkpoint needs a whole 8-GPU node. These recipes therefore split one copy across all eight GPUs with tensor parallelism. Its KV cache is light, about 96 KiB per token at FP8.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Attention backend | Node | GPUs | TP | Runbook |
| -------- | --------- | ----------------- | ---- | ---- | -- | ------- |
| `p6-b200.48xlarge` | FP8, FP8 KV cache | `TRITON_ATTN` | Single node, 1 replica | 8 | 8 | [Runbook](./fp8-p6-b200.48xlarge-fp8kv-32k-runbook.md) |
| `p6-b200.48xlarge` | FP8, FP8 KV cache | `FLASH_ATTN` | Single node, 1 replica | 8 | 8 | [Runbook](./fp8-p6-b200.48xlarge-fp8kv-32k-flashattn-runbook.md) |
| `p6-b200.48xlarge` | FP8, FP8 KV cache | `FLASHINFER` | Single node, 1 replica | 8 | 8 | [Runbook](./fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) |

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `llama4-maverick-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for the recipes here. Open-loop runs add an `openloop` suffix.
