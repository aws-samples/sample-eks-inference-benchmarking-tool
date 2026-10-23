# Muse-Glimmer 30B

The runbooks listed in this section are for [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B): a dense model of roughly 29.6 billion parameters, including a 1.8 billion parameter image encoder. Its KV cache is light, about 26 KiB per token at FP8, so the FP8 checkpoint and a full batch of 32k-token sequences fit one GPU. These recipes therefore serve one copy per GPU rather than splitting a copy across GPUs.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Attention backend | Node | GPUs | TP | Runbook |
| -------- | --------- | ----------------- | ---- | ---- | -- | ------- |
| `g7e.48xlarge` | FP8, FP8 KV cache | `FLASHINFER` | Single node, 8 replicas | 8 | unset | [Runbook](./fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer-runbook.md) |
| `p6-b200.48xlarge` | FP8, FP8 KV cache | `FLASHINFER` | Single node, 1 replica | 1 | unset | [Runbook](./fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) |

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `muse-glimmer-30b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for the recipes here. Open-loop runs add an `openloop` suffix.
