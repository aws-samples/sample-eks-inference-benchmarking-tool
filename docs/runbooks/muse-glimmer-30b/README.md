# Muse-Glimmer 30B

The runbooks listed in this section are for [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B): a dense model of roughly 29.6 billion parameters, including a 1.8 billion parameter image encoder. Its KV cache is light, about 26 KiB per token at FP8, so the FP8 checkpoint and a full batch of 32k-token sequences fit one GPU. These recipes therefore serve one copy per GPU rather than splitting a copy across GPUs.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Attention backend | Node | GPUs | TP | Runbook |
| -------- | --------- | ----------------- | ---- | ---- | -- | ------- |
| `g7e.48xlarge` | FP8, FP8 KV cache | `TRITON_ATTN` | Single node, 8 replicas | 8 | unset | [Runbook](./fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-runbook.md) |
| `g7e.48xlarge` | FP8, FP8 KV cache | `FLASHINFER` | Single node, 8 replicas | 8 | unset | [Runbook](./fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer-runbook.md) |
| `p6-b200.48xlarge` | FP8, FP8 KV cache | `FLASHINFER` | Single node, 1 replica | 1 | unset | [Runbook](./fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) |

The two g7e.48xlarge recipes are an attention-backend pair and differ only in `--attention-backend`.

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `muse-glimmer-30b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for the recipes here. Open-loop runs add an `openloop` suffix. The g7e.48xlarge runbooks also write a `replica-balance` file recording how requests split across the eight replicas.

## Known issues

Every recipe here serves an FP8 checkpoint quantized in this repo by [`fp8-model-quantize.yaml`](../../../model-jobs/muse-glimmer-30b/fp8-model-quantize.yaml) and copied to S3 by [`fp8-model-fsx-to-s3.yaml`](../../../model-jobs/muse-glimmer-30b/fp8-model-fsx-to-s3.yaml). The quantize Job runs in the vLLM serving image, because the quantizer image's transformers release predates this architecture.

Blackwell instances (g7e) have compatibility quirks that affected these recipes during development. The fixes have already been applied to the recipes in this section. For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
