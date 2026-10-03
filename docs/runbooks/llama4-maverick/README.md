# Llama 4 Maverick

The runbooks listed in this section are for [meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8): a mixture-of-experts model of roughly 400 billion parameters, about 17 billion of them active per token, with 128 routed experts on every second layer. Every expert stays resident, so the ~388 GiB FP8 checkpoint needs a whole 8-GPU node. These recipes therefore split one copy across all eight GPUs with tensor parallelism. Its KV cache is light, about 96 KiB per token at FP8.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Attention backend | Node | GPUs | TP | Runbook |
| -------- | --------- | ----------------- | ---- | ---- | -- | ------- |
| `p6-b200.48xlarge` | FP8, FP8 KV cache | `TRITON_ATTN` | Single node, 1 replica | 8 | 8 | [Runbook](./fp8-p6-b200.48xlarge-fp8kv-32k-runbook.md) |
| `p6-b200.48xlarge` | FP8, FP8 KV cache | `FLASH_ATTN` | Single node, 1 replica | 8 | 8 | [Runbook](./fp8-p6-b200.48xlarge-fp8kv-32k-flashattn-runbook.md) |
| `p6-b200.48xlarge` | FP8, FP8 KV cache | `FLASHINFER` | Single node, 1 replica | 8 | 8 | [Runbook](./fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer-runbook.md) |

The three recipes are an attention-backend set and differ only in `--attention-backend`. `TRITON_ATTN` is the baseline; each of the other two is its own labelled config and is never folded into it.

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `llama4-maverick-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for the recipes here. Open-loop runs add an `openloop` suffix.

## Known issues

Every recipe here serves the published FP8 checkpoint, staged once from Hugging Face to S3 by [`fp8-model-download.yaml`](../../../model-jobs/llama4-maverick/fp8-model-download.yaml). The `meta-llama` repository is gated, so the `hf-token` must belong to an account that accepted the Llama 4 license. The Job can instead point `MODEL_REPO` at the ungated `unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8` mirror, which it records as carrying the same shards.

The recipes pass no `--quantization` flag. The checkpoint declares a compressed-tensors FP8 export and vLLM resolves the loader from it; pinning the plain `fp8` loader on a foreign export is a known failure. For background on other issues encountered while building recipes, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md).
