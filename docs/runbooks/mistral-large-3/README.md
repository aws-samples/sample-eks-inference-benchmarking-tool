# Mistral Large 3

The runbooks listed in this section are for [mistralai/Mistral-Large-3-675B-Instruct-2512](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512): a multimodal mixture-of-experts model of 675 billion parameters, about 41 billion of them active per token, with 128 routed experts, 4 active per token. Every expert stays resident, so the ~635 GiB FP8 checkpoint needs a whole 8-GPU node. These recipes therefore split one copy across all eight GPUs with tensor parallelism. It uses multi-head latent attention, so its KV cache is about 68.6 KiB per token at 16 bits, held in full on every GPU.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Attention backend | Node | GPUs | TP | Runbook |
| -------- | --------- | ----------------- | ---- | ---- | -- | ------- |
| `p5en.48xlarge` | FP8, 16-bit KV cache | `FLASH_ATTN_MLA` | Single node, 1 replica | 8 | 8 | [Runbook](./fp8-p5en.48xlarge-runbook.md) |

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `mistral-large-3-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`. Open-loop runs add an `openloop` suffix.
