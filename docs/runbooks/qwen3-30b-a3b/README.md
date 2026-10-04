# Qwen3-30B-A3B

The runbooks listed in this section are for [Qwen/Qwen3-30B-A3B-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8): a mixture-of-experts model of 30.5 billion parameters, of which about 3.3 billion run per token. Each of its 48 layers holds 128 experts and routes every token to 8 of them. It is a hybrid-thinking model, so its accuracy gate is captured with thinking disabled.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | NCCL transport | Node | GPUs | TP | Runbook |
| -------- | --------- | -------------- | ---- | ---- | -- | ------- |
| `g7e.12xlarge` | FP8, BF16 KV cache | NCCL default selection | Single node, pinned | 2 | 2, expert parallel | [Runbook](./fp8-g7e.12xlarge-switch-default-runbook.md) |
| `g7e.12xlarge` | FP8, BF16 KV cache | `NCCL_P2P_LEVEL=PIX` | Single node, pinned | 2 | 2, expert parallel | [Runbook](./fp8-g7e.12xlarge-switch-pix-runbook.md) |

For the full three-arm, two-model study, see the [PIX versus PHB-proxy study runbook](../studies/g7e.12xlarge-pix-vs-phb-proxy-runbook.md).

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `qwen3-30b-a3b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for the recipes here: `fp8-switch-default` and `fp8-switch-pix`, the study's config tags. Open-loop runs add an `openloop` suffix.
