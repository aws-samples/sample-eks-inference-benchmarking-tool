# Qwen2.5-7B-Instruct

The runbooks listed in this section are for the [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct): a 7-billion parameter dense instruction-tuned model from Alibaba's Qwen team. Qwen2.5 supports a 128K-token context window, generates up to 8K tokens, and covers 29 languages. The recipes here leave the context length at the value the model config declares; reaching 128K needs rope scaling, which they do not enable.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).


| Instance | Precision | `--config` | Node | GPUs | TP | Runbook |
| -------- | --------- | ---------- | ---- | ---- | -- | ------- |
| `g6e.12xlarge` | FP8 | `fp8` | Single node inference | 1 | 1 | [Runbook](./fp8-g6e.12xlarge-runbook.md) |
| `g7e.12xlarge` | FP8 | `fp8` | Single node inference | 1 | 1 | [Runbook](./fp8-g7e.12xlarge-runbook.md) |


The model fits on a single GPU, so each recipe measures one GPU of the instance rather than the whole machine.

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `qwen2.5-7b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, so the recipes here carry none. Open-loop runs add an `openloop` suffix.
