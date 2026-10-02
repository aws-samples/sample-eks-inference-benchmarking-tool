# Qwen2.5-7B-Instruct

The runbooks listed in this section are for the [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct): a 7-billion parameter dense instruction-tuned model from Alibaba's Qwen team. Qwen2.5 supports a 128K-token context window, generates up to 8K tokens, and covers 29 languages. The recipes here leave the context length at the value the model config declares; reaching 128K needs rope scaling, which they do not enable.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).


| Instance | Precision | `--config` | Node | GPUs | TP | Runbook |
| -------- | --------- | ---------- | ---- | ---- | -- | ------- |
| `p5en.48xlarge` | BF16 | `bf16` | Single node inference | 1 | unset | [Runbook](./bf16-p5en.48xlarge-runbook.md) |
| `g7e.12xlarge` | FP8 | `fp8` | Single node inference | 1 | 1 | [Runbook](./fp8-g7e.12xlarge-runbook.md) |


Every recipe here fits on a single GPU, so the larger instances measure one GPU of that instance type rather than the whole machine.

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `qwen2.5-7b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, so neither recipe here carries one. Open-loop runs add an `openloop` suffix.

## Checkpoint staging

All BF16 recipes stage the published `Qwen/Qwen2.5-7B-Instruct` checkpoint. Every FP8 recipe uses `/models/quantized/Qwen2.5-7B-Instruct-fp8`, which can be populated either by the published `Qwen/Qwen2.5-7B-Instruct-FP8` staging Job or by local quantization; see the [model-download provenance notes](../../../model-jobs/README.md#fp8-weights-downloaded-or-quantized). The FP8 `g7e` runbook applies the published staging Job directly; to produce the path by local quantization instead, use the canonical [`job-quantize-model.yaml` workflow](../../../k8s/examples/README.md#job-quantize-modelyaml).

The g7e recipes run on Blackwell, so their compatibility fixes are already applied in the recipes.

For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
