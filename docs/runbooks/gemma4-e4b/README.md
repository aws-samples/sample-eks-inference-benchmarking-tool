# Gemma 4 E4B
The runbooks listed in this section are for the [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it): a 4-billion active parameter multimodal model from Google DeepMind, 8 billion on disk. It accepts text, image and audio inputs, generates text outputs, and carries a 128K-token context window. Sliding-window attention on 35 of its 42 layers and KV sharing across the last 18 make its cache unusually small for its size, so these recipes are compute-bound rather than memory-bound.
## Recipes
Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).
| Instance | Precision | Node | GPUs | TP | Runbook |
| -------- | --------- | ---- | ---- | -- | ------- |
| `g7e.2xlarge` | FP8, FP8 KV cache, Triton attention | Single node inference | 1 | unset | [Runbook](./fp8-g7e.2xlarge-fp8kv-attn-triton-runbook.md) |
| `p5.4xlarge` | FP8, FP8 KV cache, Triton attention | Single node inference | 1 | unset | [Runbook](./fp8-p5.4xlarge-fp8kv-attn-triton-runbook.md) |
## Results
Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `gemma4-e4b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for the recipes here. Open-loop runs add an `openloop` suffix.
