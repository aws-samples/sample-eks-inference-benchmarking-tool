# Gemma 4 E4B
The runbooks listed in this section are for the [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it): a 4-billion active parameter multimodal model from Google DeepMind, 8 billion on disk. It accepts text, image and audio inputs, generates text outputs, and carries a 128K-token context window. Sliding-window attention on 35 of its 42 layers and KV sharing across the last 18 make its cache unusually small for its size, so these recipes are compute-bound rather than memory-bound.
## Recipes
Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).
| Instance | Precision | Node | GPUs | TP | Runbook |
| -------- | --------- | ---- | ---- | -- | ------- |
| `g7e.2xlarge` | FP8, FP8 KV cache, Triton attention | Single node inference | 1 | unset | [Runbook](./fp8-g7e.2xlarge-fp8kv-attn-triton-runbook.md) |
[`fp8-p5.4xlarge-fp8kv-attn-triton.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton.yaml) is the same configuration on one H100 (p5.4xlarge). It has no runbook yet. It reads the same FP8 checkpoint from S3 rather than the FSx claim, staged by [`fp8-model-fsx-to-s3.yaml`](../../../model-jobs/gemma4-e4b/fp8-model-fsx-to-s3.yaml).
## Results
Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `gemma4-e4b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for both recipes here. Open-loop runs add an `openloop` suffix.
## Known issues
Blackwell instances (g7, g7e) have compatibility quirks that affected these recipes during development. The fixes have already been applied to the recipes in this section. For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
