# Gemma 4 E4B

The runbooks listed in this section are for the [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it): a 4-billion active parameter multimodal model from Google DeepMind, 8 billion on disk. It accepts text, image and audio inputs, generates text outputs, and carries a 128K-token context window. Sliding-window attention on 35 of its 42 layers and KV sharing across the last 18 make its cache unusually small for its size, so these recipes are compute-bound rather than memory-bound.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

Each precision ships a throughput-tuned and a latency-tuned recipe that differ only in `--max-num-batched-tokens` and `--stream-interval`, so the pair isolates that trade. There is one GPU per node here, so tear down one config before deploying the other.

| Instance | Precision | Node | GPUs | TP | Runbook |
| -------- | --------- | ---- | ---- | -- | ------- |
| `g7.2xlarge` | BF16 | Single node inference | 1 | unset | [Runbook](./bf16-g7.2xlarge-runbook.md) |
| `g7.2xlarge` | FP8 | Single node inference | 1 | unset | [Runbook](./fp8-g7.2xlarge-runbook.md) |
| `g7e.2xlarge` | FP8 | Single node inference | 1 | unset | [Runbook](./fp8-g7e.2xlarge-runbook.md) |
| `g7e.2xlarge` | FP8, FP8 KV cache | Single node inference | 1 | unset | [Runbook](./fp8-g7e.2xlarge-fp8kv-runbook.md) |

Two further FP8 recipes add GPU n-gram speculative decoding, `fp8-g7.2xlarge-ngram-throughput.yaml` and `fp8-g7.2xlarge-ngram-latency.yaml`. They are identical to the FP8 pair apart from `--speculative-config`, so running them back to back against it attributes the difference to speculation alone. Speculation is always its own labelled config and is never folded into a baseline.



## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `gemma4-e4b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does for the speculative and open-loop configs. Open-loop runs add an `openloop` suffix.

## Known issues

`--kv-cache-dtype` must stay `auto` on this model. Pinning the literal `bfloat16` passes argument parsing and then aborts the engine during KV initialisation, because that value is only valid for a checkpoint whose weights default to FP8. Both runbooks record this under Settings.

Blackwell instances (g7, g7e) have compatibility quirks that affected these recipes during development. The fixes have already been applied to the recipes in this section. For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
