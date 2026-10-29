# Kimi K3

The runbooks listed in this section are for the [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3): a 2.8-trillion parameter mixture-of-experts model from Moonshot AI that activates roughly 104 billion parameters per token. Kimi K3 is built on Kimi Delta Attention and Attention Residuals, accepts native vision input, and supports a 1-million-token context window.

## Recipes

Use the runbook recipe below to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

The `--config` column is what you pass to `measure`.

| Instance | Precision | `--config` | Node | GPUs | TP | PP | Runbook |
| -------- | --------- | ---------- | ---- | ---- | -- | -- | ------- |
| `p6-b300.48xlarge` | MXFP4 | `baseline` | Single node inference | 8 | 8 | none | [Runbook](./mxfp4-p6-b300.48xlarge-baseline-runbook.md) |

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `kimi-k3-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, so every recipe here carries its topology tag. Open-loop runs add an `openloop` suffix.
