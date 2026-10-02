# Gemma 4 31B

The runbooks listed in this section are for the [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it): a 31-billion parameter dense multimodal model from Google DeepMind. Gemma 4 accepts text and image inputs, generates text outputs, and includes built-in reasoning, native function calling, a 256K-token context window, and support for over 140 languages.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Node | GPUs | TP | Runbook |
| -------- | --------- | ---- | ---- | -- | ------- |
| `g7.24xlarge` | BF16 | Single node inference | 4 | 4 | [Runbook](./bf16-g7.24xlarge-runbook.md) |
| `g7e.4xlarge` | BF16 | Single node inference | 1 | unset | [Runbook](./bf16-g7e.4xlarge-runbook.md) |
| `g7.24xlarge` | FP8 | Single node inference | 4 | 4 | [Runbook](./fp8-g7.24xlarge-runbook.md) |
| `g7e.4xlarge` | FP8 | Single node inference | 1 | unset | [Runbook](./fp8-g7e.4xlarge-runbook.md) |
| `g7e.4xlarge` | FP8, FP8 KV, 32k context | Single node inference | 1 | unset | [Runbook](./fp8-g7e.4xlarge-fp8kv-32k-runbook.md) |
| `g7e.2xlarge` | FP8, FP8 KV, 32k context | Single node inference | 1 | unset | [Runbook](./fp8-g7e.2xlarge-fp8kv-32k-runbook.md) |

Another recipe serves this model on `g7.24xlarge` to compare `NCCL_P2P_LEVEL` settings rather than to measure serving performance. To learn more, see [Interconnect studies](../../../experiments/README.md#interconnect-studies).

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `gemma4-31b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>` (e.g., the NCCL study arm described below includes its p2p-pix tag). Open-loop runs add an `openloop` suffix.

## Known issues

Blackwell instances (g7, g7e) have compatibility quirks that affected these recipes during development. The fixes have already been applied to the recipes in this section. For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
