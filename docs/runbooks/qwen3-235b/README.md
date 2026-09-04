# Qwen3-235B-A22B

The runbooks listed in this section are for the [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B): a 235-billion parameter mixture-of-experts model from Alibaba's Qwen team that activates 22 billion parameters per token. Qwen3-235B-A22B handles text and code generation and supports a 128K-token context window. The recipes here leave the context length at the value the model config declares; reaching 128K needs rope scaling, which they do not enable.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Node | GPUs | TP | PP | Runbook |
| -------- | --------- | ---- | ---- | -- | -- | ------- |
| `g7e.12xlarge` | FP8 | Multi node inference | 4 | 2 | 2 | [Runbook](./fp8-g7e.12xlarge-lws-runbook.md) |

This is the only serving configuration. The recipe is a LeaderWorkerSet that splits the model across two nodes with tensor parallelism inside each node and pipeline parallelism between them.

A second runbook drives this same recipe across two nodes to compare `NCCL_P2P_LEVEL` over EFA rather than to measure serving performance. To learn more, see [Interconnect studies](../../../experiments/README.md#interconnect-studies).

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `qwen3-235b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, so results from this recipe carry its `lws` tag. Open-loop runs add an `openloop` suffix.

## Known issues

This recipe needs the LeaderWorkerSet controller and the EFA device plugin installed before it will schedule, and it needs both nodes in the same placement group to get the interconnect it expects. It runs on g7e, so the Blackwell compatibility fixes are already applied in the recipe.

For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
