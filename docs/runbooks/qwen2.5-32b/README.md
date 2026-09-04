# Qwen2.5-32B-Instruct

The runbooks listed in this section are for the [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct): a 32-billion parameter dense instruction-tuned model from Alibaba's Qwen team. Qwen2.5 supports a 128K-token context window, generates up to 8K tokens, and covers 29 languages. The recipes here leave the context length at the value the model config declares; reaching 128K needs rope scaling, which they do not enable.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Node | GPUs | TP | Runbook |
| -------- | --------- | ---- | ---- | -- | ------- |
| `g7e.4xlarge` | FP8 | Single node inference | 1 | unset | [Runbook](./fp8-g7e.4xlarge-runbook.md) |

The recipe requires a locally quantized checkpoint at `/models/quantized/Qwen2.5-32B-Instruct-fp8`; this repository does not provide a published-checkpoint staging Job for that path. Start with the [recipe runbook](./fp8-g7e.4xlarge-runbook.md), complete its environment exports and `./scripts/install-prerequisites.sh`, then follow the canonical [`job-quantize-model.yaml` workflow](../../../k8s/examples/README.md#job-quantize-modelyaml) before continuing to checkpoint verification and deployment.

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `qwen2.5-32b-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does not for this recipe. Open-loop runs add an `openloop` suffix.

## Known issues

This is the only recipe in the repo that keeps `--enforce-eager`. At this model size on g7e, capturing CUDA graphs hits an illegal memory access, so the flag is deliberate. Removing it reintroduces the crash, and eager mode costs the CUDA-graph speedup, so do not copy the flag into other recipes without a reason.

For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
