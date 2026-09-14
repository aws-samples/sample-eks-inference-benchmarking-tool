# GLM-5.2

The runbooks listed in this section are for the [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2): a mixture-of-experts model from Z.ai with roughly 753 billion total parameters, released under the MIT license. GLM-5.2 targets coding, reasoning, and long-horizon agentic work, and supports a 1-million-token context window.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Node | GPUs | TP | Runbook |
| -------- | --------- | ---- | ---- | -- | ------- |
| `p5en.48xlarge` | FP8 | Single node inference | 8 | 8 | [Runbook](./fp8-p5en.48xlarge-runbook.md) |
| `p6-b200.48xlarge` | FP8 | Single node inference | 8 | 8 | [Runbook](./fp8-p6-b200.48xlarge-runbook.md) |
| `p6-b200.48xlarge` | NVFP4 | Single node inference | 8 | 8 | [Runbook](./nvfp4-p6-b200.48xlarge-runbook.md) |
| `p6-b300.48xlarge` | NVFP4 | Single node inference | 8 | 8 | [Runbook](./nvfp4-p6-b300.48xlarge-runbook.md) |

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `glm-5.2-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, which it does not for these three recipes. Open-loop runs add an `openloop` suffix.

## Known issues

These recipes run upstream `vllm/vllm-openai:v0.26.0` rather than the AWS Deep Learning Container that most recipes in this repo use, so a version bump here is a larger change than it looks. Each runbook states where its weights come from, which differs across the three: a node directory, an FSx PVC after quantization, and a Hugging Face download to `emptyDir`.

For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
