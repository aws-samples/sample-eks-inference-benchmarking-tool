# Llama 3.3 70B Instruct

The runbook in this section serves [`nvidia/Llama-3.3-70B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8), NVIDIA's Model Optimizer FP8 checkpoint of Meta's dense, instruction-tuned Llama 3.3 70B model. The model supports up to 131,072 tokens; the recipe pins 8,192 for a fixed and affordable benchmark context.

## Recipes

Use the following runbook to get started. For additional protocols and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

| Instance | Precision | Node | GPUs | TP | Runbook |
| -------- | --------- | ---- | ---- | -- | ------- |
| `g7e.12xlarge` | FP8 weights, 16-bit KV | Single node inference | 2 | 2 | [Runbook](./fp8-g7e.12xlarge-runbook.md) |

The public checkpoint is downloaded by vLLM into node-backed `emptyDir` storage. No FSx filesystem or separate weight-staging Job is required, but replacing the pod discards the checkpoint and downloads it again.

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows `llama3.3-70b-fp8-g7e.12xlarge-<protocol>[-<profile>]`, so it identifies the model, precision, hardware, and workload that produced it. Open-loop runs add an `openloop` suffix.

## Known issues

The recipe depends on vLLM 0.26 selecting a Blackwell-compatible attention backend and on the checkpoint's Model Optimizer metadata. It deliberately disables prefix caching for a cache-free baseline and uses TP2 across both GPUs; neither choice should be changed between comparison runs.

For Blackwell startup and driver failures, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md).
