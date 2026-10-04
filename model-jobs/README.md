# Model jobs

Jobs that stage a checkpoint on a shared claim or in S3, one directory per model. Run the one your recipe needs before deploying it, once per filesystem or bucket.

Each Job a runbook uses has a script beside it with the same name, such as `fp8-model-download.sh` for `fp8-model-download.yaml`. The script fills in the Job's placeholders from the variables the runbook exports, applies it, waits for it, and fails unless the checkpoint it staged passes a check.

A download stages weights that are already published. A quantize job produces a checkpoint that is not published, so it has no repository id and a recipe can only serve it by path. Recipes do not do either: each waits for a complete checkpoint to appear and starts serving when it does, so staging is a separate step you run first and reuse across every recipe that shares a model.

## Contents

| Job | Source weights | Lands on the PVC at |
|---|---|---|
| [`gemma4-e4b/fp8-model-quantize.yaml`](gemma4-e4b/fp8-model-quantize.yaml) | `google/gemma-4-E4B-it` | `/models/quantized/gemma-4-E4B-it-fp8` |
| [`gemma4-31b/fp8-model-quantize.yaml`](gemma4-31b/fp8-model-quantize.yaml) | `google/gemma-4-31b-it` | `/models/quantized/gemma-4-31b-it-fp8` |
| [`glm-5.2/fp8-model-download.yaml`](glm-5.2/fp8-model-download.yaml) | `zai-org/GLM-5.2-FP8` | `/models/glm-5.2-fp8` |
| [`muse-glimmer-30b/fp8-model-quantize.yaml`](muse-glimmer-30b/fp8-model-quantize.yaml) | `meta-models/Muse-Glimmer-30B` | `/models/quantized/Muse-Glimmer-30B-fp8` |
| [`qwen2.5-7b/fp8-model-quantize.yaml`](qwen2.5-7b/fp8-model-quantize.yaml) | `Qwen/Qwen2.5-7B-Instruct` | `/models/quantized/Qwen2.5-7B-Instruct-fp8` |
| [`qwen3-30b-a3b/fp8-model-download.yaml`](qwen3-30b-a3b/fp8-model-download.yaml) | `Qwen/Qwen3-30B-A3B-FP8` | `/models/qwen3-30b-a3b-fp8` on the `g7e12-switch-model-storage` gp3 claim |
| [`qwen3-235b/fp8-model-download.yaml`](qwen3-235b/fp8-model-download.yaml) | `Qwen/Qwen3-235B-A22B-FP8` | `/models/qwen3-235b-a22b-fp8` |

| Utility | Purpose |
|---|---|
| [`gemma4-e4b/fp8-model-verify.yaml`](gemma4-e4b/fp8-model-verify.yaml) | Check a quantized checkpoint on the claim, and fail if it is not what its config claims |
| [`kimi-k3/mxfp4-model-download.yaml`](kimi-k3/mxfp4-model-download.yaml) | Stage the published checkpoint from Hugging Face to S3 file by file, resumably, and fail if any file is missing |
| [`kimi-k3/dspark-draft-download.yaml`](kimi-k3/dspark-draft-download.yaml) | Stage the published DSpark draft to S3 with the same stager |
| [`llama4-maverick/fp8-model-download.yaml`](llama4-maverick/fp8-model-download.yaml) | Stage the published checkpoint from Hugging Face to S3, and fail on a shard-count or size mismatch |
| [`nemotron-3-ultra/nvfp4-model-download.yaml`](nemotron-3-ultra/nvfp4-model-download.yaml) | Stage the published NVFP4 checkpoint from Hugging Face to S3, and fail on a shard-count or size mismatch |
| [`muse-glimmer-30b/fp8-model-fsx-to-s3.yaml`](muse-glimmer-30b/fp8-model-fsx-to-s3.yaml) | Copy the quantized checkpoint from the claim to S3, and fail on a shard-count mismatch |

Each download Job is three lines of shell, and everything that varies is in `env`. This follows NVIDIA's [Dynamo GLM recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.2/model-cache/model-download.yaml), including the pinned hub client and `HF_XET_HIGH_PERFORMANCE`, because Xet is the accelerated transfer path now that `hf_transfer` has been removed from the client.

These Jobs differ from that recipe in one respect: `--local-dir`. Dynamo lets the weights sit in the hub cache and serves them by repository id. The recipes here serve a path, and some of those paths are `/models/quantized/...` checkpoints produced by a quantize job, which have no repository id to resolve. The Job therefore writes to an explicit directory that a recipe's poller can wait on.


