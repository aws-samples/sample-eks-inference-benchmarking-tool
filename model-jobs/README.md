# Model jobs

Jobs that put a checkpoint on a shared PVC, one directory per model. Apply the one your recipe needs before deploying it, once per filesystem.

A download stages weights that are already published. A quantize job produces a checkpoint that is not published, so it has no repository id and a recipe can only serve it by path. Recipes do not do either: each waits for a complete checkpoint to appear and starts serving when it does, so staging is a separate step you run first and reuse across every recipe that shares a model.

## Contents

| Job | Source weights | Lands on the PVC at |
|---|---|---|
| [`gemma4-e4b/fp8-model-quantize.yaml`](gemma4-e4b/fp8-model-quantize.yaml) | `google/gemma-4-E4B-it` | `/models/quantized/gemma-4-E4B-it-fp8` |
| [`gemma4-31b/fp8-model-quantize.yaml`](gemma4-31b/fp8-model-quantize.yaml) | `google/gemma-4-31b-it` | `/models/quantized/gemma-4-31b-it-fp8` |
| [`glm-5.2/fp8-model-download.yaml`](glm-5.2/fp8-model-download.yaml) | `zai-org/GLM-5.2-FP8` | `/models/glm-5.2-fp8` |
| [`muse-glimmer-30b/fp8-model-quantize.yaml`](muse-glimmer-30b/fp8-model-quantize.yaml) | `meta-models/Muse-Glimmer-30B` | `/models/quantized/Muse-Glimmer-30B-fp8` |
| [`qwen2.5-7b/bf16-model-download.yaml`](qwen2.5-7b/bf16-model-download.yaml) | `Qwen/Qwen2.5-7B-Instruct` | `/models/Qwen2.5-7B-Instruct` |
| [`qwen2.5-7b/fp8-model-download.yaml`](qwen2.5-7b/fp8-model-download.yaml) | `Qwen/Qwen2.5-7B-Instruct-FP8` | `/models/quantized/Qwen2.5-7B-Instruct-fp8` |
| [`qwen2.5-7b/fp8-model-quantize.yaml`](qwen2.5-7b/fp8-model-quantize.yaml) | `Qwen/Qwen2.5-7B-Instruct` | `/models/quantized/Qwen2.5-7B-Instruct-fp8` |
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

The generic templates these are derived from are [`job-download-weights-to-pvc.yaml`](../k8s/examples/job-download-weights-to-pvc.yaml) and [`job-quantize-model.yaml`](../k8s/examples/job-quantize-model.yaml). The Jobs here record the per-model values that a runbook would otherwise only describe in prose.

## Prerequisites

A PVC to write to, named `fsx-pvc` in every Job here. See [`persistentvolume-fsx-lustre.yaml`](../k8s/examples/persistentvolume-fsx-lustre.yaml), and change `claimName` if yours is named differently.

A `hf-token` Secret in the same namespace, since Secrets are namespaced. Covered by Quickstart step 4 in the [root README](../README.md).

Enough free space for the whole model. This is not checked automatically. To confirm, `kubectl -n "$NAMESPACE" exec` into any pod with the PVC mounted and run `df -h /models`.

A quantize job additionally needs a GPU free on the instance type it pins, so tear down any server occupying it first.

## Apply

```bash
export EKS_CLUSTER_NAME=<your-cluster-name>
export NAMESPACE=<your-namespace>

kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-jobs/qwen2.5-7b/bf16-model-download.yaml
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait --for=condition=complete job/download-qwen25-7b-bf16 --timeout=6h
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs -f job/download-qwen25-7b-bf16
```

Re-applying is safe. `hf download` skips files already on disk, so a second run over a complete checkpoint costs a pip install and a hub round trip.

`backoffLimit: 0` means a failed Job stays failed rather than restarting, so the logs of the failed attempt remain available. Delete the Job before re-applying it: `kubectl -n "$NAMESPACE" delete job/download-qwen25-7b-bf16`.

## What checks the download

The recipe checks it, not the Job. Every recipe's `wait-for-model` initContainer waits for a complete checkpoint before vLLM starts: it counts the `.safetensors` files against the shard count the index declares, accepts a single-file checkpoint when there is no index, and requires that no `.incomplete` files remain. That check sits immediately before the step that would fail, so the Job does not duplicate it.

This has one consequence worth noting. A Job that exits part way through fails visibly, because `set -e` and `backoffLimit: 0` leave it in `Failed`. But if a download finishes short without erroring, the Job still reports complete and `kubectl wait` returns success; the symptom appears later, when the recipe's pod sits in `Init` instead of serving. For a Job that verifies its own output, [`job-download-weights-to-pvc.yaml`](../k8s/examples/job-download-weights-to-pvc.yaml) is the longer version.

## A bad quantized checkpoint does not announce itself

This is the failure mode worth understanding before running a quantize job.

A checkpoint can carry a `quantization_config` announcing FP8 while its weights are still BF16. vLLM then reads high-precision bytes as FP8 and serves fluent-looking nonsense rather than refusing to start. Nothing upstream fails: the Job exits 0, the checkpoint lands on the shared claim, and every recipe pointing at that path inherits it. An accuracy gate captured against the same bad checkpoint agrees with itself, so benchmark numbers can be published from it.

Three routes produce one:

- `MODELOPT_REAL_EXPORT=0` selects `export_legacy`, which is invalid by design and exists only so an older flow stays reproducible.
- `CALIBRATE=0` skips the calibration pass, so the static `input_scale` tensors are never populated. Static-activation quantization without them is invalid.
- A quantizer image older than `app/quantize/src/` runs its own copy of `quantize.py`, which may take the legacy path whatever this repository says.

## Two checks, one inside the producer and one outside it

`quantize.py` verifies its own output and exits non-zero when the checkpoint does not match the format it was asked for, so a bad export fails the Job instead of landing on the claim. Reproducing the legacy export therefore needs `ALLOW_INVALID_EXPORT=1`, which is deliberately awkward and must never feed a benchmark.

That check cannot cover the stale-image case, because a stale image runs its own `quantize.py` and its own verifier or none. A `*-model-verify.yaml` Job covers it from outside: `app/quantize/src/verify_checkpoint.py` is stdlib-only, parses safetensors headers directly rather than importing torch or modelopt, and runs on `python:3.12-slim`. It works against a checkpoint produced by any version of anything.

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap checkpoint-verifier \
    --from-file=verify_checkpoint.py=app/quantize/src/verify_checkpoint.py
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-jobs/gemma4-e4b/fp8-model-verify.yaml
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
    --for=condition=complete job/verify-checkpoint --timeout=5m
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs job/verify-checkpoint
```

A failed Job means do not serve that path. The log names each problem rather than only reporting a verdict. The same script runs locally against a checkpoint on disk: `python app/quantize/src/verify_checkpoint.py <dir> --format fp8 --exclude '<patterns>'`.

## Keeping the quantizer image and this repository in agreement

`scripts/push-image.sh` builds the image from `app/quantize/src/`, so a freshly built image agrees with this repository by construction. That is the intended path and needs nothing further.

Where ECR push is unavailable, a quantize Job mounts the tracked source over `/app/src` instead, which is how the Gemma 4 E4B checkpoint was produced:

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap quantizer-src \
    --from-file=app/quantize/src/quantize.py \
    --from-file=app/quantize/src/config.py \
    --from-file=app/quantize/src/fix_amax.py \
    --from-file=app/quantize/src/verify_checkpoint.py
```

That is a workaround, not the design. Drop the `src` volume and its mount once the image is built from current source.

## Recipes that stage their own weights

Some recipes fetch weights inside the serving pod and are intentionally not represented here.

The Kimi K3 recipes sync from S3 onto node-local NVMe, which exists only on the node that will serve, so a separate Job could stage onto the wrong node.

`glm-5.2/nvfp4-p6-b300.48xlarge.yaml` downloads into an `emptyDir` in a pod that already requests its full GPU allocation, which holds the node from the moment it schedules. On a Capacity Block that is the intent, so splitting the download out would release the node.

### Why they have not been moved to FSx

The first constraint is size. The Kimi checkpoint is about 1454 GiB and the GLM NVFP4 checkpoint about 433 GiB, against `fsx-pvc` claims of 1200Gi. Kimi does not fit on an empty claim, and certainly not on a filesystem already holding other models. Moving it would require provisioning 2 TiB or more.

The second constraint is the availability zone. FSx for Lustre is single-AZ, so a filesystem sized for these checkpoints commits to one zone, and p6-b200 and p6-b300 capacity determines which zone is available. Node-local storage avoids that dependency, which matters most inside a bounded Capacity Block window where provisioning a filesystem first consumes booked GPU time.

There is one case where the move has a clear benefit. `kimi-k3/mxfp4-p6-b200.48xlarge-sts.yaml` fetches the checkpoint twice, once in each of its two pods, because an `emptyDir` is per-pod and those are different nodes. That is double the S3 egress and double the wall clock on every start, where a `ReadWriteMany` filesystem would stage once. If any of these are moved, this is the candidate.

The read cost is not known. A 62 GiB checkpoint loaded from FSx at roughly 490 MB/s in this repo's own measurements, which extrapolates to roughly an hour for 1454 GiB against node-local NVMe. That figure is an extrapolation rather than a measurement, and a decision should be based on a direct measurement instead.

These recipes do keep a pre-flight space check, because an `emptyDir` lands wherever kubelet's root is rather than on the instance store by default, and a 1454 GiB sync onto a root disk fills the node.

## FP8 weights, downloaded or quantized

`/models/quantized/Qwen2.5-7B-Instruct-fp8` can be produced two ways: [`qwen2.5-7b/fp8-model-download.yaml`](qwen2.5-7b/fp8-model-download.yaml) fetches ready-made FP8 weights from Hugging Face, and [`qwen2.5-7b/fp8-model-quantize.yaml`](qwen2.5-7b/fp8-model-quantize.yaml) produces them from the BF16 weights. Use one or the other, not both.

One path has no download Job here because no published FP8 repository matches it, so [`gemma4-31b/fp8-model-quantize.yaml`](gemma4-31b/fp8-model-quantize.yaml) is the only producer of `/models/quantized/gemma-4-31b-it-fp8`. It returns HTTP 401 from the Hugging Face API under the repository names checked, which indicates gated, private, or absent. The API returns 401 rather than 404 because it does not confirm whether a private repository exists.

`Qwen/Qwen2.5-7B-Instruct-FP8` returns 401 the same way, so [`qwen2.5-7b/fp8-model-download.yaml`](qwen2.5-7b/fp8-model-download.yaml) has not been run end to end. Accept the gating on your Hugging Face account, or use a quantize job instead.

## Adding a model

Copy the closest Job into `model-jobs/<model-name>/`, named `<quant>-model-download.yaml` or `<quant>-model-quantize.yaml`, using the same model directory name as `vllm-recipes/` and `docs/runbooks/`.

For a download, everything that needs to change is in `env`: set `MODEL_REPO` to the Hugging Face repository and `MODEL_DIR` to the destination path. `MODEL_DIR` must match the path the recipe polls for, which is the `D=` line in the recipe's `wait-for-model` initContainer.
