# Model downloads

Jobs that put model weights on a shared PVC. Apply one of these before the recipe that needs it, once per filesystem.

Recipes do not download their own weights. Each one waits for a complete checkpoint to appear and starts serving when it does, so staging is a separate step you run first and reuse across every recipe that shares a model.

## Contents

| Job | Weights | Lands on the PVC at | Recipes waiting on it |
|---|---|---|---|
| [`gemma4-31b-bf16.yaml`](gemma4-31b-bf16.yaml) | `google/gemma-4-31b-it` | `/models/gemma-4-31b-it` | 3 |
| [`glm-5.2-fp8.yaml`](glm-5.2-fp8.yaml) | `zai-org/GLM-5.2-FP8` | `/models/glm-5.2-fp8` | 1 |
| [`qwen2.5-7b-bf16.yaml`](qwen2.5-7b-bf16.yaml) | `Qwen/Qwen2.5-7B-Instruct` | `/models/Qwen2.5-7B-Instruct` | 4 |
| [`qwen2.5-7b-fp8.yaml`](qwen2.5-7b-fp8.yaml) | `Qwen/Qwen2.5-7B-Instruct-FP8` | `/models/quantized/Qwen2.5-7B-Instruct-fp8` | 5 |
| [`qwen3-235b-fp8.yaml`](qwen3-235b-fp8.yaml) | `Qwen/Qwen3-235B-A22B-FP8` | `/models/qwen3-235b-a22b-fp8` | 1 |

Each Job is three lines of shell, and everything that varies is in `env`. This follows NVIDIA's [Dynamo GLM recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.2/model-cache/model-download.yaml), including the pinned hub client and `HF_XET_HIGH_PERFORMANCE`, because Xet is the accelerated transfer path now that `hf_transfer` has been removed from the client.

The one deliberate difference is `--local-dir`. Dynamo lets the weights sit in the hub cache and serves them by repository id. Our recipes serve a path, and some of those paths are `/models/quantized/...` checkpoints produced by the quantize job, which have no repository id to resolve. So the Job writes to an explicit directory that a recipe's poller can wait on.

## Before you apply one

A PVC to write to, named `fsx-pvc` in every Job here. See [`persistentvolume-fsx-lustre.yaml`](../k8s/examples/persistentvolume-fsx-lustre.yaml), and change `claimName` if yours is named differently.

A `hf-token` Secret in the same namespace, since Secrets are namespaced. Covered by Quickstart step 4 in the [root README](../README.md).

Enough free space for the whole model. Nothing checks this for you, so `kubectl -n "$NAMESPACE" exec` into anything with the PVC mounted and run `df -h /models` first if you are unsure.

## Apply

```bash
export EKS_CLUSTER_NAME=<your-cluster-name>
export NAMESPACE=<your-namespace>

kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-downloads/qwen2.5-7b-bf16.yaml
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait --for=condition=complete job/download-qwen25-7b-bf16 --timeout=6h
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs -f job/download-qwen25-7b-bf16
```

Re-applying is safe. `hf download` skips files already on disk, so a second run over a complete checkpoint costs a pip install and a hub round trip.

`backoffLimit: 0` means a failed Job stays failed rather than restarting itself, so the logs of the attempt that failed are still there when you look. Delete the Job before re-applying it: `kubectl -n "$NAMESPACE" delete job/download-qwen25-7b-bf16`.

## What checks the download

The recipe does, not the Job. Every recipe's `wait-for-model` initContainer waits for a complete checkpoint before vLLM starts: it counts the `.safetensors` files against the shard count the index declares, accepts a single-file checkpoint when there is no index, and requires that no `.incomplete` files remain. That gate sits immediately in front of the thing that would break, so the Job does not repeat it.

The consequence is worth knowing. A Job that dies part way through fails visibly, because `set -e` and `backoffLimit: 0` leave it in `Failed`. But if a download somehow finishes short without erroring, the Job still reports complete and `kubectl wait` returns success; you find out when the recipe's pod sits in `Init` instead of serving. If you want the Job itself to verify, [`job-download-weights-to-pvc.yaml`](../k8s/examples/job-download-weights-to-pvc.yaml) is the longer version that does.

## Recipes that stage their own weights

Seven recipes fetch weights inside the serving pod, and they are deliberately not here.

The Kimi K3 recipes sync from S3 onto node-local NVMe, which only exists on the node that will serve, so a separate Job could stage onto the wrong node.

`glm-5.2-nvfp4-p6-b300.48xlarge.yaml` downloads into an `emptyDir` in a pod that already requests its full GPU allocation, which holds the node from the moment it schedules. On a Capacity Block that is the point, so splitting the download out would give the node up.

### Why they have not been moved to FSx

The first blocker is size. The Kimi checkpoint is about 1454 GiB and the GLM NVFP4 checkpoint about 433 GiB, against `fsx-pvc` claims of 1200Gi. Kimi does not fit on an empty one, let alone a filesystem already holding other models. Moving it means provisioning 2 TiB or more.

The second blocker is the availability zone. FSx for Lustre is single-AZ, so a filesystem sized for these checkpoints commits you to one zone, and p6-b200 and p6-b300 capacity decides which zone you actually get. Node-local storage sidesteps that decision, which matters most inside a bounded Capacity Block window where provisioning a filesystem first costs booked GPU time.

There is one case where the move clearly pays. `kimi-k3-mxfp4-p6-b200.48xlarge-lws.yaml` fetches the checkpoint twice, once in the leader and once in the worker, because an `emptyDir` is per-pod and those are different nodes. That is double the S3 egress and double the wall clock on every start, where a `ReadWriteMany` filesystem would stage once. If any of these moves, it should be that one.

What is not known is the read cost. A 62 GiB checkpoint loaded from FSx at roughly 490 MB/s in this repo's own measurements, which would extrapolate to something like an hour for 1454 GiB against node-local NVMe. That is an extrapolation, not a measurement, and the decision needs a real one before anyone acts on it.

Those seven do keep a pre-flight space check, because an `emptyDir` lands wherever kubelet's root is rather than on the instance store by default, and a 1454 GiB sync onto a root disk fills the node.

## FP8 weights, downloaded or quantized

`/models/quantized/Qwen2.5-7B-Instruct-fp8` can be produced two ways: [`qwen2.5-7b-fp8.yaml`](qwen2.5-7b-fp8.yaml) fetches ready-made FP8 weights from Hugging Face, and [`job-quantize-model.yaml`](../k8s/examples/job-quantize-model.yaml) produces them from the BF16 weights. Use one or the other, not both.

Two paths have no Job here because no published FP8 repository matches them, so the quantize job is the only producer: `/models/quantized/gemma-4-31b-it-fp8` and `/models/quantized/Qwen2.5-32B-Instruct-fp8`. Both return HTTP 401 from the Hugging Face API under every obvious repository name, which means gated, private or absent, and 401 rather than 404 because the API does not confirm whether a private repository exists.

`Qwen/Qwen2.5-7B-Instruct-FP8` returns 401 the same way, so [`qwen2.5-7b-fp8.yaml`](qwen2.5-7b-fp8.yaml) has never been run end to end. Accept the gating on your Hugging Face account or use the quantize job instead.

One recipe expects a path no Job can fill. `glm-5.2-fp8-p5en.48xlarge.yaml` reads `/weights/glm-5.2-fp8` from a `hostPath` at `/mnt/glm-weights` rather than a PVC, so the weights have to be on that node already. The p6-b200 variant of the same model mounts the PVC and is covered by [`glm-5.2-fp8.yaml`](glm-5.2-fp8.yaml).

## Adding a model

Copy the closest Job here. Everything you need to change is in `env`: set `MODEL_REPO` to the Hugging Face repository and `MODEL_DIR` to where it should land. `MODEL_DIR` has to match the path the recipe polls for, which is the `D=` line in the recipe's `wait-for-model` initContainer.
