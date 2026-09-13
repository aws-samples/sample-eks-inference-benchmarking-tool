# Model quantization

Jobs that produce a quantized checkpoint on a shared PVC. Apply one of these before the recipe that needs it, once per filesystem.

These are the counterpart to [`model-downloads/`](../model-downloads/README.md). That directory stages checkpoints that are already published; these produce one that is not, so it has no repository id and a recipe can only serve it by path.

## Contents

| Job | Source weights | Lands on the PVC at | Recipes waiting on it |
|---|---|---|---|
| [`gemma4-e4b-fp8.yaml`](gemma4-e4b-fp8.yaml) | `google/gemma-4-E4B-it` | `/models/quantized/gemma-4-E4B-it-fp8` | 4 |

| Utility | Purpose |
|---|---|
| [`verify-checkpoint.yaml`](verify-checkpoint.yaml) | Check any quantized checkpoint on the claim, and fail if it is not what its config claims |

The generic template these are derived from is [`k8s/examples/job-quantize-model.yaml`](../k8s/examples/job-quantize-model.yaml), and [its notes](../k8s/examples/README.md#job-quantize-modelyaml) cover the shared workflow: exporting the runbook's values, confirming the claim is bound, and building the quantizer image. The Jobs here record the per-model values that a runbook would otherwise only describe in prose.

## Prerequisites

A PVC to write to, named `fsx-pvc` in every Job here, and a `hf-token` Secret in the same namespace. Both are covered by the generic notes above.

A GPU free on the pinned instance type. These Jobs request one, so tear down any server occupying it first.

## A bad checkpoint does not announce itself

This is the failure mode worth understanding before running anything here.

A checkpoint can carry a `quantization_config` announcing FP8 while its weights are still BF16. vLLM then reads high-precision bytes as FP8 and serves fluent-looking nonsense rather than refusing to start. Nothing upstream fails: the quantize Job exits 0, the checkpoint lands on the shared claim, and every recipe pointing at that path inherits it. An accuracy gate captured against the same bad checkpoint agrees with itself, so benchmark numbers can be published from it.

Three routes produce one:

- `MODELOPT_REAL_EXPORT=0` selects `export_legacy`, which is invalid by design and exists only so an older flow stays reproducible.
- `CALIBRATE=0` skips the calibration pass, so the static `input_scale` tensors are never populated. Static-activation quantization without them is invalid.
- A quantizer image older than `app/quantize/src/` runs its own copy of `quantize.py`, which may take the legacy path whatever this repository says.

## Two checks, one inside the producer and one outside it

`quantize.py` verifies its own output and exits non-zero when the checkpoint does not match the format it was asked for, so a bad export fails the Job instead of landing on the claim. Reproducing the legacy export therefore needs `ALLOW_INVALID_EXPORT=1`, which is deliberately awkward and must never feed a benchmark.

That check cannot cover the stale-image case, because a stale image runs its own `quantize.py` and its own verifier or none. [`verify-checkpoint.yaml`](verify-checkpoint.yaml) covers it from outside: `app/quantize/src/verify_checkpoint.py` is stdlib-only, parses safetensors headers directly rather than importing torch or modelopt, and runs on `python:3.12-slim`. It works against a checkpoint produced by any version of anything.

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap checkpoint-verifier \
    --from-file=verify_checkpoint.py=app/quantize/src/verify_checkpoint.py
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-quantize/verify-checkpoint.yaml
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
    --for=condition=complete job/verify-checkpoint --timeout=5m
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs job/verify-checkpoint
```

A failed Job means do not serve that path. The log names each problem rather than only reporting a verdict. The same script runs locally against a checkpoint on disk: `python app/quantize/src/verify_checkpoint.py <dir> --format fp8 --exclude '<patterns>'`.

## Keeping the image and this repository in agreement

`scripts/push-image.sh` builds the image from `app/quantize/src/`, so a freshly built image agrees with this repository by construction. That is the intended path and needs nothing further.

Where ECR push is unavailable, these Jobs mount the tracked source over `/app/src` instead, which is how the Gemma 4 E4B checkpoint was produced:

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap quantizer-src \
    --from-file=app/quantize/src/quantize.py \
    --from-file=app/quantize/src/config.py \
    --from-file=app/quantize/src/fix_amax.py \
    --from-file=app/quantize/src/verify_checkpoint.py
```

That is a workaround, not the design. Drop the `src` volume and its mount once the image is built from current source.
