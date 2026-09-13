# Model quantization

Jobs that produce a quantized checkpoint on a shared PVC. Apply one of these before the recipe that needs it, once per filesystem.

These are the counterpart to [`model-downloads/`](../model-downloads/README.md). That directory stages checkpoints that are already published; these produce one that is not, so it has no repository id and a recipe can only serve it by path.

## Contents

| Job | Source weights | Lands on the PVC at | Recipes waiting on it |
|---|---|---|---|
| [`gemma4-e4b-fp8.yaml`](gemma4-e4b-fp8.yaml) | `google/gemma-4-E4B-it` | `/models/quantized/gemma-4-E4B-it-fp8` | 4 |

The generic template these are derived from is [`k8s/examples/job-quantize-model.yaml`](../k8s/examples/job-quantize-model.yaml), and [its notes](../k8s/examples/README.md#job-quantize-modelyaml) cover the shared workflow: exporting the runbook's values, confirming the claim is bound, and building the quantizer image. The Jobs here record the per-model values that a runbook would otherwise only describe in prose.

## Prerequisites

A PVC to write to, named `fsx-pvc` in every Job here, and a `hf-token` Secret in the same namespace. Both are covered by the generic notes above.

A GPU free on the pinned instance type. These Jobs request one, so tear down any server occupying it first.

## The quantizer image must match this repo's source

`scripts/push-image.sh` builds the image from `app/quantize/src/`, so a freshly built image and this repository agree by construction. That is the intended path and needs nothing further.

An image built earlier can disagree, and the failure is quiet rather than loud: the legacy export path leaves weights in BF16 while labelling the checkpoint as FP8, which makes vLLM read BF16 bytes as FP8 and emit garbage instead of failing. `app/quantize/src/quantize.py` documents why that path exists.

Rebuilding is the fix. Where ECR push is not available, these Jobs mount the tracked source over `/app/src` instead, which is how the Gemma 4 E4B checkpoint was produced:

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap quantizer-src \
    --from-file=app/quantize/src/quantize.py \
    --from-file=app/quantize/src/config.py \
    --from-file=app/quantize/src/fix_amax.py
```

Drop the `src` volume and its mount when the image is built from current source.

## Verify before serving

Confirm the export took the real path rather than the legacy one. All three markers must appear in the Job log: one `Excluding from quantization` line per pattern, `calibration N/16 done` for each prompt, and `Exporting REAL FP8 checkpoint via export_hf_checkpoint`. A log carrying `Renamed and scaled ..._amax` instead is the legacy path.

A real export is roughly three quarters of the source size and carries `hf_quant_config.json`. A checkpoint the same size as the source, or missing that file, is invalid.
