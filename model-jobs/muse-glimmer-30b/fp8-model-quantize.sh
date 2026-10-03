#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
# Confirm the fsx-pvc claim is bound.
echo "==> Checking claim fsx-pvc"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=jsonpath='{.status.phase}'=Bound pvc/fsx-pvc \
  --timeout=2m
# Apply the quantizer-src ConfigMap from this repository's quantizer source.
echo "==> Applying ConfigMap quantizer-src"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create configmap quantizer-src \
  --from-file="$REPO_ROOT/app/quantize/src/quantize.py" \
  --from-file="$REPO_ROOT/app/quantize/src/config.py" \
  --from-file="$REPO_ROOT/app/quantize/src/fix_amax.py" \
  --from-file="$REPO_ROOT/app/quantize/src/verify_checkpoint.py" \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Start the quantize Job.
echo "==> Quantizing Muse-Glimmer-30B to /models/quantized/Muse-Glimmer-30B-fp8"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/fp8-model-quantize.yaml"
# Wait for the Job, which fails if the exported checkpoint does not verify.
echo "==> Waiting for the quantize Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/quantize-muse-glimmer-30b-fp8 \
  --timeout=3h
