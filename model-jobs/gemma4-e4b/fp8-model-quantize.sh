#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
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
# Start the quantize Job with the quantizer image registry filled in.
echo "==> Quantizing Gemma 4 E4B to /models/quantized/gemma-4-E4B-it-fp8"
sed -e "s|012345678901.dkr.ecr.us-east-1|${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}|" "$SCRIPT_DIR/fp8-model-quantize.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the Job, which fails if the exported checkpoint does not verify.
echo "==> Waiting for the quantize Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/quantize-gemma4-e4b-fp8 \
  --timeout=30m
# Apply the checkpoint-verifier ConfigMap from this repository's verifier.
echo "==> Applying ConfigMap checkpoint-verifier"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create configmap checkpoint-verifier \
  --from-file=verify_checkpoint.py="$REPO_ROOT/app/quantize/src/verify_checkpoint.py" \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Start the Job that re-verifies the checkpoint outside the quantizer image.
echo "==> Verifying /models/quantized/gemma-4-E4B-it-fp8"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/fp8-model-verify.yaml"
# Wait for the verify Job, which fails and names each problem if the checkpoint is not FP8.
echo "==> Waiting for the verify Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/verify-checkpoint \
  --timeout=5m
