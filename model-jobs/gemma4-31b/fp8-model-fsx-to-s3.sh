#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${WEIGHTS_BUCKET:?Set WEIGHTS_BUCKET before running this script}"
: "${WEIGHTS_REGION:?Set WEIGHTS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Start the Job that copies the quantized checkpoint from the claim to S3.
echo "==> Copying the Gemma 4 31B checkpoint to s3://$WEIGHTS_BUCKET/gemma4-31b-fp8-quantized/"
sed -e "s|value: SUBSTITUTE-weights-bucket|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: ${WEIGHTS_REGION}|;}" \
  "$SCRIPT_DIR/fp8-model-fsx-to-s3.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the Job, which fails on a shard-count mismatch.
echo "==> Waiting for the copy Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/gemma4-31b-fp8-fsx-to-s3 \
  --timeout=4h
# Confirm the checkpoint's hf_quant_config.json is in S3.
echo "==> Checking s3://$WEIGHTS_BUCKET/gemma4-31b-fp8-quantized/"
aws s3 ls "s3://${WEIGHTS_BUCKET}/gemma4-31b-fp8-quantized/hf_quant_config.json" --region "$WEIGHTS_REGION"
