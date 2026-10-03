#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${WEIGHTS_BUCKET:?Set WEIGHTS_BUCKET before running this script}"
: "${WEIGHTS_REGION:?Set WEIGHTS_REGION before running this script}"
: "${LLAMA4_S3_ROLE_ARN:?Set LLAMA4_S3_ROLE_ARN before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Apply the llama4-maverick-bench ServiceAccount the staging Job runs as, annotated with its IRSA role.
echo "==> Applying ServiceAccount llama4-maverick-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount llama4-maverick-bench \
  --dry-run=client -o yaml \
  | kubectl annotate --local -f - eks.amazonaws.com/role-arn="$LLAMA4_S3_ROLE_ARN" -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Start the Job that stages the published checkpoint from Hugging Face to S3.
echo "==> Staging the Llama 4 Maverick FP8 checkpoint to s3://$WEIGHTS_BUCKET/llama4-maverick-fp8/"
sed -e "s|value: SUBSTITUTE-weights-bucket|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: ${WEIGHTS_REGION}|;}" \
  "$SCRIPT_DIR/fp8-model-download.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the Job, which fails on a shard-count or size mismatch.
echo "==> Waiting for the staging Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/llama4-maverick-fp8-hf-to-s3 \
  --timeout=24h
# Confirm S3 holds all 84 shards.
echo "==> Checking the shard count in S3"
aws s3 ls "s3://${WEIGHTS_BUCKET}/llama4-maverick-fp8/" --region "$WEIGHTS_REGION" \
  | grep -c '\.safetensors$' | grep -x 84
