#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${WEIGHTS_BUCKET:?Set WEIGHTS_BUCKET before running this script}"
: "${WEIGHTS_REGION:?Set WEIGHTS_REGION before running this script}"
: "${GPU_NODE:?Set GPU_NODE before running this script}"
: "${S3_SERVICE_ACCOUNT:?Set S3_SERVICE_ACCOUNT before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Remove any earlier run of the Job, so a finished one cannot pass this script without staging.
echo "==> Removing any earlier staging Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" delete job mistral-large-3-fp8-hf-to-s3 --ignore-not-found
# Start the Job that stages the published checkpoint from Hugging Face to S3.
echo "==> Staging the Mistral Large 3 FP8 checkpoint to s3://$WEIGHTS_BUCKET/mistral-large-3-fp8/"
sed -e "s|value: SUBSTITUTE-weights-bucket|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: ${WEIGHTS_REGION}|;}" \
  -e "s|SUBSTITUTE-gpu-node|${GPU_NODE}|" \
  -e "s|SUBSTITUTE-service-account|${S3_SERVICE_ACCOUNT}|" \
  "$SCRIPT_DIR/fp8-model-download.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the Job, which fails on a shard-count or size mismatch.
echo "==> Waiting for the staging Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/mistral-large-3-fp8-hf-to-s3 \
  --timeout=24h
# Confirm S3 holds all 272 shards.
echo "==> Checking the shard count in S3"
aws s3 ls "s3://${WEIGHTS_BUCKET}/mistral-large-3-fp8/" --region "$WEIGHTS_REGION" \
  | awk '{print $NF}' | grep -c '^consolidated-.*\.safetensors$' | grep -x 272
