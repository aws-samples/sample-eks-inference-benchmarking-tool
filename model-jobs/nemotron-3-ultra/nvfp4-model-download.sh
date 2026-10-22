#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="nemotron-3-ultra-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Create the model's weights bucket if it does not exist.
echo "==> Creating bucket $WEIGHTS_BUCKET"
aws s3api head-bucket --bucket "$WEIGHTS_BUCKET" >/dev/null 2>&1 \
  || aws s3 mb "s3://$WEIGHTS_BUCKET" --region "$AWS_REGION"
# Apply the nemotron-3-ultra-bench ServiceAccount the staging Job runs as.
echo "==> Applying ServiceAccount nemotron-3-ultra-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount nemotron-3-ultra-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Start the Job that stages the published checkpoint from Hugging Face to S3.
echo "==> Staging the Nemotron 3 Ultra NVFP4 checkpoint to s3://$WEIGHTS_BUCKET/nemotron-3-ultra-nvfp4/"
sed -e "s|value: nemotron-3-ultra-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: ${AWS_REGION}|;}" \
  "$SCRIPT_DIR/nvfp4-model-download.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the Job, which fails on a shard-count or size mismatch.
echo "==> Waiting for the staging Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/nemotron-3-ultra-nvfp4-hf-to-s3 \
  --timeout=24h
# Confirm S3 holds all 113 shards.
echo "==> Checking the shard count in S3"
aws s3 ls "s3://${WEIGHTS_BUCKET}/nemotron-3-ultra-nvfp4/" --region "$AWS_REGION" \
  | grep -c '\.safetensors$' | grep -x 113
