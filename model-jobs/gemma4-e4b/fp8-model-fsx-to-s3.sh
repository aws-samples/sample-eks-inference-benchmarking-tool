#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="gemma4-e4b-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Create the model's weights bucket if it does not exist.
echo "==> Creating bucket $WEIGHTS_BUCKET"
aws s3api head-bucket --bucket "$WEIGHTS_BUCKET" >/dev/null 2>&1 \
  || aws s3 mb "s3://$WEIGHTS_BUCKET" --region "$AWS_REGION"
# Apply the gemma4-e4b-bench ServiceAccount the copy Job runs as.
echo "==> Applying ServiceAccount gemma4-e4b-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount gemma4-e4b-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Start the Job that copies the quantized checkpoint from the claim to S3.
echo "==> Copying the Gemma 4 E4B checkpoint to s3://$WEIGHTS_BUCKET/gemma4-e4b-fp8-quantized/"
sed -e "s|value: gemma4-e4b-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: ${AWS_REGION}|;}" \
  "$SCRIPT_DIR/fp8-model-fsx-to-s3.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the Job, which fails on a shard-count mismatch.
echo "==> Waiting for the copy Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/gemma4-e4b-fp8-fsx-to-s3 \
  --timeout=1h
# Confirm the checkpoint's config.json is in S3.
echo "==> Checking s3://$WEIGHTS_BUCKET/gemma4-e4b-fp8-quantized/config.json"
aws s3 ls "s3://${WEIGHTS_BUCKET}/gemma4-e4b-fp8-quantized/config.json" --region "$AWS_REGION"
# Confirm the checkpoint's hf_quant_config.json is in S3.
echo "==> Checking s3://$WEIGHTS_BUCKET/gemma4-e4b-fp8-quantized/hf_quant_config.json"
aws s3 ls "s3://${WEIGHTS_BUCKET}/gemma4-e4b-fp8-quantized/hf_quant_config.json" --region "$AWS_REGION"
# Confirm S3 holds the single shard.
echo "==> Checking the shard count in S3"
aws s3 ls "s3://${WEIGHTS_BUCKET}/gemma4-e4b-fp8-quantized/" --region "$AWS_REGION" \
  | grep -c '\.safetensors$' | grep -x 1
