#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="kimi-k3-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Create the model's weights bucket if it does not exist.
echo "==> Creating bucket $WEIGHTS_BUCKET"
aws s3api head-bucket --bucket "$WEIGHTS_BUCKET" >/dev/null 2>&1 \
  || aws s3 mb "s3://$WEIGHTS_BUCKET" --region "$AWS_REGION"
# Apply the kimi-k3-bench ServiceAccount the staging Job runs as.
echo "==> Applying ServiceAccount kimi-k3-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount kimi-k3-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Start the Job that streams the published checkpoint from Hugging Face to S3, skipping files already there.
echo "==> Staging the Kimi K3 MXFP4 checkpoint to s3://$WEIGHTS_BUCKET/kimi-k3-mxfp4/"
sed -e "s|value: kimi-k3-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: ${AWS_REGION}|;}" \
  "$SCRIPT_DIR/mxfp4-model-download.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the Job, which fails if any file is missing from S3.
echo "==> Waiting for the staging Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/kimi-k3-mxfp4-hf-to-s3 \
  --timeout=24h
# Confirm the checkpoint's config.json is in S3.
echo "==> Checking s3://$WEIGHTS_BUCKET/kimi-k3-mxfp4/config.json"
aws s3 ls "s3://${WEIGHTS_BUCKET}/kimi-k3-mxfp4/config.json" --region "$AWS_REGION"
