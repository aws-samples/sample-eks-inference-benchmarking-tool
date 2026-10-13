#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="mistral-large-3-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Apply the mistral-large-3-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount mistral-large-3-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount mistral-large-3-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the Mistral Large 3 server with the weights bucket and its region filled in.
echo "==> Deploying Mistral Large 3 FP8 on p6-b200.48xlarge"
sed -e "s|value: mistral-large-3-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${AWS_REGION}\"|;}" \
  "$SCRIPT_DIR/fp8-p6-b200.48xlarge.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Mistral Large 3 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/mistral-large-3-vllm-b200 \
  --timeout=7200s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/mistral-large-3-vllm-b200 \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/mistral-large-3-vllm-b200 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is FLASHINFER_MLA"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/mistral-large-3-vllm-b200 \
  --container vllm \
  | grep "Using AttentionBackendEnum.FLASHINFER_MLA backend"
# Validate the serving manifest.
echo "==> Validating the Mistral Large 3 serving manifest"
eks-bench validate-manifests \
  mistral-large-3/fp8-p6-b200.48xlarge.yaml
