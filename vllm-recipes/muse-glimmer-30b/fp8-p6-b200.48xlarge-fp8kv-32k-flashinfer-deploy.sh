#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="muse-glimmer-30b-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Apply the muse-glimmer-30b-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount muse-glimmer-30b-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount muse-glimmer-30b-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the Muse-Glimmer 30B server with the weights bucket and its region filled in.
echo "==> Deploying Muse-Glimmer 30B FP8 on p6-b200.48xlarge"
sed -e "s|value: muse-glimmer-30b-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${AWS_REGION}\"|;}" \
  "$SCRIPT_DIR/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Muse-Glimmer 30B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/muse-glimmer-30b-vllm-b200-fp8kv-32k-flashinfer \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/muse-glimmer-30b-vllm-b200-fp8kv-32k-flashinfer \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/muse-glimmer-30b-vllm-b200-fp8kv-32k-flashinfer \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is FLASHINFER"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/muse-glimmer-30b-vllm-b200-fp8kv-32k-flashinfer \
  --container vllm \
  | grep "Using AttentionBackendEnum.FLASHINFER backend"
# Validate the serving manifest.
echo "==> Validating the Muse-Glimmer 30B serving manifest"
eks-bench validate-manifests \
  muse-glimmer-30b/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer.yaml
