#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the Muse-Glimmer 30B server.
echo "==> Deploying Muse-Glimmer 30B FP8 on g7e.48xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml"
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Muse-Glimmer 30B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is TRITON_ATTN"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
  --container vllm \
  | grep "Using AttentionBackendEnum.TRITON_ATTN backend"
# Validate the serving manifest.
echo "==> Validating the Muse-Glimmer 30B serving manifest"
eks-bench validate-manifests \
  muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml
