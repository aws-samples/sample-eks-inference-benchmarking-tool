#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the Nemotron 3 Ultra server.
echo "==> Deploying Nemotron 3 Ultra NVFP4 on p6-b200.48xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/nvfp4-p6-b200.48xlarge-fp8kv-32k.yaml"
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Nemotron 3 Ultra server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k \
  --timeout=7200s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is TRITON_ATTN"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k \
  --container vllm \
  | grep "Using AttentionBackendEnum.TRITON_ATTN backend"
# Validate the serving manifest.
echo "==> Validating the Nemotron 3 Ultra serving manifest"
eks-bench validate-manifests \
  nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k.yaml
