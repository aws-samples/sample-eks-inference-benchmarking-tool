#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the Llama 4 Maverick server.
echo "==> Deploying Llama 4 Maverick FP8 on p6-b200.48xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/fp8-p6-b200.48xlarge-fp8kv-32k-flashattn.yaml"
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Llama 4 Maverick server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k-flashattn \
  --timeout=7200s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k-flashattn \
  --container vllm \
  -- curl -s localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k-flashattn \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is FLASH_ATTN"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k-flashattn \
  --container vllm \
  | grep "Using AttentionBackendEnum.FLASH_ATTN backend"
# Validate the serving manifest.
echo "==> Validating the Llama 4 Maverick serving manifest"
eks-bench validate-manifests \
  llama4-maverick/fp8-p6-b200.48xlarge-fp8kv-32k-flashattn.yaml
