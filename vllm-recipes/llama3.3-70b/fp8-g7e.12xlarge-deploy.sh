#!/usr/bin/env bash
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this helper}"
: "${NAMESPACE:?Set NAMESPACE before running this helper}"

# Deploy the Llama 3.3 70B server.
echo "==> Deploying Llama 3.3 70B"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
  -f vllm-recipes/llama3.3-70b/fp8-g7e.12xlarge.yaml

# Wait for model download and server startup.
echo "==> Waiting for the Llama server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/llama33-70b-vllm-g7e-fp8 \
  --timeout=3600s

# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/llama33-70b-vllm-g7e-fp8 \
  --container vllm \
  -- curl -s localhost:8000/v1/models

# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/llama33-70b-vllm-g7e-fp8 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"

# Validate the serving manifest.
echo "==> Validating the Llama serving manifest"
eks-bench validate-manifests \
  llama3.3-70b/fp8-g7e.12xlarge.yaml
