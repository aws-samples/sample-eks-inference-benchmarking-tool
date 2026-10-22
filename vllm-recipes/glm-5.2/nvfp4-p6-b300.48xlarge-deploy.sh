#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the GLM-5.2 server.
echo "==> Deploying GLM-5.2 NVFP4 on p6-b300.48xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/nvfp4-p6-b300.48xlarge.yaml"
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the GLM-5.2 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/glm-52-b300-nvfp4 \
  --timeout=6h
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/glm-52-b300-nvfp4 \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/glm-52-b300-nvfp4 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Validate the serving manifest.
echo "==> Validating the GLM-5.2 serving manifest"
eks-bench validate-manifests \
  glm-5.2/nvfp4-p6-b300.48xlarge.yaml
