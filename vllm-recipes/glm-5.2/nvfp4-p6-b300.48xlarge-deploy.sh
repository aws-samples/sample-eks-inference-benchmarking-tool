#!/usr/bin/env bash
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this helper}"
: "${NAMESPACE:?Set NAMESPACE before running this helper}"

# Deploy the GLM-5.2 NVFP4 server.
echo "==> Deploying GLM-5.2 NVFP4 on p6-b300.48xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
  -f vllm-recipes/glm-5.2/nvfp4-p6-b300.48xlarge.yaml

# Wait for the pod-local checkpoint download and server startup.
echo "==> Waiting for the GLM-5.2 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/glm-52-b300-nvfp4 \
  --timeout=6h

# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/glm-52-b300-nvfp4 \
  --container vllm \
  -- curl -s localhost:8000/v1/models

# Print the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/glm-52-b300-nvfp4 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"

# Validate the serving manifest.
echo "==> Validating the GLM-5.2 serving manifest"
python3 app/cli.py validate-manifests \
  glm-5.2/nvfp4-p6-b300.48xlarge.yaml
