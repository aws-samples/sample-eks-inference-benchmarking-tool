#!/usr/bin/env bash
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this helper}"
: "${NAMESPACE:?Set NAMESPACE before running this helper}"

# Deploy the GLM-5.2 FP8 server.
echo "==> Deploying GLM-5.2 FP8 on p6-b200.48xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
  -f vllm-recipes/glm-5.2/fp8-p6-b200.48xlarge.yaml

# Wait for the server to finish loading the FSx-hosted checkpoint.
echo "==> Waiting for the GLM-5.2 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/glm-52-b200 \
  --timeout=3600s

# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/glm-52-b200 \
  --container vllm \
  -- curl -s localhost:8000/v1/models

# Print the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/glm-52-b200 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"

# Validate the serving manifest.
echo "==> Validating the GLM-5.2 serving manifest"
eks-bench validate-manifests \
  glm-5.2/fp8-p6-b200.48xlarge.yaml
