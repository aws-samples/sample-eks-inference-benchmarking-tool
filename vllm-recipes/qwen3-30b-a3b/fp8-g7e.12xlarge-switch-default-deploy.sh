#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the Qwen3-30B-A3B server.
echo "==> Deploying Qwen3-30B-A3B FP8 on g7e.12xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/fp8-g7e.12xlarge-switch-default.yaml"
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Qwen3-30B-A3B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/qwen3-30b-a3b-g7e12-switch-default \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/qwen3-30b-a3b-g7e12-switch-default \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/qwen3-30b-a3b-g7e12-switch-default \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Validate the serving manifest.
echo "==> Validating the Qwen3-30B-A3B serving manifest"
eks-bench validate-manifests \
  qwen3-30b-a3b/fp8-g7e.12xlarge-switch-default.yaml
