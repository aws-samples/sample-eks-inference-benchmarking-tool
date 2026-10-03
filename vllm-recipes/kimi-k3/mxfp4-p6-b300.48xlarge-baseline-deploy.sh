#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the Kimi K3 server.
echo "==> Deploying Kimi K3 MXFP4 on p6-b300.48xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/mxfp4-p6-b300.48xlarge-baseline.yaml"
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Kimi K3 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/kimi-k3-b300 \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/kimi-k3-b300 \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/kimi-k3-b300 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Validate the serving manifest.
echo "==> Validating the Kimi K3 serving manifest"
eks-bench validate-manifests \
  kimi-k3/mxfp4-p6-b300.48xlarge-baseline.yaml
