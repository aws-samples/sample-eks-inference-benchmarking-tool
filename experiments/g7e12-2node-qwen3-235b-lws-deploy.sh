#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the Qwen3-235B server.
echo "==> Deploying Qwen3-235B FP8 on two g7e.12xlarge nodes"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/g7e12-2node-qwen3-235b-lws.yaml"
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Qwen3-235B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=Available leaderworkerset/qwen3-235b-lws \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  pod/qwen3-235b-lws-0 \
  --container vllm-leader \
  -- curl -s localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  pod/qwen3-235b-lws-0 \
  --container vllm-leader \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
