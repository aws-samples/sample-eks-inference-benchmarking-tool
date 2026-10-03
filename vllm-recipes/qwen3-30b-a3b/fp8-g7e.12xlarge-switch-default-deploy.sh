#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="$SCRIPT_DIR/fp8-g7e.12xlarge-switch-default.yaml"
if grep -qF 'ip-10-0-0-1.us-east-1.compute.internal' "$MANIFEST"; then
  echo "FATAL: $MANIFEST still pins the placeholder node; substitute GPU_NODE into kubernetes.io/hostname first." >&2
  exit 1
fi
# Deploy the Qwen3-30B-A3B server with NCCL default transport selection.
echo "==> Deploying Qwen3-30B-A3B FP8 on g7e.12xlarge (switch study, NCCL default)"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$MANIFEST"
# Wait for server startup.
echo "==> Waiting for the Qwen3-30B-A3B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/qwen3-30b-a3b-g7e12-switch-default \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/qwen3-30b-a3b-g7e12-switch-default \
  --container vllm \
  -- curl -s localhost:8000/v1/models
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
