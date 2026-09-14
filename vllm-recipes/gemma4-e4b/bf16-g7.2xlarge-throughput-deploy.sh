#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Deploy the Gemma 4 E4B server. Its init container stages weights in emptyDir.
echo "==> Deploying Gemma 4 E4B BF16 on g7.2xlarge (throughput config)"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
  -f "$SCRIPT_DIR/bf16-g7.2xlarge-throughput.yaml"

# Wait for model download and server startup.
echo "==> Waiting for the Gemma server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/gemma4-e4b-vllm-g7-2xl-throughput \
  --timeout=3600s

# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-e4b-vllm-g7-2xl-throughput \
  --container vllm \
  -- curl -s localhost:8000/v1/models

# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-e4b-vllm-g7-2xl-throughput \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"

# Validate the serving manifest.
echo "==> Validating the Gemma serving manifest"
eks-bench validate-manifests \
  gemma4-e4b/bf16-g7.2xlarge-throughput.yaml
