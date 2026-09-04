#!/usr/bin/env bash
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Deploy the Gemma 4 31B server.
echo "==> Deploying Gemma 4 31B FP8 on g6e.12xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
  -f "$SCRIPT_DIR/fp8-g6e.12xlarge.yaml"

# Wait for server startup.
echo "==> Waiting for the Gemma server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/gemma4-31b-vllm-g6e-fp8 \
  --timeout=3600s

# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-31b-vllm-g6e-fp8 \
  --container vllm \
  -- curl -s localhost:8000/v1/models

# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-31b-vllm-g6e-fp8 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"

# Validate the serving manifest.
echo "==> Validating the Gemma serving manifest"
python3 app/cli.py validate-manifests \
  gemma4-31b/fp8-g6e.12xlarge.yaml
