#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the Gemma 4 31B server.
echo "==> Deploying Gemma 4 31B FP8 on p6-b200.48xlarge"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/fp8-p6-b200.48xlarge-fp8kv-32k-dp8-textonly-flashinfer.yaml"
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Gemma 4 31B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/gemma4-31b-vllm-b200-fp8kv-32k-dp8-textonly-flashinfer \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-31b-vllm-b200-fp8kv-32k-dp8-textonly-flashinfer \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-31b-vllm-b200-fp8kv-32k-dp8-textonly-flashinfer \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is FLASHINFER"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/gemma4-31b-vllm-b200-fp8kv-32k-dp8-textonly-flashinfer \
  --container vllm \
  | grep "Using AttentionBackendEnum.FLASHINFER backend"
# Validate the serving manifest.
echo "==> Validating the Gemma 4 31B serving manifest"
eks-bench validate-manifests \
  gemma4-31b/fp8-p6-b200.48xlarge-fp8kv-32k-dp8-textonly-flashinfer.yaml
