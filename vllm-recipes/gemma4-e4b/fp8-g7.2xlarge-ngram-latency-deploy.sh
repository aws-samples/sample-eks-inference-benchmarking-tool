#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Deploy the Gemma 4 E4B server. It serves the FP8 checkpoint from fsx-pvc.
echo "==> Deploying Gemma 4 E4B FP8 on g7.2xlarge (n-gram latency config)"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
  -f "$SCRIPT_DIR/fp8-g7.2xlarge-ngram-latency.yaml"

# Wait for server startup.
echo "==> Waiting for the Gemma server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/gemma4-e4b-vllm-g7-2xl-fp8-ngram-latency \
  --timeout=3600s

# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-e4b-vllm-g7-2xl-fp8-ngram-latency \
  --container vllm \
  -- curl -s localhost:8000/v1/models

# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-e4b-vllm-g7-2xl-fp8-ngram-latency \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"

# Confirm speculation is actually active, since a silently ignored config reads as a null result.
echo "==> Confirming speculative decoding is live"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-e4b-vllm-g7-2xl-fp8-ngram-latency \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep -c 'vllm:spec_decode' \
      || (echo 'FATAL: no vllm:spec_decode metrics; speculation is not enabled' >&2; exit 1)"

# Validate the serving manifest.
echo "==> Validating the Gemma serving manifest"
eks-bench validate-manifests \
  gemma4-e4b/fp8-g7.2xlarge-ngram-latency.yaml
