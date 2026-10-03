#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="$SCRIPT_DIR/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer.yaml"
if grep -q 'SUBSTITUTE-weights-bucket' "$MANIFEST"; then
  echo "FATAL: $MANIFEST still contains the placeholder weights bucket; set WEIGHTS_BUCKET and AWS_REGION first." >&2
  exit 1
fi
# Deploy the Llama 4 Maverick server. It syncs the FP8 checkpoint from S3.
echo "==> Deploying Llama 4 Maverick FP8 on one p6-b200.48xlarge, TP8 (FP8 KV, FLASHINFER)"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$MANIFEST"
# Wait for the weight sync and server startup.
echo "==> Waiting for the Llama 4 Maverick server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k-flashinfer \
  --timeout=7200s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k-flashinfer \
  --container vllm \
  -- curl -s localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k-flashinfer \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Confirm the pinned attention backend took effect.
echo "==> Confirming the attention backend is FLASHINFER"
BACKEND_LOG=$(kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k-flashinfer --container vllm)
if ! grep -q "Using AttentionBackendEnum.FLASHINFER backend" <<<"$BACKEND_LOG"; then
  echo "FAIL: requested FLASHINFER but the server did not report it" >&2
  exit 1
fi
echo "    backend confirmed: FLASHINFER"
# Validate the serving manifest.
echo "==> Validating the Llama 4 Maverick serving manifest"
eks-bench validate-manifests \
  llama4-maverick/fp8-p6-b200.48xlarge-fp8kv-32k-flashinfer.yaml
