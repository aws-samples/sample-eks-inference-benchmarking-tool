#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="$SCRIPT_DIR/nvfp4-p6-b200.48xlarge-fp8kv-32k.yaml"
if grep -q 'SUBSTITUTE-weights-bucket' "$MANIFEST"; then
  echo "FATAL: $MANIFEST still contains the placeholder weights bucket; set WEIGHTS_BUCKET and AWS_REGION first." >&2
  exit 1
fi
# Deploy the Nemotron 3 Ultra server. It syncs the NVFP4 checkpoint from S3.
echo "==> Deploying Nemotron 3 Ultra NVFP4 on one p6-b200.48xlarge, TP8 (FP8 KV, TRITON_ATTN)"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$MANIFEST"
# Wait for the weight sync and server startup.
echo "==> Waiting for the Nemotron 3 Ultra server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k \
  --timeout=7200s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k \
  --container vllm \
  -- curl -s localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Confirm the pinned attention backend took effect.
echo "==> Confirming the attention backend is TRITON_ATTN"
BACKEND_LOG=$(kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k --container vllm)
if ! grep -q "Using AttentionBackendEnum.TRITON_ATTN backend" <<<"$BACKEND_LOG"; then
  echo "FAIL: requested TRITON_ATTN but the server did not report it" >&2
  exit 1
fi
echo "    backend confirmed: TRITON_ATTN"
# Validate the serving manifest.
echo "==> Validating the Nemotron 3 Ultra serving manifest"
eks-bench validate-manifests \
  nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k.yaml
