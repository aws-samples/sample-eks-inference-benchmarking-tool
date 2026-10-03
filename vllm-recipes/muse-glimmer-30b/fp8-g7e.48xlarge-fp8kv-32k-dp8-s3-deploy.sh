#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="$SCRIPT_DIR/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml"
if grep -q 'SUBSTITUTE-weights-bucket' "$MANIFEST"; then
  echo "FATAL: $MANIFEST still contains the placeholder weights bucket; set WEIGHTS_BUCKET and AWS_REGION first." >&2
  exit 1
fi
# Deploy the Muse-Glimmer-30B server. It syncs the FP8 checkpoint from S3.
echo "==> Deploying Muse-Glimmer-30B FP8 on g7e.48xlarge (8 replicas, FP8 KV, TRITON_ATTN)"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$MANIFEST"
# Wait for the weight sync and server startup.
echo "==> Waiting for the Muse-Glimmer-30B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
  --container vllm \
  -- curl -s localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Confirm the pinned attention backend took effect.
echo "==> Confirming the attention backend is TRITON_ATTN"
BACKEND_LOG=$(kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 --container vllm)
if ! grep -q "Using AttentionBackendEnum.TRITON_ATTN backend" <<<"$BACKEND_LOG"; then
  echo "FAIL: requested TRITON_ATTN but the server did not report it" >&2
  exit 1
fi
echo "    backend confirmed: TRITON_ATTN"
# Validate the serving manifest.
echo "==> Validating the Muse-Glimmer-30B serving manifest"
eks-bench validate-manifests \
  muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml
