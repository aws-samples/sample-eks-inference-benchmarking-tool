#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Deploy the Gemma 4 E4B server. It serves the FP8 checkpoint from the shared claim.
echo "==> Deploying Gemma 4 E4B FP8 on g7e.2xlarge (FP8 KV, TRITON_ATTN)"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
  -f "$SCRIPT_DIR/fp8-g7e.2xlarge-fp8kv-attn-triton.yaml"

# Wait for model download and server startup.
echo "==> Waiting for the Gemma server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/gemma4-e4b-vllm-g7e-2xl-fp8kv-triton \
  --timeout=3600s

# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-e4b-vllm-g7e-2xl-fp8kv-triton \
  --container vllm \
  -- curl -s localhost:8000/v1/models

# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-e4b-vllm-g7e-2xl-fp8kv-triton \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"

# Confirm the pinned attention backend actually took effect. v0.29.0 does not read
# VLLM_ATTENTION_BACKEND, so a pin passed as an environment variable leaves the engine
# on whatever it auto-selects, and the result is labelled with a backend that never ran.
# Fail here rather than let that reach a result file.
echo "==> Confirming the attention backend is TRITON_ATTN"
BACKEND_LOG=$(kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/gemma4-e4b-vllm-g7e-2xl-fp8kv-triton --container vllm)
if grep -q "Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND" <<<"$BACKEND_LOG"; then
  echo "FAIL: server ignored VLLM_ATTENTION_BACKEND; the pin must be --attention-backend" >&2
  exit 1
fi
if ! grep -q "Using AttentionBackendEnum.TRITON_ATTN backend" <<<"$BACKEND_LOG"; then
  echo "FAIL: requested TRITON_ATTN but the server did not report it" >&2
  grep -E "Using AttentionBackendEnum" <<<"$BACKEND_LOG" >&2 || echo "  no backend line in the log" >&2
  exit 1
fi
echo "    backend confirmed: TRITON_ATTN"

# Validate the serving manifest.
echo "==> Validating the Gemma serving manifest"
eks-bench validate-manifests \
  gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-triton.yaml
