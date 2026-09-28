#!/usr/bin/env bash
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEPLOYMENT=gemma4-31b-vllm-g7e-fp8kv-32k

# Deploy the Gemma 4 31B server. It serves the FP8 checkpoint from the shared claim.
echo "==> Deploying Gemma 4 31B FP8 on g7e.4xlarge (FP8 KV, 32k context, TRITON_ATTN)"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
  -f "$SCRIPT_DIR/fp8-g7e.4xlarge-fp8kv-32k.yaml"

# Wait for server startup. A 30 GiB checkpoint off Lustre plus graph capture takes several minutes.
echo "==> Waiting for the Gemma server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  "deployment/$DEPLOYMENT" \
  --timeout=3600s

# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  "deployment/$DEPLOYMENT" \
  --container vllm \
  -- curl -s localhost:8000/v1/models

# Record the live KV-cache allocation, and fail if the KV cache is not FP8. This is the
# recipe's whole point, and a server that silently fell back to BF16 would still serve.
echo "==> Reading the KV-cache configuration"
CACHE_INFO=$(kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  "deployment/$DEPLOYMENT" \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'")
echo "$CACHE_INFO"
if ! grep -q 'cache_dtype="fp8"' <<<"$CACHE_INFO"; then
  echo "FAIL: cache_config_info does not report cache_dtype=\"fp8\"; the KV cache is not FP8" >&2
  exit 1
fi
echo "    KV cache confirmed: fp8"

# Confirm the pinned attention backend actually took effect. v0.29.0 does not read
# VLLM_ATTENTION_BACKEND, so a pin passed as an environment variable leaves the engine on
# whatever it auto-selects. Fail here rather than let that reach a result file.
echo "==> Confirming the attention backend is TRITON_ATTN"
SERVER_LOG=$(kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  "deployment/$DEPLOYMENT" --container vllm)
if grep -q "Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND" <<<"$SERVER_LOG"; then
  echo "FAIL: server ignored VLLM_ATTENTION_BACKEND; the pin must be --attention-backend" >&2
  exit 1
fi
if ! grep -q "Using AttentionBackendEnum.TRITON_ATTN backend" <<<"$SERVER_LOG"; then
  echo "FAIL: requested TRITON_ATTN but the server did not report it" >&2
  grep -E "Using AttentionBackendEnum" <<<"$SERVER_LOG" >&2 || echo "  no backend line in the log" >&2
  exit 1
fi
echo "    backend confirmed: TRITON_ATTN"

# Record the KV pool this context and precision produced, and the uncalibrated-scale caveat.
echo "==> KV pool and scale caveat, from the server log"
grep -E "GPU KV cache size|Maximum concurrency|KV cache scaling factor" <<<"$SERVER_LOG" || true

# Validate the serving manifest.
echo "==> Validating the Gemma serving manifest"
eks-bench validate-manifests \
  gemma4-31b/fp8-g7e.4xlarge-fp8kv-32k.yaml
