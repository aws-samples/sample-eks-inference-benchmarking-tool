#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${WEIGHTS_BUCKET:?Set WEIGHTS_BUCKET before running this script}"
: "${WEIGHTS_REGION:?Set WEIGHTS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Apply the gemma4-31b-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount gemma4-31b-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount gemma4-31b-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the Gemma 4 E4B server with the weights bucket and its region filled in.
echo "==> Deploying Gemma 4 E4B FP8 on p5.4xlarge"
sed -e "s|value: SUBSTITUTE-weights-bucket|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${WEIGHTS_REGION}\"|;}" \
  "$SCRIPT_DIR/fp8-p5.4xlarge-fp8kv-attn-triton.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Gemma 4 E4B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is TRITON_ATTN"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
  --container vllm \
  | grep "Using AttentionBackendEnum.TRITON_ATTN backend"
# Validate the serving manifest.
echo "==> Validating the Gemma 4 E4B serving manifest"
eks-bench validate-manifests \
  gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton.yaml
