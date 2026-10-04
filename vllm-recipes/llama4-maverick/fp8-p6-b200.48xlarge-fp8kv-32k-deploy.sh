#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${WEIGHTS_BUCKET:?Set WEIGHTS_BUCKET before running this script}"
: "${WEIGHTS_REGION:?Set WEIGHTS_REGION before running this script}"
: "${LLAMA4_S3_ROLE_ARN:?Set LLAMA4_S3_ROLE_ARN before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Apply the llama4-maverick-bench ServiceAccount the server runs as, annotated with its IRSA role.
echo "==> Applying ServiceAccount llama4-maverick-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount llama4-maverick-bench \
  --dry-run=client -o yaml \
  | kubectl annotate --local -f - eks.amazonaws.com/role-arn="$LLAMA4_S3_ROLE_ARN" -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the Llama 4 Maverick server with the weights bucket and its region filled in.
echo "==> Deploying Llama 4 Maverick FP8 on p6-b200.48xlarge"
sed -e "s|value: SUBSTITUTE-weights-bucket|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${WEIGHTS_REGION}\"|;}" \
  "$SCRIPT_DIR/fp8-p6-b200.48xlarge-fp8kv-32k.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Llama 4 Maverick server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k \
  --timeout=7200s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is TRITON_ATTN"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/llama4-maverick-vllm-b200-fp8kv-32k \
  --container vllm \
  | grep "Using AttentionBackendEnum.TRITON_ATTN backend"
# Validate the serving manifest.
echo "==> Validating the Llama 4 Maverick serving manifest"
eks-bench validate-manifests \
  llama4-maverick/fp8-p6-b200.48xlarge-fp8kv-32k.yaml
