#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${WEIGHTS_BUCKET:?Set WEIGHTS_BUCKET before running this script}"
: "${WEIGHTS_REGION:?Set WEIGHTS_REGION before running this script}"
: "${GPU_NODE:?Set GPU_NODE before running this script}"
: "${S3_SERVICE_ACCOUNT:?Set S3_SERVICE_ACCOUNT before running this script}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN_MLA}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the Mistral Large 3 server with the bucket, region, node, ServiceAccount and attention backend filled in.
echo "==> Deploying Mistral Large 3 FP8 on p5en.48xlarge with $ATTENTION_BACKEND"
sed -e "s|value: SUBSTITUTE-weights-bucket|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${WEIGHTS_REGION}\"|;}" \
  -e "/name: ATTENTION_BACKEND/{n;s|value: .*|value: \"${ATTENTION_BACKEND}\"|;}" \
  -e "s|SUBSTITUTE-gpu-node|${GPU_NODE}|" \
  -e "s|SUBSTITUTE-service-account|${S3_SERVICE_ACCOUNT}|" \
  "$SCRIPT_DIR/fp8-p5en.48xlarge.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Mistral Large 3 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/mistral-large-3-vllm-p5en \
  --timeout=7200s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/mistral-large-3-vllm-p5en \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models \
  | grep '"id":"mistral-large-3"'
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/mistral-large-3-vllm-p5en \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Confirm the server selected the requested attention backend.
echo "==> Confirming the attention backend is $ATTENTION_BACKEND"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/mistral-large-3-vllm-p5en \
  --container vllm \
  | grep "Using AttentionBackendEnum.${ATTENTION_BACKEND} backend"
# Validate the serving manifest.
echo "==> Validating the Mistral Large 3 serving manifest"
eks-bench validate-manifests \
  mistral-large-3/fp8-p5en.48xlarge.yaml
