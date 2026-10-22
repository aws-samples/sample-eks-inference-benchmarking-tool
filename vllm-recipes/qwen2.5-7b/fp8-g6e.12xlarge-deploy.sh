#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Deploy the Qwen2.5-7B server; its init container quantizes to FP8 using the quantizer image.
echo "==> Deploying Qwen2.5-7B FP8 on g6e.12xlarge"
sed -e "s|012345678901.dkr.ecr.us-east-1|${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}|" \
  "$SCRIPT_DIR/fp8-g6e.12xlarge.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Qwen2.5-7B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/qwen7b-vllm-g6e-12xl-fp8 \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/qwen7b-vllm-g6e-12xl-fp8 \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/qwen7b-vllm-g6e-12xl-fp8 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Validate the serving manifest.
echo "==> Validating the Qwen2.5-7B serving manifest"
eks-bench validate-manifests \
  qwen2.5-7b/fp8-g6e.12xlarge.yaml
