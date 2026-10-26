#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${GPU_NODE:?Set GPU_NODE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="qwen3-30b-a3b-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Confirm the pinned node advertises two GPUs.
echo "==> Checking that $GPU_NODE advertises two GPUs"
kubectl --context "$EKS_CLUSTER_NAME" get node "$GPU_NODE" \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' | grep -x 2
# Confirm no running pod on the pinned node holds a GPU.
echo "==> Checking that no pod on $GPU_NODE holds a GPU"
test -z "$(kubectl --context "$EKS_CLUSTER_NAME" get pods -A \
  --field-selector "spec.nodeName=$GPU_NODE,status.phase=Running" \
  -o jsonpath='{.items[*].spec.containers[*].resources.limits.nvidia\.com/gpu}')"
# Apply the qwen3-30b-a3b-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount qwen3-30b-a3b-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount qwen3-30b-a3b-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the Qwen3-30B-A3B server with the weights bucket, its region and GPU_NODE filled in.
echo "==> Deploying Qwen3-30B-A3B FP8 on $GPU_NODE"
sed -e "s|value: qwen3-30b-a3b-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${AWS_REGION}\"|;}" \
  -e "s|SUBSTITUTE-gpu-node|${GPU_NODE}|" \
  "$SCRIPT_DIR/g7e12-switch-qwen3-30b-a3b-default.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Qwen3-30B-A3B server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/qwen3-30b-a3b-g7e12-switch-default \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/qwen3-30b-a3b-g7e12-switch-default \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/qwen3-30b-a3b-g7e12-switch-default \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
