#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="kimi-k3-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Confirm both p6-b200.48xlarge nodes advertise 8 EFA interfaces.
echo "==> Checking EFA on the p6-b200.48xlarge nodes"
kubectl --context "$EKS_CLUSTER_NAME" get nodes -l node.kubernetes.io/instance-type=p6-b200.48xlarge \
  -o jsonpath='{.items[*].status.allocatable.vpc\.amazonaws\.com/efa}' | grep -x '8 8'
# Apply the kimi-k3-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount kimi-k3-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount kimi-k3-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the Kimi K3 server with the weights bucket, its region and the image registry filled in.
echo "==> Deploying Kimi K3 MXFP4 on p6-b200.48xlarge"
sed -e "s|value: kimi-k3-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${AWS_REGION}\"|;}" \
  -e "s|012345678901.dkr.ecr.us-east-1|${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}|" \
  "$SCRIPT_DIR/mxfp4-p6-b200.48xlarge-sts.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Kimi K3 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  statefulset/kimi-k3-b200 \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  pod/kimi-k3-b200-0 \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  pod/kimi-k3-b200-0 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Validate the serving manifest.
echo "==> Validating the Kimi K3 serving manifest"
eks-bench validate-manifests \
  kimi-k3/mxfp4-p6-b200.48xlarge-sts.yaml
