#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="kimi-k3-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Apply the kimi-k3-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount kimi-k3-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount kimi-k3-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the Kimi K3 server with the weights bucket and its region filled in.
echo "==> Deploying Kimi K3 MXFP4 with expert parallelism on p6-b300.48xlarge"
sed -e "s|value: kimi-k3-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${AWS_REGION}\"|;}" \
  "$SCRIPT_DIR/mxfp4-p6-b300.48xlarge-ep.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Kimi K3 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/kimi-k3-b300-ep \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/kimi-k3-b300-ep \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/kimi-k3-b300-ep \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print any expert-parallel / EPLB lines from the startup log to confirm the feature engaged.
# Informational: confirm the exact log string on the first run, then harden this into a gate.
echo "==> Confirming expert parallelism engaged"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/kimi-k3-b300-ep \
  --container vllm \
  | grep -iE "expert parallel|EPLB|num_redundant_experts" || true
# Validate the serving manifest.
echo "==> Validating the Kimi K3 serving manifest"
eks-bench validate-manifests \
  kimi-k3/mxfp4-p6-b300.48xlarge-ep.yaml
