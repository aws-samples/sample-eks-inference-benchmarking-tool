#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SPEC=${1:?Pass the speculative-token count, for example 5}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="kimi-k3-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Apply the kimi-k3-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount kimi-k3-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount kimi-k3-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the DSpark recipe with the weights bucket, region, and the swept draft-token count filled in.
echo "==> Deploying Kimi K3 DSpark with ${SPEC} speculative tokens on p6-b300.48xlarge"
sed -e "s|value: kimi-k3-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${AWS_REGION}\"|;}" \
  -e "/name: NUM_SPEC_TOKENS/{n;s|value: .*|value: \"${SPEC}\"|;}" \
  -e "/name: BENCH_CONFIG/{n;s|value: .*|value: \"b300-dspark-tp8-spec${SPEC}\"|;}" \
  "$SCRIPT_DIR/mxfp4-p6-b300.48xlarge-dspark.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Kimi K3 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/kimi-k3-b300-dspark \
  --timeout=3600s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/kimi-k3-b300-dspark \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Validate the serving manifest (the committed spec=7 recipe; only env values are swept).
echo "==> Validating the Kimi K3 serving manifest"
eks-bench validate-manifests \
  kimi-k3/mxfp4-p6-b300.48xlarge-dspark.yaml
