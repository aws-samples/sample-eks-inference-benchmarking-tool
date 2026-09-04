#!/usr/bin/env bash
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME to the EKS cluster name before running this helper.}"
: "${NAMESPACE:?Set NAMESPACE to the target Kubernetes namespace before running this helper.}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$SCRIPT_DIR/mxfp4-p6-b300.48xlarge-dspark.yaml"

if grep -Eq '^[[:space:]]*image:[[:space:]].*012345678901' "$MANIFEST"; then
  echo "FATAL: $MANIFEST still contains placeholder ECR account 012345678901; replace the image reference before deploying." >&2
  exit 1
fi

echo "==> Deploying the DSpark Kimi K3 B300 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$MANIFEST"

echo "==> Waiting for the DSpark Kimi K3 B300 Deployment"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/kimi-k3-b300-dspark --timeout=3600s

echo "==> Checking the served model from the API Pod"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/kimi-k3-b300-dspark --container vllm -- \
  curl -fsS localhost:8000/v1/models

echo "==> Reading the live KV-cache configuration from the API Pod"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/kimi-k3-b300-dspark --container vllm -- \
  sh -c "curl -fsS localhost:8000/metrics | grep 'vllm:cache_config_info'"

echo "==> Validating the Kimi K3 serving manifest"
(
  cd "$REPO_ROOT"
  python3 app/cli.py validate-manifests kimi-k3/mxfp4-p6-b300.48xlarge-dspark.yaml
)
