#!/usr/bin/env bash
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME to the EKS cluster name before running this helper.}"
: "${NAMESPACE:?Set NAMESPACE to the target Kubernetes namespace before running this helper.}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$SCRIPT_DIR/mxfp4-p6-b200.48xlarge-lws.yaml"

if grep -Eq '^[[:space:]]*image:[[:space:]].*012345678901' "$MANIFEST"; then
  echo "FATAL: $MANIFEST still contains placeholder ECR account 012345678901; replace the image reference before deploying." >&2
  exit 1
fi

echo "==> Deploying Kimi K3 B200 LeaderWorkerSet"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$MANIFEST"

echo "==> Discovering the Kimi K3 leader Pod"
leader_pod=""
for ((attempt = 1; attempt <= 60; attempt++)); do
  leader_pod="$(kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" get pods \
    -l 'app=kimi-k3-b200,role=leader' \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$leader_pod" ]] && break
  sleep 5
done
if [[ -z "$leader_pod" ]]; then
  echo "FATAL: no leader Pod found with labels app=kimi-k3-b200,role=leader after 300 seconds." >&2
  exit 1
fi

echo "==> Waiting for leader Pod $leader_pod"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  "pod/$leader_pod" --for=condition=Ready --timeout=3600s

echo "==> Checking the served model from leader Pod $leader_pod"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  "pod/$leader_pod" --container vllm-leader -- \
  curl -fsS localhost:8000/v1/models

echo "==> Reading the live KV-cache configuration from leader Pod $leader_pod"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  "pod/$leader_pod" --container vllm-leader -- \
  sh -c "curl -fsS localhost:8000/metrics | grep 'vllm:cache_config_info'"

echo "==> Validating the Kimi K3 serving manifest"
(
  cd "$REPO_ROOT"
  python3 app/cli.py validate-manifests kimi-k3/mxfp4-p6-b200.48xlarge-lws.yaml
)
