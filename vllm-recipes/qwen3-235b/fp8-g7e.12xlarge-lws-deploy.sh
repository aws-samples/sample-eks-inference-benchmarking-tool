#!/usr/bin/env bash
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this helper}"
: "${NAMESPACE:?Set NAMESPACE before running this helper}"

selector='app=qwen3-235b-lws,role=leader'

# Deploy the Qwen3 235B LeaderWorkerSet and its leader-only Service.
echo "==> Deploying Qwen3 235B"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
  -f vllm-recipes/qwen3-235b/fp8-g7e.12xlarge-lws.yaml

# Discover the controller-created leader pod by stable workload labels.
echo "==> Waiting for the Qwen3 leader pod to be created"
deadline=$((SECONDS + 3600))
leader_pod=""
while (( SECONDS < deadline )); do
  if leader_pod="$(kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" get pods \
      -l "$selector" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)" \
      && [[ -n "$leader_pod" ]]; then
    break
  fi
  sleep 5
done

if [[ -z "$leader_pod" ]]; then
  echo "Timed out waiting for a leader pod matching $selector" >&2
  kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" get pods \
    -l "$selector" -o wide >&2 || true
  exit 1
fi

# Pod readiness is tied to the leader's vLLM /health probe.
echo "==> Waiting for leader pod $leader_pod to become Ready"
remaining=$((deadline - SECONDS))
if (( remaining <= 0 )); then
  echo "Timed out before leader pod $leader_pod became Ready" >&2
  exit 1
fi
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=Ready "pod/$leader_pod" --timeout="${remaining}s"

# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  "$leader_pod" --container vllm-leader -- \
  curl -fsS localhost:8000/v1/models

# Print live KV-cache allocation evidence.
echo "==> Reading the live KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  "$leader_pod" --container vllm-leader -- \
  sh -c "curl -fsS localhost:8000/metrics | grep 'vllm:cache_config_info'"

# Validate the serving manifest.
echo "==> Validating the Qwen3 serving manifest"
eks-bench validate-manifests \
  qwen3-235b/fp8-g7e.12xlarge-lws.yaml
