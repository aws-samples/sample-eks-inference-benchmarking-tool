#!/usr/bin/env bash
set -euo pipefail

: "${EKS_CLUSTER_NAME:?EKS_CLUSTER_NAME is required}"
: "${NAMESPACE:?NAMESPACE is required}"

DEPLOYMENT=qwen7b-vllm-g6e-bf16
MANIFEST=vllm-recipes/qwen2.5-7b/bf16-g6e.4xlarge.yaml

echo "==> Deploying Qwen2.5 7B"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$MANIFEST"

echo "==> Waiting for the Qwen2.5 server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  "deployment/$DEPLOYMENT" --timeout=3600s

echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  "deployment/$DEPLOYMENT" --container vllm -- curl -s localhost:8000/v1/models

echo "==> Reading the live KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  "deployment/$DEPLOYMENT" --container vllm -- \
  sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"

echo "==> Reading startup cache evidence"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  "deployment/$DEPLOYMENT" --container vllm \
  | grep -E "GPU KV cache size|Maximum concurrency|model weights took"

echo "==> Validating the Qwen2.5 serving manifest"
python3 app/cli.py validate-manifests qwen2.5-7b/bf16-g6e.4xlarge.yaml
