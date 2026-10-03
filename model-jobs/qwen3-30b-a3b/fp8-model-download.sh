#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${GPU_NODE:?Set GPU_NODE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Confirm the pinned node advertises two GPUs.
echo "==> Checking that $GPU_NODE advertises two GPUs"
kubectl --context "$EKS_CLUSTER_NAME" get node "$GPU_NODE" \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' | grep -x 2
# Confirm no running pod on the pinned node holds a GPU.
echo "==> Checking that no pod on $GPU_NODE holds a GPU"
test -z "$(kubectl --context "$EKS_CLUSTER_NAME" get pods -A \
  --field-selector "spec.nodeName=$GPU_NODE,status.phase=Running" \
  -o jsonpath='{.items[*].spec.containers[*].resources.limits.nvidia\.com/gpu}')"
# Create the claim and the download Job, pinned to the node the recipes use.
echo "==> Staging Qwen/Qwen3-30B-A3B-FP8 onto g7e12-switch-model-storage"
sed "s|ip-10-0-0-1.us-east-1.compute.internal|${GPU_NODE}|" "$SCRIPT_DIR/fp8-model-download.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the download to finish.
echo "==> Waiting for the download"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/download-qwen3-30b-a3b-fp8 \
  --timeout=7200s
# Confirm the claim is bound.
echo "==> Checking the claim"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=jsonpath='{.status.phase}'=Bound pvc/g7e12-switch-model-storage \
  --timeout=2m
# Confirm the Job found all 7 shards and no partial file.
echo "==> Checking the shard count"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs job/download-qwen3-30b-a3b-fp8 \
  | grep -x 'shards: 7/7, partial files: 0'
