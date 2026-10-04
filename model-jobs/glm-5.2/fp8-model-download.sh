#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Confirm the fsx-pvc claim is bound.
echo "==> Checking claim fsx-pvc"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=jsonpath='{.status.phase}'=Bound pvc/fsx-pvc \
  --timeout=2m
# Start the Job that downloads the published checkpoint onto the claim.
echo "==> Downloading zai-org/GLM-5.2-FP8 to /models/glm-5.2-fp8"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f "$SCRIPT_DIR/fp8-model-download.yaml"
# Wait for the download to finish.
echo "==> Waiting for the download"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/download-glm-52-fp8 \
  --timeout=6h
# Confirm the Job found all 141 shards and no partial file.
echo "==> Checking the shard count"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs job/download-glm-52-fp8 \
  | grep -x 'shards: 141/141, partial files: 0'
