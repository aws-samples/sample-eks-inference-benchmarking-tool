#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"

# Configure kubectl with a context named after the EKS cluster.
echo "==> Configuring kubeconfig context $EKS_CLUSTER_NAME"
aws eks update-kubeconfig \
  --name "$EKS_CLUSTER_NAME" \
  --region "$AWS_REGION" \
  --alias "$EKS_CLUSTER_NAME"

# Create the namespace used by benchmark workloads.
echo "==> Creating namespace $NAMESPACE"
kubectl --context "$EKS_CLUSTER_NAME" create namespace "$NAMESPACE" \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" apply -f -

# Add NVIDIA's device-plugin Helm repository.
echo "==> Adding the NVIDIA device-plugin Helm repository"
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin --force-update

# Refresh the local device-plugin chart metadata.
echo "==> Updating the NVIDIA device-plugin Helm repository"
helm repo update nvdp

# Install the pinned NVIDIA Kubernetes device plugin.
echo "==> Installing NVIDIA device plugin chart 0.20.0"
helm upgrade --install nvdp nvdp/nvidia-device-plugin \
  --version 0.20.0 \
  --namespace nvidia-device-plugin \
  --create-namespace \
  --kube-context "$EKS_CLUSTER_NAME" \
  --atomic \
  --wait \
  --timeout 5m

# Create or update the Hugging Face token Secret used by model downloads.
echo "==> Applying Secret $NAMESPACE/hf-token"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
  create secret generic hf-token \
  --from-literal=token="$HUGGING_FACE_HUB_TOKEN" \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -

# Confirm the Hugging Face token secret exists.
echo "==> Confirming secret $NAMESPACE/hf-token"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" get secret hf-token
