#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Instance types whose EFA interfaces the plugin must advertise; default to the Kimi K3 serving node.
TYPES=("$@")
if [ ${#TYPES[@]} -eq 0 ]; then
  TYPES=(p6-b200.48xlarge)
fi
# Install or upgrade the AWS EFA device plugin, which advertises vpc.amazonaws.com/efa on EFA nodes.
echo "==> Installing the aws-efa-k8s-device-plugin Helm chart"
helm repo add eks https://aws.github.io/eks-charts >/dev/null 2>&1 || true
helm repo update eks >/dev/null
helm upgrade --install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin \
  --namespace kube-system --kube-context "$EKS_CLUSTER_NAME" --wait
# Teach the plugin about any instance type newer than its hardcoded allow-list, which is why a fresh
# node can otherwise advertise no EFA. Idempotent: adds only missing entries.
echo "==> Patching the EFA plugin allow-list for ${TYPES[*]}"
"$SCRIPT_DIR/patch-efa-plugin.sh" "$EKS_CLUSTER_NAME" "${TYPES[@]}"
