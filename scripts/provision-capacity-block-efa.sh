#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
INSTANCE_TYPE=${1:?Pass the instance type, for example p6-b200.48xlarge}
RESERVATION_ID=${2:?Pass a Capacity Block or ODCR reservation id (e.g. cr-0123456789abcdef0) sized for all nodes. To find one, run: aws ec2 describe-capacity-block-offerings --instance-type <type> --instance-count <n> --capacity-duration-hours <h>. This script never purchases a reservation.}
NODE_COUNT=${3:-2}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# A Karpenter resource name cannot contain dots, so the node pool name uses dashes. The name is distinct
# from the single-node bench-cb-<instance> pool so the two never overwrite each other.
NODE_NAME="bench-cb-efa-${INSTANCE_TYPE//./-}"
# These follow the standard Karpenter blueprint naming; override either if your cluster differs.
NODE_ROLE="${KARPENTER_NODE_ROLE:-eksctl-KarpenterNodeRole-${EKS_CLUSTER_NAME}}"
DISCOVERY="${KARPENTER_DISCOVERY:-${EKS_CLUSTER_NAME}}"
# One node's GPU count times the node count caps the pool at the whole multi-node run.
PER_NODE_GPUS=$(aws ec2 describe-instance-types --instance-types "$INSTANCE_TYPE" --region "$AWS_REGION" \
  --query 'InstanceTypes[0].GpuInfo.Gpus[0].Count' --output text)
GPU_LIMIT=$((PER_NODE_GPUS * NODE_COUNT))
# Apply the EC2NodeClass and NodePool with the cluster, instance type and reservation filled in.
echo "==> Applying EFA Capacity Block NodePool $NODE_NAME ($INSTANCE_TYPE x $NODE_COUNT, $RESERVATION_ID, $GPU_LIMIT GPUs, EFA, NVMe RAID0)"
sed -e "s|NODE_NAME|${NODE_NAME}|g" \
  -e "s|NODE_ROLE|${NODE_ROLE}|g" \
  -e "s|CLUSTER_NAME|${DISCOVERY}|g" \
  -e "s|INSTANCE_TYPE|${INSTANCE_TYPE}|g" \
  -e "s|RESERVATION_ID|${RESERVATION_ID}|g" \
  -e "s|GPU_LIMIT|${GPU_LIMIT}|g" \
  "$SCRIPT_DIR/../k8s/examples/karpenter-efa-nodepool.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" apply -f -
# Show the node pool. Nodes launch only while the reservation is active and the benchmark pods need them.
echo "==> Showing the NodePool"
kubectl --context "$EKS_CLUSTER_NAME" get nodepool "$NODE_NAME"
