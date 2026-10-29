#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Authenticate to public ECR so eksctl can pull the Karpenter Helm chart (the token lives in us-east-1).
echo "==> Logging in to public ECR for the Karpenter chart"
aws ecr-public get-login-password --region us-east-1 | helm registry login --username AWS --password-stdin public.ecr.aws
# Discover the region's standard availability zones so the cluster spans them all. This guarantees a
# Capacity Block in any of the region's zones has a subnet in its AZ for Karpenter to launch into.
# Local zones and opt-in zones are excluded, since the EKS control plane does not run in them.
echo "==> Discovering standard availability zones in $AWS_REGION"
AZS=$(aws ec2 describe-availability-zones --region "$AWS_REGION" \
  --filters "Name=zone-type,Values=availability-zone" "Name=opt-in-status,Values=opt-in-not-required" \
  --query 'AvailabilityZones[].ZoneName' --output text)
AZ_LIST="[$(printf '%s' "$AZS" | tr '[:space:]' '\n' | grep -v '^$' | sort | paste -sd, -)]"
echo "    availability zones: $AZ_LIST"
# Create the cluster if it does not exist, filling the name, region and AZs from the environment.
echo "==> Creating cluster $EKS_CLUSTER_NAME in $AWS_REGION"
eksctl get cluster --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" >/dev/null 2>&1 \
  || sed -e "s|CLUSTER_NAME|${EKS_CLUSTER_NAME}|g" -e "s|region: AWS_REGION|region: ${AWS_REGION}|" \
    -e "s|availabilityZones: AVAILABILITY_ZONES|availabilityZones: ${AZ_LIST}|" \
    "$SCRIPT_DIR/../k8s/examples/eksctl-cluster.yaml" \
    | eksctl create cluster -f -
# Point kubectl at the cluster under a context named exactly $EKS_CLUSTER_NAME, which the steps below
# and every runbook use via --context. Runs on a rerun too, so the context exists even when the
# cluster already did.
echo "==> Writing kubeconfig context $EKS_CLUSTER_NAME"
aws eks update-kubeconfig --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" --alias "$EKS_CLUSTER_NAME"
# Grant the Karpenter controller role permission to read Capacity Reservations. eksctl's Karpenter
# policy omits it, and without it Karpenter cannot launch into a Capacity Block or ODCR.
echo "==> Granting Karpenter the capacity-reservation read permission"
KARPENTER_ROLE=$(kubectl --context "$EKS_CLUSTER_NAME" --namespace karpenter get serviceaccount karpenter \
  -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}')
aws iam put-role-policy --role-name "${KARPENTER_ROLE##*/}" \
  --policy-name karpenter-capacity-reservations \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Sid":"CapacityReservations","Effect":"Allow","Action":["ec2:DescribeCapacityReservations"],"Resource":"*"}]}'
# Apply the standing CPU NodePool that every benchmark client Job runs on, off the serving node.
echo "==> Applying the benchmark-client CPU NodePool"
NODE_ROLE="${KARPENTER_NODE_ROLE:-eksctl-KarpenterNodeRole-${EKS_CLUSTER_NAME}}"
DISCOVERY="${KARPENTER_DISCOVERY:-${EKS_CLUSTER_NAME}}"
sed -e "s|NODE_ROLE|${NODE_ROLE}|g" -e "s|CLUSTER_NAME|${DISCOVERY}|g" \
  "$SCRIPT_DIR/../k8s/examples/karpenter-cpu-nodepool.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" apply -f -
# Apply the large-NVMe CPU NodePool that huge-checkpoint HF-to-S3 staging Jobs run on.
echo "==> Applying the large-NVMe CPU staging NodePool"
sed -e "s|NODE_ROLE|${NODE_ROLE}|g" -e "s|CLUSTER_NAME|${DISCOVERY}|g" \
  "$SCRIPT_DIR/../k8s/examples/karpenter-cpu-nvme-nodepool.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" apply -f -
# Show the cluster.
echo "==> Showing cluster $EKS_CLUSTER_NAME"
eksctl get cluster --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION"
