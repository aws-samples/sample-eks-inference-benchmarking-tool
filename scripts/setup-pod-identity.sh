#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SERVICE_ACCOUNT=${1:?Pass the ServiceAccount name, for example mistral-large-3-bench}
ROLE_NAME="${EKS_CLUSTER_NAME}-${SERVICE_ACCOUNT}-s3"
# The ServiceAccount is <model>-bench, and the model's weights bucket is <model>-model-weights-<account-id>-<region>.
MODEL=${SERVICE_ACCOUNT%-bench}
WEIGHTS_BUCKET="${MODEL//./-}-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Install the EKS Pod Identity Agent add-on if the cluster lacks it.
echo "==> Installing the eks-pod-identity-agent add-on"
aws eks describe-addon --cluster-name "$EKS_CLUSTER_NAME" --addon-name eks-pod-identity-agent \
  --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws eks create-addon --cluster-name "$EKS_CLUSTER_NAME" --addon-name eks-pod-identity-agent \
    --region "$AWS_REGION" >/dev/null
# Wait for the add-on to become active.
echo "==> Waiting for the eks-pod-identity-agent add-on"
aws eks wait addon-active --cluster-name "$EKS_CLUSTER_NAME" --addon-name eks-pod-identity-agent \
  --region "$AWS_REGION"
# Create the IAM role that EKS Pod Identity can assume, if it does not exist.
echo "==> Creating IAM role $ROLE_NAME"
aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1 \
  || aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"pods.eks.amazonaws.com"},"Action":["sts:AssumeRole","sts:TagSession"]}]}' >/dev/null
# Grant the role read and write on the weights bucket only.
echo "==> Granting $ROLE_NAME read and write on s3://$WEIGHTS_BUCKET"
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name weights-bucket \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\",\"s3:GetBucketLocation\"],\"Resource\":\"arn:aws:s3:::${WEIGHTS_BUCKET}\"},{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:AbortMultipartUpload\"],\"Resource\":\"arn:aws:s3:::${WEIGHTS_BUCKET}/*\"}]}"
# Associate the role with the ServiceAccount, unless an association already exists.
echo "==> Associating $ROLE_NAME with $NAMESPACE/$SERVICE_ACCOUNT"
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)
aws eks list-pod-identity-associations --cluster-name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" \
  --namespace "$NAMESPACE" --service-account "$SERVICE_ACCOUNT" \
  --query 'associations[0].associationId' --output text | grep -v None \
  || aws eks create-pod-identity-association --cluster-name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" \
    --namespace "$NAMESPACE" --service-account "$SERVICE_ACCOUNT" --role-arn "$ROLE_ARN" >/dev/null
# Show the association in force.
echo "==> Showing the Pod Identity association"
aws eks list-pod-identity-associations --cluster-name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" \
  --namespace "$NAMESPACE" --service-account "$SERVICE_ACCOUNT" \
  --query 'associations[0].associationArn' --output text | grep "arn:aws"
