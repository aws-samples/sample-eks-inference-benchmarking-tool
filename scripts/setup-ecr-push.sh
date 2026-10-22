#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
SERVICE_ACCOUNT=${1:-quantizer-build}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ROLE_NAME="${EKS_CLUSTER_NAME}-${SERVICE_ACCOUNT}-ecr"
REPOSITORY="serving-sweep-quantizer"
REPO_ARN="arn:aws:ecr:${AWS_REGION}:${ACCOUNT}:repository/${REPOSITORY}"
CACHE_ARN="arn:aws:ecr:${AWS_REGION}:${ACCOUNT}:repository/${REPOSITORY}-cache"
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
# Grant ECR push on the quantizer repositories; GetAuthorizationToken is account-wide by design.
echo "==> Granting $ROLE_NAME push on $REPOSITORY and ${REPOSITORY}-cache"
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name ecr-push \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"ecr:GetAuthorizationToken\",\"Resource\":\"*\"},{\"Effect\":\"Allow\",\"Action\":[\"ecr:BatchCheckLayerAvailability\",\"ecr:InitiateLayerUpload\",\"ecr:UploadLayerPart\",\"ecr:CompleteLayerUpload\",\"ecr:PutImage\",\"ecr:BatchGetImage\",\"ecr:GetDownloadUrlForLayer\"],\"Resource\":[\"${REPO_ARN}\",\"${CACHE_ARN}\"]}]}"
# Create the build ServiceAccount the kaniko Job runs as, if it does not exist.
echo "==> Applying ServiceAccount $NAMESPACE/$SERVICE_ACCOUNT"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount "$SERVICE_ACCOUNT" \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
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
