#!/bin/bash
# push-image.sh — Create the quantizer's ECR repository if needed, then build and push the image.
#
# Run this once before your first benchmark, and again any time
# app/quantize/Dockerfile or app/quantize/src/quantize.py changes.
#
# Usage, from any directory:
#   AWS_ACCOUNT_ID=012345678901 AWS_REGION=us-east-1 ./scripts/push-image.sh
#   ECR_IMAGE=<your-full-uri> ./scripts/push-image.sh   # override the whole URI instead

set -e

# The Dockerfile's COPY paths are repo-root-relative, so the build context has to be the repo
# root. Resolving it from this script's own location means the caller's directory is irrelevant.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID to your 12-digit AWS account id}"
: "${AWS_REGION:?set AWS_REGION to the region holding your ECR repository}"

ECR_IMAGE="${ECR_IMAGE:-${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/serving-sweep-quantizer:latest}"
REGION="$AWS_REGION"
ACCOUNT=$(echo "$ECR_IMAGE" | cut -d. -f1)
REPOSITORY="${ECR_IMAGE#*/}"
REPOSITORY="${REPOSITORY%:*}"
echo "==> Creating ECR repository $REPOSITORY if it does not exist..."
aws ecr describe-repositories --region "$REGION" --repository-names "$REPOSITORY" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPOSITORY" >/dev/null

echo "==> Authenticating Docker to ECR..."
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Building quantizer image..."
docker build --platform linux/amd64 -t "$ECR_IMAGE" -f app/quantize/Dockerfile .

echo "==> Pushing to ECR..."
docker push "$ECR_IMAGE"

echo ""
echo "Done. Image pushed to: $ECR_IMAGE"
echo "Make sure your quantize Job references this image URI "
echo "(start from k8s/examples/job-quantize-model.yaml)."
