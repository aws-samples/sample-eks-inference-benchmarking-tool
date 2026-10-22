#!/bin/bash
# push-image-kaniko.sh — Build the quantizer image in-cluster with kaniko and push it to ECR.
#
# Builds on an amd64 cluster node with no Docker daemon, so no local Docker is required and an
# arm64 laptop never has to emulate the ~25 GB base. Pair of push-image.sh, which builds locally.
#
# The build identity (the ServiceAccount named by BUILD_SA, or the node role when it is "default")
# must be allowed to push to ECR: ecr:GetAuthorizationToken, ecr:BatchCheckLayerAvailability,
# ecr:InitiateLayerUpload, ecr:UploadLayerPart, ecr:CompleteLayerUpload, ecr:PutImage,
# ecr:BatchGetImage. The build node needs ~120Gi of free ephemeral storage to unpack the base.
#
# Usage, from any directory:
#   EKS_CLUSTER_NAME=my-cluster NAMESPACE=default AWS_REGION=us-east-1 \
#     AWS_ACCOUNT_ID=012345678901 ./scripts/push-image-kaniko.sh [tag]
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID to your 12-digit AWS account id}"
TAG=${1:-latest}
BUILD_SA=${BUILD_SA:-default}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY="serving-sweep-quantizer"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
DEST="${REGISTRY}/${REPOSITORY}:${TAG}"
CACHE="${REGISTRY}/${REPOSITORY}-cache"

# Create the image and cache ECR repositories if they do not exist.
echo "==> Ensuring ECR repositories exist"
aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$REPOSITORY" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$AWS_REGION" --repository-name "$REPOSITORY" >/dev/null
aws ecr describe-repositories --region "$AWS_REGION" --repository-names "${REPOSITORY}-cache" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$AWS_REGION" --repository-name "${REPOSITORY}-cache" >/dev/null

# Assemble a flat build context from the live quantize files. The ConfigMap mounts every file at
# /workspace root, so the Dockerfile's repo-root-relative COPY paths are flattened to match.
echo "==> Building the kaniko context ConfigMap from app/quantize"
CTX="$(mktemp -d)"
cp "$REPO_ROOT/app/quantize/requirements.txt" "$CTX/requirements.txt"
cp "$REPO_ROOT"/app/quantize/src/*.py "$CTX/"
sed -e 's#COPY app/quantize/requirements.txt#COPY requirements.txt#' \
    -e 's#COPY app/quantize/src/ /app/src/#COPY *.py /app/src/#' \
    "$REPO_ROOT/app/quantize/Dockerfile" > "$CTX/Dockerfile"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create configmap quantizer-build-ctx \
  --from-file "$CTX" --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
rm -rf "$CTX"

# Start the kaniko build Job, replacing any previous one so its log is clean. backoffLimit 0 means
# one attempt, and kaniko exits non-zero on a build error, which fails the Job. The pod carries no
# GPU toleration, so it lands on a non-GPU node, where the CPU-only build belongs.
echo "==> Starting the kaniko build Job for $DEST"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" delete job quantizer-build --ignore-not-found
cat <<EOF | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: quantizer-build
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: ${BUILD_SA}
      containers:
        - name: kaniko
          image: ghcr.io/osscontainertools/kaniko:v1.28.5@sha256:738807f0e31daf07743f89260a95956dcc6ee4f62553f4f6344c990cabab8675
          args:
            - --dockerfile=/workspace/Dockerfile
            - --context=dir:///workspace
            - --destination=${DEST}
            - --cache=true
            - --cache-repo=${CACHE}
          volumeMounts:
            - name: ctx
              mountPath: /workspace
          resources:
            requests:
              cpu: "4"
              memory: 24Gi
              ephemeral-storage: 80Gi
            limits:
              ephemeral-storage: 120Gi
      volumes:
        - name: ctx
          configMap:
            name: quantizer-build-ctx
EOF

# Stream the build log, then fail the script if the Job did not complete. Unpacking the ~25 GB base
# takes roughly 10-20 minutes on a cold cache; a warm cache is much faster.
echo "==> Waiting for the build to start, then streaming its log"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait --for=condition=ready \
  pod -l job-name=quantizer-build --timeout=5m || true
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs -f job/quantizer-build || true
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait --for=condition=complete \
  job/quantizer-build --timeout=30m

echo "==> Image pushed: $DEST"
