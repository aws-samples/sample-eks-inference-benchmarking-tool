#!/bin/bash
# push-image-kaniko.sh — Build a container image in-cluster with kaniko and push it to ECR.
#
# Builds on an amd64 cluster node with no Docker daemon, so no local Docker is required and an
# arm64 laptop never has to emulate the base. Pair of push-image.sh, which builds locally.
#
# Defaults build the quantizer image from app/quantize. Override to build any other image:
#   REPOSITORY=<repo> CONTEXT_DIR=<dir-with-a-self-contained-Dockerfile> \
#     ./scripts/push-image-kaniko.sh <tag>
# With CONTEXT_DIR set, that directory's whole contents become the build context and its Dockerfile
# must use COPY paths relative to the directory. With CONTEXT_DIR unset, the script assembles the
# quantizer context from app/quantize and rewrites its repo-root-relative COPY paths.
#
# The ServiceAccount named by BUILD_SA (default quantizer-build) is granted ECR push on REPOSITORY
# here via setup-ecr-push.sh. The build node needs ~120Gi of free ephemeral storage to unpack the base.
#
# Usage, from any directory:
#   EKS_CLUSTER_NAME=my-cluster NAMESPACE=default AWS_REGION=us-east-1 \
#     AWS_ACCOUNT_ID=012345678901 ./scripts/push-image-kaniko.sh [tag]
set -euo pipefail

: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID to your 12-digit AWS account id}"
TAG=${1:-${TAG:-latest}}
BUILD_SA=${BUILD_SA:-quantizer-build}
REPOSITORY=${REPOSITORY:-serving-sweep-quantizer}
CONTEXT_DIR=${CONTEXT_DIR:-}
# Layer caching speeds rebuilds but a stale cache repo can fail a COPY; set BUILD_CACHE=false to bypass.
BUILD_CACHE=${BUILD_CACHE:-true}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
DEST="${REGISTRY}/${REPOSITORY}:${TAG}"
CACHE="${REGISTRY}/${REPOSITORY}-cache"
JOB="build-${REPOSITORY}"
CONFIGMAP="${JOB}-ctx"

# Grant the build ServiceAccount ECR push on this repository (idempotent).
echo "==> Ensuring $BUILD_SA can push to $REPOSITORY"
"$SCRIPT_DIR/setup-ecr-push.sh" "$BUILD_SA" "$REPOSITORY"

# Create the image and cache ECR repositories if they do not exist.
echo "==> Ensuring ECR repositories exist"
aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$REPOSITORY" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$AWS_REGION" --repository-name "$REPOSITORY" >/dev/null
aws ecr describe-repositories --region "$AWS_REGION" --repository-names "${REPOSITORY}-cache" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$AWS_REGION" --repository-name "${REPOSITORY}-cache" >/dev/null

# Assemble the kaniko build context. A CONTEXT_DIR ships as-is; the quantizer default is flattened
# because its Dockerfile uses repo-root-relative COPY paths and the ConfigMap mounts files flat.
CTX="$(mktemp -d)"
if [ -n "$CONTEXT_DIR" ]; then
  echo "==> Building the kaniko context from $CONTEXT_DIR"
  cp -R "$REPO_ROOT/$CONTEXT_DIR/." "$CTX/"
  test -f "$CTX/Dockerfile" || { echo "FATAL: no Dockerfile in $CONTEXT_DIR" >&2; exit 1; }
else
  echo "==> Building the kaniko context from app/quantize"
  cp "$REPO_ROOT/app/quantize/requirements.txt" "$CTX/requirements.txt"
  cp "$REPO_ROOT"/app/quantize/src/*.py "$CTX/"
  sed -e 's#COPY app/quantize/requirements.txt#COPY requirements.txt#' \
      -e 's#COPY app/quantize/src/ /app/src/#COPY *.py /app/src/#' \
      "$REPO_ROOT/app/quantize/Dockerfile" > "$CTX/Dockerfile"
fi
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create configmap "$CONFIGMAP" \
  --from-file "$CTX" --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
rm -rf "$CTX"

# Start the kaniko build Job, replacing any previous one so its log is clean. backoffLimit 0 means
# one attempt, and kaniko exits non-zero on a build error, which fails the Job. The pod carries no
# GPU toleration, so it lands on a non-GPU node, where the CPU-only build belongs.
echo "==> Starting the kaniko build Job for $DEST"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" delete job "$JOB" --ignore-not-found
cat <<EOF | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: ${BUILD_SA}
      initContainers:
        # ConfigMap keys mount as symlinks (requirements.txt -> ..data/requirements.txt). kaniko copies
        # the symlink itself, leaving a dangling link in the image so the next RUN cannot read the file.
        # Dereference with cp -L into an emptyDir and build from there, which gives kaniko real files.
        - name: stage-context
          image: public.ecr.aws/docker/library/busybox:1.36
          command: ["sh", "-c", "cp -L /ctx/* /work/ && ls -la /work"]
          volumeMounts:
            - { name: ctx, mountPath: /ctx }
            - { name: work, mountPath: /work }
      containers:
        - name: kaniko
          image: ghcr.io/osscontainertools/kaniko:v1.28.5@sha256:738807f0e31daf07743f89260a95956dcc6ee4f62553f4f6344c990cabab8675
          args:
            - --dockerfile=/work/Dockerfile
            - --context=dir:///work
            - --destination=${DEST}
            - --cache=${BUILD_CACHE}
            - --cache-repo=${CACHE}
          volumeMounts:
            - { name: work, mountPath: /work }
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
            name: ${CONFIGMAP}
        - name: work
          emptyDir: {}
EOF

# Stream the build log, then fail the script if the Job did not complete. Unpacking a large base
# takes roughly 10-20 minutes on a cold cache; a warm cache is much faster.
echo "==> Waiting for the build to start, then streaming its log"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait --for=condition=ready \
  pod -l job-name="$JOB" --timeout=5m || true
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs -f job/"$JOB" || true
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait --for=condition=complete \
  job/"$JOB" --timeout=30m

echo "==> Image pushed: $DEST"
