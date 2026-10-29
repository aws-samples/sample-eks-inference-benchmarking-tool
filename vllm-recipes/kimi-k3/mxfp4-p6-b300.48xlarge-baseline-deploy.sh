#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="kimi-k3-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Apply the kimi-k3-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount kimi-k3-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount kimi-k3-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the Kimi K3 server with the weights bucket and its region filled in.
echo "==> Deploying Kimi K3 MXFP4 on p6-b300.48xlarge"
sed -e "s|value: kimi-k3-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${AWS_REGION}\"|;}" \
  "$SCRIPT_DIR/mxfp4-p6-b300.48xlarge-baseline.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Block until the weights stage and the server passes its readiness probe. A cold start is long
# (stage weights, load the checkpoint, capture CUDA graphs, JIT-compile kernels), so a background
# poll prints the pod state and the latest server log line every 30s while we wait. This command
# does not return until the server is ready, so this script finishing is the signal that
# capture-reference and measure are safe to run.
# Wait until the server is Ready, then return — on its own, never needing Ctrl-C. A short per-cycle
# rollout-status check doubles as the readiness probe and paces the loop, printing a status line about
# every 30s. The loop exits the instant the deployment is available; a 120-minute deadline makes it
# give up and exit non-zero on its own if the server never comes up. Foreground only, so nothing
# outlives the script.
echo "==> Waiting for the Kimi K3 server to become ready; cold start is ~30-80 min. Prints status ~every 30s and returns on its own when Ready:"
READY_DEADLINE=$(( $(date +%s) + 7200 ))
until kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/kimi-k3-b300 --timeout=30s >/dev/null 2>&1; do
    printf '    [%s] status: ' "$(date -u +%H:%M:%S)"
    kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" get pod -l app=kimi-k3-b300 -o 'custom-columns=READY:.status.containerStatuses[0].ready,PHASE:.status.phase,RESTARTS:.status.containerStatuses[0].restartCount' --no-headers 2>/dev/null || true
    kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs deployment/kimi-k3-b300 --all-containers=true --tail=1 2>/dev/null | sed 's/^/      latest: /' | cut -c1-150 || true
    [ "$(date +%s)" -lt "$READY_DEADLINE" ] || { echo "==> FATAL: deployment/kimi-k3-b300 not Ready after 120 min; aborting. Inspect: kubectl --context \"$EKS_CLUSTER_NAME\" -n \"$NAMESPACE\" get pods -l app=kimi-k3-b300" >&2; exit 1; }
done
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/kimi-k3-b300 \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation from the startup log. Informational, so it never fails the
# deploy: the server is already serving by this point.
echo "==> Reading the KV-cache allocation"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/kimi-k3-b300 \
  --container vllm 2>/dev/null \
  | grep -i "GPU KV cache size" | tail -1 || true
# Validate the serving manifest.
echo "==> Validating the Kimi K3 serving manifest"
eks-bench validate-manifests \
  kimi-k3/mxfp4-p6-b300.48xlarge-baseline.yaml
# Announce readiness unmistakably, so the next step is never started against a server still loading.
echo "==> READY: deployment/kimi-k3-b300 is serving in namespace $NAMESPACE. Safe to run capture-reference and measure now."
