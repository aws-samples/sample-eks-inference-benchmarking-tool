#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="nemotron-3-ultra-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Apply the nemotron-3-ultra-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount nemotron-3-ultra-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount nemotron-3-ultra-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the Nemotron 3 Ultra server with the weights bucket and its region filled in.
echo "==> Deploying Nemotron 3 Ultra NVFP4 (DP2xTP4 + expert parallel, bf16 SSM cache, 512 seqs) on p6-b200.48xlarge"
sed -e "s|value: nemotron-3-ultra-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${AWS_REGION}\"|;}" \
  "$SCRIPT_DIR/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-bf16seq512.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the Nemotron 3 Ultra server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k-dp2tp4-bf16seq512 \
  --timeout=7200s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k-dp2tp4-bf16seq512 \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation. Confirm the pool grew versus the dp2tp4 baseline.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k-dp2tp4-bf16seq512 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Check for expert parallelism in the server log. The result file's provenance block is the
# authoritative record ("expert parallel on"); this is a convenience and never fails the deploy.
echo "==> Checking for expert parallelism in the server log"
if ! kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k-dp2tp4-bf16seq512 \
  --container vllm \
  | grep -iqE "expert.?parallel|EP size|enable_expert_parallel"; then
  echo "  NOTE: no expert-parallel line matched; confirm 'expert parallel on' in the result provenance block"
fi
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is FLASHINFER"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k-dp2tp4-bf16seq512 \
  --container vllm \
  | grep "Using AttentionBackendEnum.FLASHINFER backend"
# Confirm the Mamba SSM cache is bfloat16, the memory change that funds the higher sequence ceiling.
echo "==> Confirming the Mamba SSM cache dtype is bfloat16"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/nemotron-3-ultra-vllm-b200-fp8kv-32k-dp2tp4-bf16seq512 \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep -o 'mamba_ssm_cache_dtype=\"[^\"]*\"'"
# Validate the serving manifest.
echo "==> Validating the Nemotron 3 Ultra serving manifest"
eks-bench validate-manifests \
  nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-bf16seq512.yaml
