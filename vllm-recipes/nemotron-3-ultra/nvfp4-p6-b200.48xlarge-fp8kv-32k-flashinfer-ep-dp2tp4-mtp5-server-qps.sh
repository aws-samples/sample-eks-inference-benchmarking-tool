#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
# Run the open-loop server_qps capacity test against the mtp5 server. Poisson arrivals sweep a range
# of request rates; the harness reports the rate at which the SLA still holds. Pass --dry-run to render
# the Job without creating it. Any extra arguments are forwarded to eks-bench, so the dry-run step and
# the real run share this one script.
echo "==> Running the open-loop server_qps capacity test"
eks-bench measure \
  --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
  --service nemotron-3-ultra-vllm-b200-fp8kv-32k-flashinfer-ep-dp2tp4-mtp5 \
  --protocol server_qps --profile coding_repo_context \
  --instance p6-b200.48xlarge --model-tag nemotron-3-ultra \
  --config fp8kv-32k-dp2tp4-flashinfer-ep-mtp5 --quant nvfp4 \
  --model-id nemotron-3-ultra \
  --tokenizer nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --out-dir docs/results \
  --reps 1 \
  "$@"
