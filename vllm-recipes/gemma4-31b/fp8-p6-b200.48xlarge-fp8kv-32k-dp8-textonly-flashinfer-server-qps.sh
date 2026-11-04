#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
# Run the open-loop server_qps capacity test against the gemma4-31b server. Poisson arrivals sweep a
# range of request rates; the harness reports the rate at which TTFT p99 stays within 500 ms and TPOT
# p99 within 50 ms. Pass --dry-run to render the Job without creating it. Any extra arguments are
# forwarded to eks-bench, so the dry-run step and the real run share this one script.
echo "==> Running the open-loop server_qps capacity test (TTFT p99 <= 500 ms, TPOT p99 <= 50 ms)"
eks-bench measure \
  --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-b200-fp8kv-32k-dp8-textonly-flashinfer \
  --protocol server_qps --profile coding_repo_context \
  --instance p6-b200.48xlarge --model-tag gemma4-31b \
  --config fp8kv-32k-dp8-textonly-flashinfer --quant fp8 \
  --model-id gemma4-31b \
  --tokenizer google/gemma-4-31b-it \
  --out-dir docs/results \
  --reps 1 \
  "$@"
