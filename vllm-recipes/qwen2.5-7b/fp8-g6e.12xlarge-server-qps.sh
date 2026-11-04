#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
# Run the open-loop server_qps capacity test against the qwen2.5-7b server. Poisson arrivals sweep a
# range of request rates; the harness reports the rate at which TTFT p99 stays within 500 ms and TPOT
# p99 within 50 ms. Pass --dry-run to render the Job without creating it. Any extra arguments are
# forwarded to eks-bench, so the dry-run step and the real run share this one script.
echo "==> Running the open-loop server_qps capacity test (TTFT p99 <= 500 ms, TPOT p99 <= 50 ms)"
eks-bench measure \
  --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-12xl-fp8 \
  --protocol server_qps --profile legacy_corpus \
  --instance g6e.12xlarge --model-tag qwen2.5-7b \
  --config fp8 --quant fp8 \
  --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --out-dir docs/results \
  --reps 1 \
  "$@"
