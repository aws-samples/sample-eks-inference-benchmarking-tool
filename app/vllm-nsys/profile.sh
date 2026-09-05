#!/bin/bash
# Capture an Nsight Systems trace of a vLLM server to see where step time goes across attention,
# MLP, NCCL all-reduce and scheduling gaps, complementing the client-side TTFT, TPOT and ITL
# metrics that say how slow rather than why. CUDA and NCCL tracing needs the profiler present at
# process start, so this launches vllm serve under nsys rather than attaching, with --delay
# skipping model load and warmup and --duration bounding the capture to a steady-state window, so
# drive a load burst once the window opens. It is model-agnostic: set MODEL by environment variable
# or as the first argument.

set -euo pipefail

# MODEL is required by environment or first argument, with no default for a general-purpose profiler.
MODEL="${MODEL:-${1:-}}"
TP="${TP:-1}"
PORT="${PORT:-8000}"
NSYS_DELAY="${NSYS_DELAY:-180}"
NSYS_DURATION="${NSYS_DURATION:-60}"
OUT="${OUT:-./vllm-trace-$(date +%Y%m%d-%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -z "$MODEL" ]; then
  echo "ERROR: MODEL is required (path or HF id)." >&2
  echo "  e.g.  MODEL=Qwen/Qwen2.5-7B-Instruct ./app/vllm-nsys/profile.sh" >&2
  echo "  or    ./app/vllm-nsys/profile.sh /models/quantized/Qwen2.5-7B-Instruct-fp8" >&2
  exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not found. Use a CUDA devel image or install Nsight Systems." >&2
  echo "       (CUDA devel images ship nsys; the vLLM DLC does not by default.)" >&2
  exit 1
fi

echo "=== Nsight Systems capture of vLLM serve ==="
echo "  model=$MODEL  TP=$TP  port=$PORT"
echo "  capture: start after ${NSYS_DELAY}s, run for ${NSYS_DURATION}s -> ${OUT}.nsys-rep"
echo "  Drive load with app/bench/src/main.py once the capture window opens."
echo "============================================"

# Trace the CUDA runtime, NVTX ranges that NCCL emits, OS runtime and the math libs.
exec nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --delay "$NSYS_DELAY" \
  --duration "$NSYS_DURATION" \
  --force-overwrite true \
  -o "$OUT" \
  vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --port "$PORT" \
    $EXTRA_ARGS
