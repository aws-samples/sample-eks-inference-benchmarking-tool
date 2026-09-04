#!/bin/bash
# profile.sh — Capture an Nsight Systems (nsys) trace of a vLLM server to show where step
# time goes: attention vs MLP vs NCCL all-reduce vs scheduling gaps. This complements the
# client-side latency metrics (TTFT/TPOT/ITL) in app/bench/src/main.py, which report how
# long a request took rather than which stage consumed the time.
#
# The script launch-wraps rather than attaches, because CUDA/NCCL tracing needs the profiler
# present at process start, so `vllm serve` runs under nsys. --delay skips model load and
# warmup; --duration bounds the capture to a steady-state window. Start a benchmark burst
# (below) so the captured window contains real decode traffic.
#
# Model-agnostic: works on ANY vLLM-servable model. Set MODEL (env or first arg).
#
# USAGE (local / inside a GPU container that has nsys + vllm):
#   MODEL=Qwen/Qwen2.5-7B-Instruct TP=1 ./app/vllm-nsys/profile.sh
#   MODEL=/models/qwen3-235b-a22b-fp8 TP=4 NSYS_DELAY=900 ./app/vllm-nsys/profile.sh
#   ./app/vllm-nsys/profile.sh /models/quantized/Qwen2.5-7B-Instruct-fp8   # model as $1
#   # ...then, once the server is up and the capture window opens, drive load:
#   VLLM_API_URL=http://localhost:8000/v1 MODEL_ID=$MODEL INSTANCE_TYPE=<type> \
#     CONCURRENCY=32 NUM_REQUESTS=200 python app/bench/src/main.py
#
# OUTPUT: $OUT.nsys-rep — open in the Nsight Systems GUI, or:
#   nsys stats $OUT.nsys-rep            # summary tables (incl. NCCL/kernel time)
#
# ENV (all optional):
#   MODEL   (REQUIRED)  model path or HF id — any vLLM-servable model (env or $1)
#   TP             tensor-parallel size                (default: 1)
#   PORT           vLLM port                           (default: 8000)
#   NSYS_DELAY     seconds to wait before capturing    (default: 180 — MUST exceed model
#                  load + warmup; raise for large models, e.g. ~900 for a 235B from FSx)
#   NSYS_DURATION  seconds to capture                  (default: 60)
#   OUT            output trace path (no extension)     (default: ./vllm-trace-<ts>)
#   EXTRA_ARGS     extra flags passed to `vllm serve`  (default: empty)
#
# NOTE (Kubernetes): to profile the in-cluster server, put this nsys-wrapped command
# in the container `args` of the serving manifest (the image must include nsys —
# the CUDA devel images do; the vLLM DLC does not by default, so add it or use a
# devel base). Then run the benchmark via port-forward during the capture window.
#
# ALTERNATIVE (lighter weight): vLLM has a built-in Torch profiler. Set
#   VLLM_TORCH_PROFILER_DIR=/traces
# on the server, then hit POST /start_profile and POST /stop_profile around a
# benchmark burst. It emits Chrome/Perfetto traces without launch-wrapping — often
# easier than nsys for a first look, though nsys gives deeper CUDA/NCCL detail.

set -euo pipefail

# MODEL is required — via env or first positional arg. No default, since this is a
# general-purpose profiler that should run against whatever model you point it at.
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

# -t cuda,nvtx,osrt,cudnn,cublas: trace CUDA runtime, NVTX ranges (NCCL emits these),
# OS runtime, and the math libs. --delay/--duration bound the window to steady state.
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
