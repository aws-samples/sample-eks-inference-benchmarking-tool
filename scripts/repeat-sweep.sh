#!/bin/bash
# repeat-sweep.sh — run the benchmark sweep N times against localhost:8000 and record
# tokens/sec at c=128 and c=512 per rep. Used for the NCCL_P2P_LEVEL noise study.
# Usage: ./repeat-sweep.sh <MODEL_ID> <LABEL> <NREPS>
MODEL="${1:?model id}"
LABEL="${2:?label}"
N="${3:-5}"
OUT="/tmp/reps_${LABEL}.csv"
echo "rep,c128,c512" > "$OUT"
for i in $(seq 1 "$N"); do
  R=$(VLLM_API_URL=http://localhost:8000/v1 MODEL_ID="$MODEL" INSTANCE_TYPE=g7.24xlarge \
      QUANTIZATION=fp8 BENCH_ENDPOINT=chat NUM_REQUESTS=100 CONCURRENCY_SWEEP=128,512 \
      .benchvenv/bin/python -u app/bench/src/main.py 2>/dev/null)
  C128=$(echo "$R" | awk '$1=="128"{print $2}')
  C512=$(echo "$R" | awk '$1=="512"{print $2}')
  echo "${i},${C128},${C512}" | tee -a "$OUT"
done
echo "DONE_${LABEL}"
