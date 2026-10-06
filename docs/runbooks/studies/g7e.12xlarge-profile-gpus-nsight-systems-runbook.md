# Profile GPUs with Nsight Systems

Captures an Nsight Systems trace of a vLLM server to show how step time splits across GEMM, the NCCL collective, attention, softmax, sampling and scheduling gaps. Use it when a sweep shows a ceiling that the metrics do not explain.

## Before you begin

Complete the following before you start. `nsys` must be the parent process of `vllm serve`, so this runs as its own pod rather than as a change to an existing deployment.

- Install the AWS CLI, `kubectl`, and Helm locally
- Provide a `g7e.12xlarge` GPU node, which has enough GPUs for the TP=2 collective measured here
- Provide a CPU system node labeled `role=system` for the image build
- [Review the pod's capabilities](../../../k8s/examples/pod-profile-nsys.yaml) (`SYS_ADMIN` for CPU sampling, `IPC_LOCK` for pinned NCCL memory, both already in the manifest)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-nsys-qwen25-7b-g7e-12xlarge
   export INSTANCE_TYPE=g7e.12xlarge
   ```

2. Install the `eks-bench` CLI into a Python virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e '.[tokenizer]'
   ```

3. Configure the EKS context and namespace, install the NVIDIA device plugin, and create the `hf-token` secret.
   ```bash
   ./scripts/install-prerequisites.sh
   ```

4. Build a vLLM image with Nsight Systems and push it to ECR.

   ```bash
   export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
   export ECR_REPOSITORY="vllm-nsys"
   export ECR_CACHE_REPOSITORY="vllm-nsys-cache"
   export IMAGE_TAG="qwen25-7b-nsys"
   export NSYS_IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG"
   export NSYS_CACHE_REPO="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_CACHE_REPOSITORY"

   for repository in "$ECR_REPOSITORY" "$ECR_CACHE_REPOSITORY"; do
     aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$repository" >/dev/null 2>&1 \
       || aws ecr create-repository --region "$AWS_REGION" --repository-name "$repository" >/dev/null
   done

   cp k8s/examples/job-build-image-kaniko.yaml my-nsys-image-build.yaml
   python3 - "$NSYS_IMAGE" "$NSYS_CACHE_REPO" <<'PY'
   from pathlib import Path
   import sys

   manifest_path = Path("my-nsys-image-build.yaml")
   dockerfile = Path("app/vllm-nsys/Dockerfile").read_text()
   manifest = manifest_path.read_text()

   block_start = manifest.index("  Dockerfile: |\n") + len("  Dockerfile: |\n")
   block_end = manifest.index("\n---\n", block_start)
   dockerfile_block = "".join(f"    {line}\n" for line in dockerfile.splitlines())
   manifest = manifest[:block_start] + dockerfile_block + manifest[block_end:]

   replacements = {
       "012345678901.dkr.ecr.us-east-1.amazonaws.com/my-repo:my-tag": sys.argv[1],
       "012345678901.dkr.ecr.us-east-1.amazonaws.com/my-repo-cache": sys.argv[2],
   }
   for old, new in replacements.items():
       if old not in manifest:
           raise SystemExit(f"expected placeholder not found: {old}")
       manifest = manifest.replace(old, new)
   manifest_path.write_text(manifest)
   PY

   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete job build-image --ignore-not-found
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f my-nsys-image-build.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
     --for=condition=complete job/build-image --timeout=1800s
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs job/build-image
   ```

5. Launch the vLLM server under `nsys` with a delayed capture window.

   ```bash
   cp k8s/examples/pod-profile-nsys.yaml my-nsys-profile.yaml
   python3 - "$NSYS_IMAGE" <<'PY'
   from pathlib import Path
   import sys

   manifest_path = Path("my-nsys-profile.yaml")
   manifest = manifest_path.read_text()
   placeholder = "012345678901.dkr.ecr.us-east-1.amazonaws.com/vllm-nsys:my-tag"
   if placeholder not in manifest:
       raise SystemExit(f"expected placeholder not found: {placeholder}")
   manifest_path.write_text(manifest.replace(placeholder, sys.argv[1]))
   PY

   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f my-nsys-profile.yaml
   ```

6. Wait for the vLLM server to answer before the capture starts.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs -f pod/profile-vllm -c vllm \
     --pod-running-timeout=10m | grep -m1 "launching vLLM under nsys"

   attempt=0
   until kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec pod/profile-vllm -c vllm -- \
     curl --max-time 2 -fsS localhost:8000/v1/models; do
     attempt=$((attempt + 1))
     if (( attempt >= 36 )); then
       kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs pod/profile-vllm -c vllm --tail=200
       exit 1
     fi
     sleep 10
   done
   ```

7. Send benchmark load from inside the pod until the capture ends.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec pod/profile-vllm -c vllm -- \
     mkdir -p /tmp/vllm-benchmarks/app/bench
   kubectl --context "$EKS_CLUSTER_NAME" cp app/bench/src \
     "${NAMESPACE}/profile-vllm:/tmp/vllm-benchmarks/app/bench" -c vllm
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec pod/profile-vllm -c vllm -- bash -lc '
     set -u
     api=http://127.0.0.1:8000/v1
     cd /tmp/vllm-benchmarks/app/bench/src || exit 1

     VLLM_API_URL="$api" \
     MODEL_ID=Qwen/Qwen2.5-7B-Instruct INSTANCE_TYPE=g7e.12xlarge QUANTIZATION=bf16 \
     CONCURRENCY=32 NUM_REQUESTS=1 MIN_DURATION_S=86400 WARMUP_REQUESTS=0 MAX_TOKENS=512 \
     python3 main.py throughput &
     load_pid=$!
     cleanup() {
       kill "$load_pid" 2>/dev/null || true
       wait "$load_pid" 2>/dev/null || true
     }
     trap cleanup EXIT HUP INT TERM

     misses=0
     while kill -0 "$load_pid" 2>/dev/null; do
       if curl --max-time 2 -fsS "$api/models" >/dev/null; then
         misses=0
       else
         misses=$((misses + 1))
         (( misses >= 3 )) && break
       fi
       sleep 1
     done

     if kill -0 "$load_pid" 2>/dev/null; then
       kill "$load_pid" 2>/dev/null || true
       rc=0
       wait "$load_pid" 2>/dev/null || rc=$?
       trap - EXIT HUP INT TERM
       case "$rc" in
         0|143) exit 0 ;;
         *) exit "$rc" ;;
       esac
     fi

     rc=0
     wait "$load_pid" || rc=$?
     trap - EXIT HUP INT TERM
     exit "$rc"'
   ```

8. Read the `nsys` summary tables from the pod log.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs pod/profile-vllm -c vllm \
     | sed -n '/CAPTURE COMPLETE/,$p'
   ```

9. (Optional) Copy the `.nsys-rep` trace file to open in the Nsight Systems GUI.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" cp \
     "${NAMESPACE}/profile-vllm:/traces/vllm-trace.nsys-rep" \
     "$HOME/vllm-trace.nsys-rep" -c vllm
   ```

## Tear down

The pod sleeps for an hour after capturing, so it holds the GPUs until deleted.

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f my-nsys-profile.yaml
```

## Ad-hoc capture outside Kubernetes

[`app/vllm-nsys/profile.sh`](../../../app/vllm-nsys/profile.sh) launch-wraps `vllm serve` the same way for a container you are already inside, where `nsys` and `vllm` are both on PATH. `MODEL` is required, by env or first argument.

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct TP=1 ./app/vllm-nsys/profile.sh
NSYS_DELAY=900 TP=4 ./app/vllm-nsys/profile.sh /models/qwen3-235b-a22b-fp8
```

Knobs: `TP`, `PORT`, `NSYS_DELAY` (default 180 s, must exceed model load and warmup, raise to roughly 900 for a 235B staged on FSx), `NSYS_DURATION` (default 60 s), `OUT`, `EXTRA_ARGS`. Output is `$OUT.nsys-rep`. Summarise it with `nsys stats <file>.nsys-rep`.

---


## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **The walkthrough model.** Qwen2.5-7B-Instruct in BF16 at TP=2 on one g7e.12xlarge.
- **The image build Job runs as the system-node IAM role.** The nodes labeled `role=system` need ECR authorization, layer upload and `PutImage` on both repositories; the Job configures no ServiceAccount.
- **`my-nsys-image-build.yaml` and `my-nsys-profile.yaml` are your copies.** Steps 4 and 5 write the image name and Dockerfile into them, so the repository templates stay untouched.
- **`NSYS_DELAY` and `NSYS_DURATION`** The capture starts `NSYS_DELAY` seconds after launch and runs for `NSYS_DURATION`. The log line `launching vLLM under nsys` means `nsys` started, not that the capture has. If the server is not ready before the capture is nearly over, raise `NSYS_DELAY` in `my-nsys-profile.yaml` and retry.
- **Load runs inside the pod.** It reaches the server without a port-forward and keeps one concurrency level active until `nsys` stops vLLM. The vLLM image already has `aiohttp`, so nothing is installed.
- **The pod stays up after the capture.** It prints the summary tables after `CAPTURE COMPLETE`, so the findings survive even if you never copy the trace. The `.nsys-rep` file is large and gitignored; keep it outside the repository or in S3.
- **Do not `exec` the `nsys` command.** `nsys` stops the target when the capture closes, and with `exec` the shell is gone, so nothing runs afterwards to preserve the trace. One attempt lost the `.nsys-rep` when the kubelet reclaimed the `emptyDir` on container exit.
- **Do not pipe `nsys stats` into `head`.** SIGPIPE plus `pipefail` kills the script before the remaining reports are written.
- **Keep TP at 2 or more.** The manifest pins TP=2 so the collective exists to be measured.

### What a trace changed

The `NCCL_P2P_LEVEL` study found the knob barely moved end-to-end throughput and described the collective as a small, well-overlapped fraction of step time. A trace contradicted the second half: the collective was a large share of GPU kernel time and the largest share of annotated range time. The knob still does not help, but the stated reason was wrong, and only a trace could show that.

A trace splits GPU kernel time across these categories. For a TP=2 decode-bound server, GEMM and the collective dominate:

```
GEMM (linear / MLP)              collective (NCCL all-gather)
elementwise / memcpy / other     sampling / logits
softmax                          attention
```

The same trace showed a non-trivial share of annotated range time going to cuBLAS algorithm selection rather than math, and confirmed the server was GPU-bound rather than launch-bound. Your own split is what the capture reports.
