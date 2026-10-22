# Gemma 4 E4B FP8 with an FP8 KV cache on g7e.2xlarge, Triton attention

Serves a locally quantized ModelOpt FP8 build of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) with an FP8 KV cache and the attention backend pinned to `TRITON_ATTN`, using [`fp8-g7e.2xlarge-fp8kv-attn-triton.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-triton.yaml).

## Capacity sizing

The recipe runs at `--gpu-memory-utilization 0.90`. Read `KV capacity from` and `KV full-ctx seqs` in the result file's provenance block for the allocation in force.

The pool is not the constraint. `chat_terse` sends 96 wire input tokens with a 30-token output cap, so a sequence occupies about 126 tokens and the working set at concurrency 512 is a small fraction of the pool. What the backend changes is the kernel doing the attention arithmetic, and with an FP8 KV cache this build performs that arithmetic in FP8 rather than converting back to 16-bit first, which puts the backend on the critical path.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 4B active, 8B on disk | Model card |
| **Layers** | 42, of which 7 full attention and 35 sliding at window 512 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 2; 256 sliding, 512 full attention | `config.json`, `head_dim` and `global_head_dim` |
| **Checkpoint size, FP8** | 11.09 GiB, one safetensors file | quantize Job output |
| **Instance** | g7e.2xlarge, 1x RTX PRO 6000 Blackwell at 96 GB, sm_120 | 8 vCPU, 64 GiB host RAM |
| **Weight precision** | FP8 E4M3, static input scales | `hf_quant_config.json`, `quant_algo: FP8` |
| **KV cache precision** | FP8 E4M3, uncalibrated scales | `KV_CACHE_DTYPE=fp8`; see Settings |
| **Attention backend** | `TRITON_ATTN` | `--attention-backend`, from `ATTENTION_BACKEND` |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |

## Before you begin

Complete the following tasks before you perform the steps in this runbook.

- [Create an EKS cluster with Karpenter](../../../scripts/create-cluster.sh) (sets the discovery tags and node role that these steps rely on)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Export the checkpoint to S3](../../../model-jobs/gemma4-e4b/fp8-model-export.yaml) (only when the claim is in a region the checkpoint was not quantized in)
- A Lustre filesystem at version 2.15 if you create one for this (a 2.10 filesystem fails to mount with `mount.lustre ... Invalid argument`)

## Quickstart

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export AWS_ACCOUNT_ID=<your-account-id>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-e4b-fp8kv-triton-g7e-2xlarge
   export INSTANCE_TYPE=g7e.2xlarge
   ```

2. Install the `eks-bench` CLI into a Python virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e '.[tokenizer]'
   ```

3. Configure the EKS context and namespace, install the NVIDIA device plugin, create the `hf-token` secret, and grant ECR push permissions.
   ```bash
   ./scripts/install-prerequisites.sh
   ```

4. Provision a GPU node pool with NVMe RAID0 for the instance type.
   ```bash
   ./scripts/provision-capacity.sh "$INSTANCE_TYPE"
   ```

5. Provision an on-demand GPU node pool with NVMe RAID0 for the quantizer.
   ```bash
   ./scripts/provision-capacity.sh g7.2xlarge
   ```

6. Build the quantizer tool image (the toolbox, not the model) in-cluster with kaniko, which creates the ECR repository and pushes it. Takes several minutes; to build locally on x86 Docker instead, run `./scripts/push-image.sh`.
   ```bash
   ./scripts/push-image-kaniko.sh
   ```

7. Quantize the model to ModelOpt FP8 on the `fsx-pvc` claim, then verify the checkpoint.
   ```bash
   ./model-jobs/gemma4-e4b/fp8-model-quantize.sh
   ```

## Throughput test

1. Deploy the inference server and confirm it serves the model.
   ```bash
   ./vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-triton-deploy.sh
   ```

2. Confirm the KV cache is FP8 and record the KV pool size.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec \
       deployment/gemma4-e4b-vllm-g7e-2xl-fp8kv-triton -c vllm -- \
       sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
   ```

3. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-g7e-2xl-fp8kv-triton \
       --mode fingerprint \
       --model-tag gemma4-e4b --config fp8kv-attn-triton \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

4. Dry-run the exact in-cluster benchmark.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-g7e-2xl-fp8kv-triton \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8kv-attn-triton --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8kv-triton -o jsonpath='{.items[0].metadata.name}')" \
       --out-dir docs/results \
       --reps 3 \
       --dry-run
   ```

5. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-g7e-2xl-fp8kv-triton \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8kv-attn-triton --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8kv-triton -o jsonpath='{.items[0].metadata.name}')" \
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-triton.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-g7e-2xlarge ec2nodeclass/bench-g7e-2xlarge --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-g7-2xlarge ec2nodeclass/bench-g7-2xlarge --ignore-not-found
```

Deleting the server leaves the node empty, and Karpenter scales both the serving and quantize pools to zero. The checkpoint on `fsx-pvc` survives, so a later run needs no requantization.

---

## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **The checkpoint stays on the claim.** A later run can skip Quickstart steps 6 and 7.
- **The backend is pinned with `--attention-backend`, not `VLLM_ATTENTION_BACKEND`.** vLLM 0.29.0 does not read that environment variable. It logs `Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND` and proceeds on its auto-selected backend. The deploy script fails unless the server log reports `TRITON_ATTN`.
- **The backend is pinned rather than auto-selected** because vLLM's auto-priority tables are written for sm_10x B200. This card is sm_120, so leaving it unset makes the backend an unrecorded variable.
- **Why `TRITON_ATTN`.** It is portable across GPU generations and head sizes. `FLASH_ATTN` refuses an FP8 KV cache on this GPU at startup (`FP8 KV cache requires FA3 on SM90 or FA4 on SM100`). `FLASHINFER` serves, but its XQA decode path rejects `head_dim=512` and falls back to a slower native decode for the full-attention layers.
- **KV scales are uncalibrated.** vLLM 0.29.0 rejects `--calculate-kv-scales`, so the server logs `Using KV cache scaling factor 1.0 for fp8_e4m3`. This bounds any quality comparison.
- **`--no-enable-prefix-caching`** Held off for an uncached measurement. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **g7e.2xlarge spot capacity is thin.** A single-subnet node group spent several minutes returning `InsufficientInstanceCapacity`. Check `aws ec2 get-spot-placement-scores` first and consider on-demand if the score is low.
- **The benchmark client runs on the standing `bench-cpu` NodePool.** `measure` runs the client as a Job requesting 8 CPU, pinned off the serving node, so Karpenter launches a CPU node for it and scales it to zero after the run.
