# Qwen3-30B-A3B FP8 on g7e.12xlarge, TP2 with expert parallel, explicit PIX peer-to-peer

Serves [`Qwen/Qwen3-30B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8) across both GPUs of one g7e.12xlarge (2x RTX PRO 6000 Blackwell, sm_120, PCIe, no NVLink) with tensor and expert parallelism, using [`fp8-g7e.12xlarge-switch-pix.yaml`](../../../vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-pix.yaml). It sets `NCCL_P2P_LEVEL=PIX`.

## Capacity sizing

Qwen3-30B-A3B is a mixture-of-experts model. Each of its 48 layers holds 128 experts and routes every token to 8 of them, so about 3.3B of its 30.5B parameters run per token. The FP8 checkpoint is about 31 GiB and would fit one 96 GB card; this recipe spans both GPUs because the study measures the transport between them. The KV cache is BF16 at 96 KiB per token across 48 layers, 48 KiB per GPU under TP2, so a full 8,192-token sequence costs 0.75 GiB.

`legacy_corpus` sends a median of about 17 wire input tokens and asks for up to 150 output tokens, so 512 concurrent sequences hold roughly 85k tokens, under 8 GiB of KV. `--max-num-seqs 512`, not KV memory, caps the batch for this workload. Read `KV cache size` in the result file's provenance block for the allocation in force.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 30.5B total, 3.3B active per token | Model card |
| **Layers** | 48 | `config.json`, `num_hidden_layers` |
| **Experts** | 128 per layer, 8 routed per token | `config.json`, `num_experts`, `num_experts_per_tok` |
| **KV heads, head dimension** | 4, 128 | `config.json`, `num_key_value_heads`, `head_dim` |
| **KV cache per token, BF16** | 96 KiB, 48 KiB per GPU | `2 x 4 x 128 x 2 bytes x 48 layers`, halved by TP2 |
| **Checkpoint size, FP8** | ~31 GiB, 7 safetensors shards | `fetch-weights` init container checks |
| **Instance** | g7e.12xlarge, 2x RTX PRO 6000 Blackwell at 96 GB, PCIe, no NVLink | Instance type; `nvidia-smi topo -m` in the server log |
| **GPUs per pod** | 2 | `nvidia.com/gpu: "2"` |
| **Parallelism** | TP2 with expert parallel, `allgather_reducescatter` | `--tensor-parallel-size`, `--enable-expert-parallel`, `--all2all-backend` |
| **Weight precision** | FP8 E4M3, 128x128 blocks | Checkpoint `quantization_config`; the recipe passes no `--quantization` |
| **KV cache precision** | BF16 | `--kv-cache-dtype` |
| **NCCL transport** | P2P allowed up to `PIX`, a shared PCIe switch | `NCCL_P2P_LEVEL` |
| **Context length** | 8,192 | `--max-model-len` |
| **Max sequences** | 512 | `--max-num-seqs` |
| **Batched tokens per step** | 8,192 | `--max-num-batched-tokens` |
| **Memory fraction** | 0.90 | `--gpu-memory-utilization` |
| **Image** | `vllm/vllm-openai:v0.26.0` | `image` |

## Before you begin

Create the following before you start.

- [Create a `g7e.12xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml) (set the instance type; the node must be idle)
- [Mount its instance store as RAID0 for kubelet storage](../../../k8s/examples/nodeconfig-local-nvme-raid0.yaml) (the pod stages ~31 GiB)
- [Rebuild the NVIDIA open kernel module if the node advertises no GPUs](../../../k8s/examples/job-node-driver-fix.yaml)

## Quickstart

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen3-30b-a3b-fp8-g7e-12xlarge-switch
   export INSTANCE_TYPE=g7e.12xlarge
   export GPU_NODE=<your-g7e-12xlarge-node-name>
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

4. Install the Pod Identity Agent and grant the `qwen3-30b-a3b-bench` ServiceAccount S3 access.
   ```bash
   ./scripts/setup-pod-identity.sh qwen3-30b-a3b-bench
   ```

5. Copy the published checkpoint from Hugging Face to your S3 bucket.
   ```bash
   ./model-jobs/qwen3-30b-a3b/fp8-model-download.sh
   ```

## Throughput test

1. Deploy the inference server on `GPU_NODE` and confirm it serves the model.
   ```bash
   ./vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-pix-deploy.sh
   ```

2. Confirm `/models` is on the NVMe RAID0 array, not the EBS root volume.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec deployment/qwen3-30b-a3b-g7e12-switch-pix \
       --container vllm -- df -h /models
   ```

3. Confirm `NCCL_P2P_LEVEL=PIX` is set and NCCL chose P2P, not SHM.
   ```bash
   SERVER_LOG=$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs \
       deployment/qwen3-30b-a3b-g7e12-switch-pix -c vllm)
   grep -E 'NCCL transport override|^NCCL_P2P_(LEVEL|DISABLE)=' <<<"$SERVER_LOG"
   grep -cE 'P2P/direct pointer|via P2P/' <<<"$SERVER_LOG"
   grep -cE 'via SHM/' <<<"$SERVER_LOG"
   ```

4. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service qwen3-30b-a3b-g7e12-switch-pix \
       --mode fingerprint \
       --model-tag qwen3-30b-a3b --config fp8-switch-pix \
       --model-id qwen3-30b-a3b-fp8 \
       --thinking off \
       --out-dir docs/results
   ```

5. Dry-run the exact in-cluster benchmark.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service qwen3-30b-a3b-g7e12-switch-pix \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen3-30b-a3b \
       --config fp8-switch-pix --quant fp8 \
       --model-id qwen3-30b-a3b-fp8 \
       --tokenizer Qwen/Qwen3-30B-A3B-FP8 \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=qwen3-30b-a3b-g7e12-switch-pix -o jsonpath='{.items[0].metadata.name}')" \
       --out-dir docs/results \
       --reps 5 \
       --dry-run
   ```

6. Warm up the server with one unrecorded benchmark run, saved outside `docs/results`.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service qwen3-30b-a3b-g7e12-switch-pix \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen3-30b-a3b \
       --config fp8-switch-pix --quant fp8 \
       --model-id qwen3-30b-a3b-fp8 \
       --tokenizer Qwen/Qwen3-30B-A3B-FP8 \
       --skip-accuracy --no-latency --no-provenance \
       --out-dir /tmp/qwen3-30b-a3b-switch-warmup/fp8-switch-pix \
       --reps 1
   ```

7. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service qwen3-30b-a3b-g7e12-switch-pix \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen3-30b-a3b \
       --config fp8-switch-pix --quant fp8 \
       --model-id qwen3-30b-a3b-fp8 \
       --tokenizer Qwen/Qwen3-30B-A3B-FP8 \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=qwen3-30b-a3b-g7e12-switch-pix -o jsonpath='{.items[0].metadata.name}')" \
       --out-dir docs/results \
       --reps 5
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-pix.yaml --ignore-not-found
```

Once both arms are measured, scale the node groups to zero. The checkpoint in S3 survives, so a later run skips Quickstart step 5.

```bash
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name gpu-spot-1node \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```
