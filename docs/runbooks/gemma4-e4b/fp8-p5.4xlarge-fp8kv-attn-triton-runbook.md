# Gemma 4 E4B FP8 with an FP8 KV cache on p5.4xlarge, Triton attention

Serves a locally quantized ModelOpt FP8 build of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) on one H100 with an FP8 KV cache and the attention backend pinned to `TRITON_ATTN`, using [`fp8-p5.4xlarge-fp8kv-attn-triton.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton.yaml).

## Capacity sizing

The recipe runs at `--gpu-memory-utilization 0.90` on an 80 GB card. Read `KV capacity from` and `KV full-ctx seqs` in the result file's provenance block for the allocation in force.

The pool is not the constraint. `chat_terse` sends 96 wire input tokens with a 30-token output cap, so a sequence occupies about 126 tokens and the working set at concurrency 512 is a small fraction of the pool. The backend is the kernel doing the attention arithmetic, and with an FP8 KV cache that arithmetic runs in FP8, which puts the backend on the critical path.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 4B active, 8B on disk | Model card |
| **Layers** | 42, of which 7 full attention and 35 sliding at window 512 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 2; 256 sliding, 512 full attention | `config.json`, `head_dim` and `global_head_dim` |
| **Checkpoint size, FP8** | 11.09 GiB, one safetensors file | quantize init container output |
| **Instance** | p5.4xlarge, 1x H100 at 80 GB, sm_90 | Instance type |
| **Weight precision** | FP8 E4M3, static input scales | `hf_quant_config.json`, `quant_algo: FP8` |
| **KV cache precision** | FP8 E4M3, uncalibrated scales | `KV_CACHE_DTYPE=fp8`; see Settings |
| **Attention backend** | `TRITON_ATTN` | `ATTENTION_BACKEND` |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Max sequences** | 1,024 | `MAX_NUM_SEQS` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Stream interval** | 10 tokens | `STREAM_INTERVAL` |

## Before you begin

Complete the following tasks before you perform the steps in this runbook.

- [Create an EKS cluster with Karpenter, Karpenter discovery tags and node role](../../../scripts/create-cluster.sh)
- [Purchase a p5.4xlarge Capacity Block for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)

## Quickstart

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export AWS_ACCOUNT_ID=<your-account-id>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-e4b-fp8kv-triton-p5-4xlarge
   export INSTANCE_TYPE=p5.4xlarge
   export RESERVATION_ID=<your-capacity-block-id>
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

4. Provision a Capacity Block GPU node pool with NVMe RAID0 for the instance type.
   ```bash
   ./scripts/provision-capacity-block.sh "$INSTANCE_TYPE" "$RESERVATION_ID"
   ```

5. Build the quantizer tool image in-cluster with kaniko, create the ECR repository, and push it. This step takes several minutes to complete. Alternatively, to build locally on x86 Docker, run `./scripts/push-image.sh`.
   ```bash
   ./scripts/push-image-kaniko.sh
   ```

## Throughput test

1. Deploy the inference server, whose init container quantizes Gemma 4 E4B to FP8 on NVMe on first start, and confirm it serves the model.
   ```bash
   ./vllm-recipes/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton-deploy.sh
   ```

2. Confirm `/models` is on the NVMe RAID0 array, not the EBS root volume.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec \
       deployment/gemma4-e4b-vllm-p5-4xl-fp8kv-triton -c vllm -- df -h /models
   ```

3. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
       --mode fingerprint \
       --model-tag gemma4-e4b --config p5-fp8kv-attn-triton \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

4. Dry-run the exact in-cluster benchmark.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
       --protocol corpus --profile chat_terse \
       --instance p5.4xlarge --model-tag gemma4-e4b \
       --config p5-fp8kv-attn-triton --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --out-dir docs/results \
       --reps 3 \
       --dry-run
   ```

5. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
       --protocol corpus --profile chat_terse \
       --instance p5.4xlarge --model-tag gemma4-e4b \
       --config p5-fp8kv-attn-triton --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-cb-p5-4xlarge ec2nodeclass/bench-cb-p5-4xlarge --ignore-not-found
```

Deleting the server leaves the node empty, and Karpenter scales it to zero.
