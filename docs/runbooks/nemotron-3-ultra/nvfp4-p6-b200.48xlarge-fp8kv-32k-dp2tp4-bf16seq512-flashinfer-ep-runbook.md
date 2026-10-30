# Nemotron 3 Ultra NVFP4 with an FP8 KV cache on p6-b200.48xlarge, FlashInfer attention, data-parallel attention over expert parallel, bfloat16 Mamba cache

Serves NVIDIA's published NVFP4 checkpoint [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) across all eight GPUs of a p6-b200.48xlarge as two data-parallel replicas, each a four-way tensor-parallel shard, with expert parallelism, a bfloat16 Mamba SSM state cache, and a 512-sequence concurrency ceiling, using [`nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-bf16seq512.yaml`](../../../vllm-recipes/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-bf16seq512.yaml). This is the throughput-maximizing recipe: on `corpus`/`coding_repo_context` it sustained about 4,276 tokens/sec at concurrency 512, the highest of the configurations tested. For the lowest single-request latency instead, use the [MTP recipe](./nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-mtp5-flashinfer-ep-runbook.md).

## Capacity sizing

Nemotron 3 Ultra is a hybrid of Mamba-2 and mixture-of-experts layers with 512 routed experts, 22 active per token. This recipe runs two independent replicas of four GPUs each: each replica shards the weights four ways and owns its own KV pool, and the front end load-balances requests across the two. Aggregate KV capacity and the number of concurrent sequences the node sustains, not single-stream speed, are what this configuration maximizes.

Two changes versus a plain DP2xTP4 configuration (float32 Mamba cache, 256 sequences, which this recipe supersedes) work together as one throughput lever. The Mamba SSM state cache drops from float32 to bfloat16 (`--mamba-ssm-cache-dtype bfloat16`), halving this model's ~0.38 GiB-per-sequence Mamba state, and that freed memory funds doubling `--max-num-seqs` from 256 to 512. The larger concurrent-batch ceiling is where the extra throughput comes from. The measured effect was modest here, about +5% at concurrency 512, because prompt compute rather than KV memory is the limit for this workload; the KV pool grew from 15.7M to 20.4M tokens per replica even so.

`coding_repo_context` sends about 18.8k wire input tokens with this tokenizer and asks for up to 1,024 output tokens. Prompt processing through ~55B active parameters per token, not KV memory, is the limit for this workload.

| Property | Value | Source |
|---|---|---|
| **Parameters** | ~550B total, ~55B active per token | Checkpoint name, `550B-A55B` |
| **Layers** | 108: 48 Mamba-2, 48 MoE, 12 attention | `config.json`, `layers_block_type` |
| **Experts** | 512 routed, 22 per token, 1 shared | `config.json`, `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts` |
| **KV heads, head dimension** | 2, 128 | `config.json`, `num_key_value_heads`, `head_dim` |
| **KV cache per token, FP8** | 6 KiB | `2 x 2 x 128 x 1 byte x 12 attention layers` |
| **Mamba state per sequence** | ~0.19 GiB, bfloat16 | `256 heads x 64 x 128 state x 2 bytes x 48 layers`; `mamba_ssm_cache_dtype` |
| **Checkpoint size, NVFP4** | ~328 GiB, 113 safetensors shards | `fetch-weights` init container checks |
| **Instance** | p6-b200.48xlarge, 8x B200 at 180 GB, sm_100, NVLink | `nvidia.com/gpu: "8"` |
| **Data parallel size** | 2 replicas | `--data-parallel-size 2` |
| **Tensor parallel size** | 4 per replica | `--tensor-parallel-size 4` |
| **Expert parallelism** | enabled | `--enable-expert-parallel` |
| **Mamba SSM cache precision** | bfloat16 | `--mamba-ssm-cache-dtype` |
| **KV pool** | 20,368,042 tokens per replica, ~621x concurrency | Result provenance, `KV capacity from` |
| **Weight precision** | ModelOpt mixed NVFP4 and FP8, 243 modules kept high precision | Checkpoint `quantization_config`; no `--quantization` flag |
| **KV cache precision** | FP8 | `KV_CACHE_DTYPE` |
| **Attention backend** | `FLASHINFER` | `ATTENTION_BACKEND` |
| **Context length** | 32,768 of a native 262,144 | `MAX_MODEL_LEN`; `config.json`, `max_position_embeddings` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Stream interval** | 10 tokens | `STREAM_INTERVAL` |

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create an EKS cluster with Karpenter with `./scripts/create-cluster.sh`](../../../scripts/create-cluster.sh)
- [Purchase a p6-b200.48xlarge Capacity Block for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html) 

## Quickstart

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-nemotron-3-ultra-dp2tp4-bf16seq512-p6-b200-48xlarge
   export INSTANCE_TYPE=p6-b200.48xlarge
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

5. Install the Pod Identity Agent and grant the `nemotron-3-ultra-bench` ServiceAccount S3 access.
   ```bash
   ./scripts/setup-pod-identity.sh nemotron-3-ultra-bench
   ```

6. Copy the published checkpoint from Hugging Face to your S3 bucket.
   ```bash
   ./model-jobs/nemotron-3-ultra/nvfp4-model-download.sh
   ```

## Throughput test

1. Deploy the inference server and confirm it serves the model.
   ```bash
   ./vllm-recipes/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-bf16seq512-deploy.sh
   ```

2. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service nemotron-3-ultra-vllm-b200-fp8kv-32k-dp2tp4-bf16seq512 \
       --mode fingerprint \
       --model-tag nemotron-3-ultra --config fp8kv-32k-dp2tp4-flashinfer-ep-bf16seq512 \
       --model-id nemotron-3-ultra \
       --out-dir docs/results
   ```

3. Dry-run the exact in-cluster benchmark.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service nemotron-3-ultra-vllm-b200-fp8kv-32k-dp2tp4-bf16seq512 \
       --protocol corpus --profile coding_repo_context \
       --instance p6-b200.48xlarge --model-tag nemotron-3-ultra \
       --config fp8kv-32k-dp2tp4-flashinfer-ep-bf16seq512 --quant nvfp4 \
       --model-id nemotron-3-ultra \
       --tokenizer nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
       --out-dir docs/results \
       --reps 3 \
       --dry-run
   ```

4. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service nemotron-3-ultra-vllm-b200-fp8kv-32k-dp2tp4-bf16seq512 \
       --protocol corpus --profile coding_repo_context \
       --instance p6-b200.48xlarge --model-tag nemotron-3-ultra \
       --config fp8kv-32k-dp2tp4-flashinfer-ep-bf16seq512 --quant nvfp4 \
       --model-id nemotron-3-ultra \
       --tokenizer nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-bf16seq512.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    job/nemotron-3-ultra-nvfp4-hf-to-s3 --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-cb-p6-b200-48xlarge ec2nodeclass/bench-cb-p6-b200-48xlarge --ignore-not-found
```

Deleting the server leaves the node empty, and Karpenter scales it to zero. The checkpoint in S3 survives, so a later run skips weights staging in previous steps.
