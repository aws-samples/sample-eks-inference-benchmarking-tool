# Nemotron 3 Ultra NVFP4 with an FP8 KV cache on p6-b200.48xlarge, FlashInfer attention, data-parallel attention over expert parallel, MTP speculative decoding

Serves NVIDIA's published NVFP4 checkpoint [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) across all eight GPUs of a p6-b200.48xlarge as two data-parallel replicas, each a four-way tensor-parallel shard, with expert parallelism and native MTP speculative decoding proposing five tokens, using [`nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-mtp5.yaml`](../../../vllm-recipes/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-mtp5.yaml). This is the latency-minimizing recipe: on `corpus`/`coding_repo_context` a single request streamed at about 3.5 ms per output token and 4.0 s end to end, roughly three times faster than without speculation. It gives up aggregate throughput under load; for the highest tokens/sec, use the [throughput recipe](./nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-bf16seq512-flashinfer-ep-runbook.md) instead.

## Capacity sizing

Nemotron 3 Ultra ships a Multi-Token Prediction (MTP) layer (`config.json`, `num_nextn_predict_layers: 1`). Speculative decoding runs that layer as a draft, proposing five tokens per step that the target model then verifies in one pass, so an accepted run returns several tokens for the cost of one forward pass. Over this benchmark 65% of drafted tokens were accepted across the five positions, which is what cuts per-token latency.

Speculation is not free on memory. The draft pass and the five-token lookahead buffers claim space the KV pool would otherwise take, so the pool shrank to 2.6M tokens per replica (~80x concurrency) from the 15.7M of the same DP2xTP4 layout without speculation. GPU memory utilization is held at 0.80, not 0.90, because at 0.90 the KV pool consumed the headroom the MTP buffers needed and warmup aborted with an out-of-memory error. The small pool is why throughput falls off under heavy concurrency; the win is single-request latency, so read the latency table in the result, not the throughput table.

`coding_repo_context` sends about 18.8k wire input tokens with this tokenizer and asks for up to 1,024 output tokens.

| Property | Value | Source |
|---|---|---|
| **Parameters** | ~550B total, ~55B active per token | Checkpoint name, `550B-A55B` |
| **Layers** | 108: 48 Mamba-2, 48 MoE, 12 attention | `config.json`, `layers_block_type` |
| **Experts** | 512 routed, 22 per token, 1 shared | `config.json`, `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts` |
| **MTP** | 1 layer, 5 speculative tokens per step | `config.json`, `num_nextn_predict_layers`; `--speculative-config` |
| **KV heads, head dimension** | 2, 128 | `config.json`, `num_key_value_heads`, `head_dim` |
| **KV cache per token, FP8** | 6 KiB | `2 x 2 x 128 x 1 byte x 12 attention layers` |
| **Mamba state per sequence** | ~0.38 GiB, float32 | `256 heads x 64 x 128 state x 4 bytes x 48 layers`; `mamba_ssm_cache_dtype` |
| **Checkpoint size, NVFP4** | ~328 GiB, 113 safetensors shards | `fetch-weights` init container checks |
| **Instance** | p6-b200.48xlarge, 8x B200 at 180 GB, sm_100, NVLink | `nvidia.com/gpu: "8"` |
| **Data parallel size** | 2 replicas | `--data-parallel-size 2` |
| **Tensor parallel size** | 4 per replica | `--tensor-parallel-size 4` |
| **Expert parallelism** | enabled | `--enable-expert-parallel` |
| **Speculative decoding** | MTP, 5 tokens, ~65% acceptance | `--speculative-config`; `vllm:spec_decode_*` delta |
| **KV pool** | 2,629,632 tokens per replica, ~80x concurrency | Result provenance, `KV capacity from` |
| **Weight precision** | ModelOpt mixed NVFP4 and FP8, 243 modules kept high precision | Checkpoint `quantization_config`; no `--quantization` flag |
| **KV cache precision** | FP8 | `KV_CACHE_DTYPE` |
| **Attention backend** | `FLASHINFER` | `ATTENTION_BACKEND` |
| **Context length** | 32,768 of a native 262,144 | `MAX_MODEL_LEN`; `config.json`, `max_position_embeddings` |
| **Max sequences** | 256 | `MAX_NUM_SEQS` |
| **Memory fraction** | 0.80 | `GPU_MEM_UTIL` |
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
   export NAMESPACE=vb-nemotron-3-ultra-dp2tp4-mtp5-p6-b200-48xlarge
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

1. Deploy the inference server and confirm it serves the model. The deploy fails closed unless `vllm:spec_decode_*` metrics appear, proving MTP is active.
   ```bash
   ./vllm-recipes/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-mtp5-deploy.sh
   ```

2. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service nemotron-3-ultra-vllm-b200-fp8kv-32k-flashinfer-ep-dp2tp4-mtp5 \
       --mode fingerprint \
       --model-tag nemotron-3-ultra --config fp8kv-32k-dp2tp4-flashinfer-ep-mtp5 \
       --model-id nemotron-3-ultra \
       --out-dir docs/results
   ```

3. Dry-run the exact in-cluster benchmark.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service nemotron-3-ultra-vllm-b200-fp8kv-32k-flashinfer-ep-dp2tp4-mtp5 \
       --protocol corpus --profile coding_repo_context \
       --instance p6-b200.48xlarge --model-tag nemotron-3-ultra \
       --config fp8kv-32k-dp2tp4-flashinfer-ep-mtp5 --quant nvfp4 \
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
       --service nemotron-3-ultra-vllm-b200-fp8kv-32k-flashinfer-ep-dp2tp4-mtp5 \
       --protocol corpus --profile coding_repo_context \
       --instance p6-b200.48xlarge --model-tag nemotron-3-ultra \
       --config fp8kv-32k-dp2tp4-flashinfer-ep-mtp5 --quant nvfp4 \
       --model-id nemotron-3-ultra \
       --tokenizer nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-ep-dp2tp4-mtp5.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    job/nemotron-3-ultra-nvfp4-hf-to-s3 --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-cb-p6-b200-48xlarge ec2nodeclass/bench-cb-p6-b200-48xlarge --ignore-not-found
```

Deleting the server leaves the node empty, and Karpenter scales it to zero. The checkpoint in S3 survives, so a later run skips weights staging in previous steps.
