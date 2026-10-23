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

- [Create an EKS cluster with Karpenter, Karpenter discovery tags and node role](../../../scripts/create-cluster.sh)
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

Deleting the server leaves the node empty, and Karpenter scales it to zero. The checkpoint in S3 survives, so a later run skips Quickstart step 6.

---

## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **This is the throughput recipe; MTP is the latency recipe.** It maximizes aggregate tokens/sec under load. For the lowest single-request latency, serve the [MTP arm](./nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-mtp5-flashinfer-ep-runbook.md) instead, which trades peak throughput for a roughly threefold lower per-token latency.
- **bfloat16 Mamba SSM cache funds the higher sequence ceiling.** `--mamba-ssm-cache-dtype bfloat16` halves the per-sequence Mamba state versus the float32 default, and the freed memory is spent on `--max-num-seqs 512`. The two move together: raising the sequence cap without the memory saving would not fit. The deploy script prints `mamba_ssm_cache_dtype` from the live metrics; confirm it reads `bfloat16`.
- **The gain is modest because the workload is compute-bound.** Doubling the concurrent-batch ceiling lifted throughput only about 5% at concurrency 512, since prompt processing through ~55B active parameters, not KV memory, is the limit. The KV pool nonetheless grew to 20,368,042 tokens per replica (~621x concurrency); re-read `KV capacity from` in the run you publish.
- **Two data-parallel replicas of four GPUs each.** `--data-parallel-size 2` runs two independent engines; `--tensor-parallel-size 4` shards the weights across the four GPUs of each. The server prints `nvidia-smi topo -m` at start and warns when it finds no NVLink pairs; do not trust a run that logged that warning.
- **Four-way tensor parallelism is what makes this fit.** Pure data parallelism (`--data-parallel-size 8`) replicates the full ~328 GiB checkpoint on each GPU and runs out of memory during KV allocation on B200. Sharding each replica four ways keeps the per-GPU weight footprint near 51 GiB.
- **Expert parallelism is on.** `--enable-expert-parallel` places whole experts on individual ranks within each replica. The server's provenance block records `expert parallel on`; confirm it before attributing a result.
- **The backend is pinned with `--attention-backend`, not `VLLM_ATTENTION_BACKEND`.** vLLM 0.29.0 does not read that environment variable. The deploy script fails unless the server logs `Using AttentionBackendEnum.FLASHINFER backend`.
- **NVFP4 on Blackwell is not comparable to a Hopper run.** On sm_100 the checkpoint runs the native cutlass W4A4 GEMM path. A Hopper GPU would fall back to FP4-Marlin W4A16, so never read this recipe's results against a Hopper one.
- **No `--quantization` flag.** The checkpoint's `quantization_config` declares a ModelOpt mixed NVFP4 and FP8 export, and the loader resolves it. Pinning a single method would mis-read the mixed export.
- **`--stream-interval 10`** The server streams in 10-token chunks. That changes inter-token latency only; never compare ITL against a recipe with a different stream interval.
- **`--no-enable-prefix-caching`** Held off for an uncached measurement. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`--async-scheduling` and `--enable-chunked-prefill`** Both on.
- **`MAX_NUM_SEQS=512`, `MAX_CUDAGRAPH=512` and `MAX_NUM_BATCHED_TOKENS=32768`.** The CUDA graph capture size matches the sequence cap. A 32,768-token batch budget prefills a full `coding_repo_context` prompt in one chunk.
- **No reasoning parser.** The recipe sets no `--reasoning-parser`, so any reasoning trace arrives in the message content and counts against the profile's 1,024-token output cap. The corpus protocol truncates a reasoning model's output by design.
- **Speculative decoding is left off.** This is the throughput arm; MTP is measured on its own.
- **`--served-model-name nemotron-3-ultra`** The server answers to that name rather than its path, so `--model-id` is `nemotron-3-ultra`.
- **Weights come from S3 into an `emptyDir` on the NVMe array.** A pod restart re-downloads ~328 GiB. The `fetch-weights` init container fails unless `/models` has 400 GiB free, then fails unless it finds 113 shards and at least 300 GiB.
- **`progressDeadlineSeconds: 7200`** The S3 sync and load outlast the 600-second default. The deploy script waits the same 7,200 seconds, and the startup probe allows 3,600 seconds once the init container finishes.
- **`serviceAccountName: nemotron-3-ultra-bench`** Supplies the pod-scoped S3 identity through EKS Pod Identity. The staging Job runs as the same ServiceAccount. Run `setup-pod-identity.sh nemotron-3-ultra-bench` in this namespace.
