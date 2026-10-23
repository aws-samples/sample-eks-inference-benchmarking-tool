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

- [Create an EKS cluster with Karpenter, Karpenter discovery tags and node role](../../../scripts/create-cluster.sh)
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

Deleting the server leaves the node empty, and Karpenter scales it to zero. The checkpoint in S3 survives, so a later run skips Quickstart step 6.

---

## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **This is the latency recipe; bf16seq512 is the throughput recipe.** MTP cuts single-request latency to about 3.5 ms per output token and 4.0 s end to end at concurrency 1, roughly a third of the non-speculative figure, but its small KV pool costs aggregate throughput under load. For the highest tokens/sec, serve the [throughput arm](./nvfp4-p6-b200.48xlarge-fp8kv-32k-dp2tp4-bf16seq512-flashinfer-ep-runbook.md). Read the latency table in the result, not the throughput table.
- **MTP is the A/B variable, proven active by metrics.** `--speculative-config '{"method": "mtp", "num_speculative_tokens": 5}'` runs the checkpoint's MTP layer as a draft. The deploy script drives one request and then fails unless `vllm:spec_decode_num_accepted_tokens` is present, so a silently ignored config cannot read as "MTP did not help."
- **Capture acceptance as a counter delta.** `vllm:spec_decode_*` are cumulative and never reset, so snapshot them before and after the benchmark and take the difference. This run accepted about 65% of drafted tokens across the five positions.
- **Memory fraction is 0.80, not 0.90.** At 0.90 the KV pool claimed the headroom MTP's draft pass and lookahead buffers need, and warmup aborted trying to allocate ~15 GiB that was not reserved. Lowering it to 0.80 leaves that headroom; the KV pool settles near 2,629,632 tokens per replica (~80x concurrency). Re-read `KV capacity from` in the run you publish.
- **Warmup outlasts the default engine-ready timeout.** MTP compiles a draft pass on top of the target, so the recipe sets `VLLM_ENGINE_READY_TIMEOUT_S=1800`; the 600 s default times out.
- **More speculative tokens did not beat one here.** Five tokens gave about the same single-stream latency as one token on this model, because acceptance falls off at later positions while the draft cost grows. Five is kept to match NVIDIA's Dynamo recipe; one is a lighter alternative with nearly the same latency and a larger KV pool.
- **Two data-parallel replicas of four GPUs each.** `--data-parallel-size 2` runs two independent engines; `--tensor-parallel-size 4` shards the weights across the four GPUs of each. The server prints `nvidia-smi topo -m` at start and warns when it finds no NVLink pairs; do not trust a run that logged that warning.
- **Expert parallelism is on.** `--enable-expert-parallel` places whole experts on individual ranks within each replica. The server's provenance block records `expert parallel on`; confirm it before attributing a result.
- **The backend is pinned with `--attention-backend`, not `VLLM_ATTENTION_BACKEND`.** vLLM 0.29.0 does not read that environment variable. The deploy script fails unless the server logs `Using AttentionBackendEnum.FLASHINFER backend`.
- **NVFP4 on Blackwell is not comparable to a Hopper run.** On sm_100 the checkpoint runs the native cutlass W4A4 GEMM path. A Hopper GPU would fall back to FP4-Marlin W4A16, so never read this recipe's results against a Hopper one.
- **No `--quantization` flag.** The checkpoint's `quantization_config` declares a ModelOpt mixed NVFP4 and FP8 export, and the loader resolves it. Pinning a single method would mis-read the mixed export.
- **`--stream-interval 10`** The server streams in 10-token chunks. That changes inter-token latency only; never compare ITL against a recipe with a different stream interval.
- **`--no-enable-prefix-caching`** Held off for an uncached measurement. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`--async-scheduling` and `--enable-chunked-prefill`** Both on. Async scheduling is compatible with MTP speculative decoding on vLLM 0.29.0.
- **`MAX_NUM_SEQS=256`, `MAX_CUDAGRAPH=256` and `MAX_NUM_BATCHED_TOKENS=32768`.** The CUDA graph capture size matches the sequence cap. A 32,768-token batch budget prefills a full `coding_repo_context` prompt in one chunk.
- **No reasoning parser.** The recipe sets no `--reasoning-parser`, so any reasoning trace arrives in the message content and counts against the profile's 1,024-token output cap. The corpus protocol truncates a reasoning model's output by design.
- **`--served-model-name nemotron-3-ultra`** The server answers to that name rather than its path, so `--model-id` is `nemotron-3-ultra`.
- **Weights come from S3 into an `emptyDir` on the NVMe array.** A pod restart re-downloads ~328 GiB. The `fetch-weights` init container fails unless `/models` has 400 GiB free, then fails unless it finds 113 shards and at least 300 GiB.
- **`progressDeadlineSeconds: 7200`** The S3 sync and load outlast the 600-second default. The deploy script waits the same 7,200 seconds, and the startup probe allows 3,600 seconds once the init container finishes.
- **`serviceAccountName: nemotron-3-ultra-bench`** Supplies the pod-scoped S3 identity through EKS Pod Identity. The staging Job runs as the same ServiceAccount. Run `setup-pod-identity.sh nemotron-3-ultra-bench` in this namespace.
