# Nemotron 3 Ultra NVFP4 with an FP8 KV cache on p6-b200.48xlarge, FlashInfer attention

Serves NVIDIA's published NVFP4 checkpoint [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) as one tensor-parallel copy across all eight GPUs of a p6-b200.48xlarge, using [`nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer.yaml`](../../../vllm-recipes/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer.yaml).

## Capacity sizing

Nemotron 3 Ultra is a hybrid of Mamba-2 and mixture-of-experts layers, so most of its per-sequence memory is fixed state rather than KV cache. Only 12 of its 108 layers are attention layers. 2 KV heads at head size 128 across those 12 cost 6 KiB per token at FP8, so a full 32,768-token sequence holds 0.19 GiB of KV. Each of the 48 Mamba-2 layers keeps a float32 SSM state of 8 MiB, which adds about 0.38 GiB per sequence whatever its length. A full-length sequence therefore costs about 0.56 GiB across the eight GPUs, and the 256-sequence cap about 145 GiB. At a 0.90 memory fraction the ~328 GiB checkpoint leaves well over 800 GiB, so the 256-sequence cap, not memory, bounds the batch. vLLM pages attention and Mamba state together, so read `KV capacity from` and `KV full-ctx seqs` in the result file's provenance block for the allocation in force rather than dividing by the per-token figure.

`coding_repo_context` sends about 18.8k wire input tokens with this tokenizer and asks for up to 1,024 output tokens. Prompt processing through ~55B active parameters per token, not KV memory, is the limit for this workload.

| Property | Value | Source |
|---|---|---|
| **Parameters** | ~550B total, ~55B active per token | Checkpoint name, `550B-A55B` |
| **Layers** | 108: 48 Mamba-2, 48 MoE, 12 attention | `config.json`, `layers_block_type` |
| **Experts** | 512 routed, 22 per token, 1 shared | `config.json`, `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts` |
| **KV heads, head dimension** | 2, 128 | `config.json`, `num_key_value_heads`, `head_dim` |
| **KV cache per token, FP8** | 6 KiB | `2 x 2 x 128 x 1 byte x 12 attention layers` |
| **Mamba state per sequence** | ~0.38 GiB, float32 | `256 heads x 64 x 128 state x 4 bytes x 48 layers`; `mamba_ssm_cache_dtype` |
| **Checkpoint size, NVFP4** | ~328 GiB, 113 safetensors shards | `fetch-weights` init container checks |
| **Instance** | p6-b200.48xlarge, 8x B200 at 180 GB, sm_100, NVLink | `nvidia.com/gpu: "8"` |
| **Tensor parallel size** | 8 | `--tensor-parallel-size 8` |
| **Weight precision** | ModelOpt mixed NVFP4 and FP8, 243 modules kept high precision | Checkpoint `quantization_config`; no `--quantization` flag |
| **KV cache precision** | FP8 | `KV_CACHE_DTYPE` |
| **Attention backend** | `FLASHINFER` | `ATTENTION_BACKEND` |
| **Context length** | 32,768 of a native 262,144 | `MAX_MODEL_LEN`; `config.json`, `max_position_embeddings` |
| **Max sequences** | 256 | `MAX_NUM_SEQS` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Stream interval** | 10 tokens | `STREAM_INTERVAL` |

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `p6-b200.48xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml) (set the instance type)
- [Mount its instance store as RAID0 for kubelet storage](../../../k8s/examples/nodeconfig-local-nvme-raid0.yaml) (the init container needs 400 GiB free)
- [Create a benchmark-client node group](../../../k8s/examples/eksctl-nodegroup-bench-client.yaml) (the client Job does not tolerate the GPU taint)
- [Create an IAM OIDC provider for the cluster](https://docs.aws.amazon.com/eks/latest/userguide/enable-iam-roles-for-service-accounts.html)
- [Create an IAM role for IRSA that reads the S3 checkpoint prefix](https://docs.aws.amazon.com/eks/latest/userguide/associate-service-account-role.html) (and writes it, if the checkpoint is not in S3 yet)

## Quickstart

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-nemotron-3-ultra-fp8kv-32k-flashinfer-p6-b200-48xlarge
   export INSTANCE_TYPE=p6-b200.48xlarge
   export WEIGHTS_BUCKET=<your-weights-bucket>
   export WEIGHTS_REGION=<your-weights-bucket-region>
   export NEMOTRON_S3_ROLE_ARN=<your-nemotron-s3-role-arn>
   ```

2. Configure the EKS context and namespace, install the NVIDIA device plugin, and create the `hf-token` secret.
   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Stage the published checkpoint from Hugging Face to S3; skip this when an earlier run already did.
   ```bash
   ./model-jobs/nemotron-3-ultra/nvfp4-model-download.sh
   ```

## Throughput test

1. Deploy the server; the script fails unless the engine reports `FLASHINFER`.
   ```bash
   ./vllm-recipes/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer-deploy.sh
   ```

2. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service nemotron-3-ultra-vllm-b200-fp8kv-32k-flashinfer \
       --mode fingerprint \
       --model-tag nemotron-3-ultra --config fp8kv-32k-flashinfer \
       --model-id nemotron-3-ultra \
       --out-dir docs/results
   ```

3. Dry-run the in-cluster benchmark, which renders the Job without creating it.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service nemotron-3-ultra-vllm-b200-fp8kv-32k-flashinfer \
       --protocol corpus --profile coding_repo_context \
       --instance p6-b200.48xlarge --model-tag nemotron-3-ultra \
       --config fp8kv-32k-flashinfer --quant nvfp4 \
       --model-id nemotron-3-ultra \
       --tokenizer nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

4. Benchmark in-cluster.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service nemotron-3-ultra-vllm-b200-fp8kv-32k-flashinfer \
       --protocol corpus --profile coding_repo_context \
       --instance p6-b200.48xlarge --model-tag nemotron-3-ultra \
       --config fp8kv-32k-flashinfer --quant nvfp4 \
       --model-id nemotron-3-ultra \
       --tokenizer nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/nemotron-3-ultra/nvfp4-p6-b200.48xlarge-fp8kv-32k-flashinfer.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    job/nemotron-3-ultra-nvfp4-hf-to-s3 --ignore-not-found
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name gpu-spot-1node \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```

Scale the benchmark-client node group to zero too if you created it for this run. The checkpoint in S3 survives, so a later run skips Quickstart step 3.

---

## Settings

- **One copy across all eight GPUs.** The recipe requests `nvidia.com/gpu: "8"` and shards one copy with `--tensor-parallel-size 8`, so results are per-node figures. The server prints `nvidia-smi topo -m` at start and warns when it finds no NVLink pairs; do not trust a run that logged that warning.
- **One arm at a time.** Each attention-backend arm takes the whole node and runs in its own namespace, so tear one down before deploying the next. One vLLM pod per namespace means `measure` needs no `--pod`.
- **The backend is pinned with `--attention-backend`, not `VLLM_ATTENTION_BACKEND`.** vLLM 0.29.0 does not read that environment variable. The deploy script fails unless the server logs `Using AttentionBackendEnum.FLASHINFER backend`.
- **NVFP4 on Blackwell is not comparable to a Hopper run.** On sm_100 the checkpoint runs the native cutlass W4A4 GEMM path. A Hopper GPU would fall back to FP4-Marlin W4A16, so never read this recipe's results against a Hopper one.
- **No `--quantization` flag.** The checkpoint's `quantization_config` declares a ModelOpt mixed NVFP4 and FP8 export, and the loader resolves it. Pinning a single method would mis-read the mixed export.
- **`--stream-interval 10`** The server streams in 10-token chunks. That changes inter-token latency only, so never compare ITL against a recipe with a different stream interval. All three Nemotron arms share it.
- **`--no-enable-prefix-caching`** Held off for an uncached measurement. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`--async-scheduling` and `--enable-chunked-prefill`** On in all three arms, so neither is a variable between them.
- **`MAX_NUM_SEQS=256`, `MAX_CUDAGRAPH=256` and `MAX_NUM_BATCHED_TOKENS=32768`.** The CUDA graph capture size matches the sequence cap. A 32,768-token batch budget prefills a full `coding_repo_context` prompt in one chunk.
- **No reasoning parser.** The recipe sets no `--reasoning-parser`, so any reasoning trace arrives in the message content and counts against the profile's 1,024-token output cap. The corpus protocol truncates a reasoning model's output by design.
- **The checkpoint ships an MTP layer that this recipe leaves off.** `config.json` sets `num_nextn_predict_layers: 1`, but the recipe passes no `--speculative-config`.
- **`--served-model-name nemotron-3-ultra`** The server answers to that name rather than its path, so `--model-id` is `nemotron-3-ultra`.
- **Weights come from S3 into an `emptyDir` on the NVMe array.** A pod restart re-downloads ~328 GiB. The `fetch-weights` init container fails unless `/models` has 400 GiB free, then fails unless it finds 113 shards and at least 300 GiB. Its log prints `df -h /models`; confirm that is the NVMe array.
- **`progressDeadlineSeconds: 7200`** The S3 sync and load outlast the 600-second default. The deploy script waits the same 7,200 seconds, and the startup probe allows 3,600 seconds once the init container finishes.
- **`serviceAccountName: nemotron-3-ultra-bench`** Supplies the pod-scoped S3 identity through IRSA. The staging Job runs as the same ServiceAccount, so the role also needs write access to the prefix while staging. Its trust policy must name each namespace that runs an arm.
- **`WEIGHTS_BUCKET` and `AWS_REGION` are placeholders in git.** The staging and deploy scripts fill them in from `WEIGHTS_BUCKET` and `WEIGHTS_REGION` as they apply each manifest, so the files in git stay untouched.
- **The staging Job tolerates the GPU taint and requests 350 GiB of ephemeral storage.** On most clusters that places it on the GPU node's NVMe array. It fails on a shard-count, partial-file or size mismatch, so a completed Job means all 113 shards reached S3.
