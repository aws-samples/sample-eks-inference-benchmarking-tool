# Mistral Large 3 FP8 on p5en.48xlarge, tensor parallel 8, FlashAttention MLA

Serves the published FP8 checkpoint of [`mistralai/Mistral-Large-3-675B-Instruct-2512`](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) as one copy split across all eight H200 GPUs of a p5en.48xlarge, using [`fp8-p5en.48xlarge.yaml`](../../../vllm-recipes/mistral-large-3/fp8-p5en.48xlarge.yaml).

## Capacity sizing

Mistral Large 3 is a mixture-of-experts model with multi-head latent attention (MLA). Every expert stays resident, so the ~635 GiB FP8 checkpoint needs the whole node: about 79 GiB of weights on each 141 GB H200. MLA caches one 576-element latent per token per layer instead of full keys and values, which is 1,152 bytes at 16 bits, or about 68.6 KiB per token across 61 layers. Under tensor parallelism every GPU holds the full latent cache. At a 0.90 memory fraction that leaves roughly 34 GiB per GPU, room for about 520,000 tokens, or about 15 full 32,768-token sequences. Read `GPU KV cache size` in the server log for the allocation in force.

`chat_typical` sends about 1,245 wire input tokens and asks for up to 1,024 output tokens, so a request occupies about 2,300 tokens. About 225 such requests fit at once, just under the 256-sequence cap, so KV memory and the cap bound the batch together for this workload.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 675B total, ~41B active per token; 128 routed experts, 4 active, 1 shared; first 3 layers dense | `params.json`, `moe` |
| **Layers** | 61 | `params.json`, `n_layers` |
| **MLA latent per token per layer** | 512 + 64 = 576 elements | `params.json`, `kv_lora_rank` + `qk_rope_head_dim` |
| **KV cache per token, 16-bit** | ~68.6 KiB, replicated on every GPU | `576 x 2 bytes x 61 layers` |
| **Checkpoint size, FP8** | ~635 GiB, 272 `consolidated-*.safetensors` shards | `fetch-weights` init container checks |
| **Instance** | p5en.48xlarge, 8x H200 at 141 GB, sm_90, NVLink | `nvidia.com/gpu: "8"` |
| **Tensor parallel size** | 8 | `--tensor-parallel-size 8` |
| **Weight precision** | FP8, compressed-tensors export, block-scaled | `params.json` `quantization_config`; no `--quantization` flag |
| **KV cache precision** | 16-bit | `KV_CACHE_DTYPE` |
| **Attention backend** | `FLASH_ATTN_MLA` | `ATTENTION_BACKEND` |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Max sequences** | 256 | `MAX_NUM_SEQS` |
| **Max batched tokens** | 32,768 | `MAX_NUM_BATCHED_TOKENS` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Stream interval** | 10 tokens | `STREAM_INTERVAL` |

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `p5en.48xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml) (set the instance type)
- [Install the CoreDNS add-on](https://docs.aws.amazon.com/eks/latest/userguide/managing-coredns.html) (EKS Auto Mode clusters lack it for managed node groups)
- [Install the EKS Pod Identity Agent add-on](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-agent-setup.html)
- [Create an EKS Pod Identity association with S3 read and write access to the bucket](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-association.html)

## Quickstart

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-mistral-large-3-fp8-p5en-48xlarge
   export INSTANCE_TYPE=p5en.48xlarge
   export WEIGHTS_BUCKET=<your-weights-bucket>
   export WEIGHTS_REGION=<your-weights-bucket-region>
   export GPU_NODE=<your-p5en-node-name>
   export S3_SERVICE_ACCOUNT=<your-pod-identity-service-account>
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

4. Assemble the NVMe instance store on `GPU_NODE` as RAID0 at `/mnt/nvme-raid0`.
   ```bash
   sed -e "s|SUBSTITUTE-gpu-node|${GPU_NODE}|" k8s/examples/pod-nvme-raid0.yaml \
       | kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f -
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=jsonpath='{.status.phase}'=Succeeded pod/nvme-raid0 --timeout=10m
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs nvme-raid0 | grep -x /dev/md0
   ```

5. Copy the published checkpoint from Hugging Face to your S3 bucket.
   ```bash
   ./model-jobs/mistral-large-3/fp8-model-download.sh
   ```

## Throughput test

1. Deploy the vLLM inference server on `GPU_NODE` and confirm it serves the model.
   ```bash
   ./vllm-recipes/mistral-large-3/fp8-p5en.48xlarge-deploy.sh
   ```

2. Confirm `/models` is on the NVMe RAID0 array, not the EBS root volume.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec deployment/mistral-large-3-vllm-p5en \
       --container vllm -- df -h /models
   ```

3. Warm up the server with one unrecorded benchmark run, saved outside `docs/results`.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service mistral-large-3-vllm-p5en \
       --protocol corpus --profile chat_typical \
       --instance p5en.48xlarge --model-tag mistral-large-3 \
       --config fp8 --quant fp8 \
       --model-id mistral-large-3 \
       --tokenizer mistralai/Mistral-Large-3-675B-Instruct-2512 \
       --skip-accuracy --no-latency --no-provenance \
       --out-dir /tmp/mistral-large-3-warmup \
       --reps 1
   ```

4. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service mistral-large-3-vllm-p5en \
       --mode fingerprint \
       --model-tag mistral-large-3 --config fp8 \
       --model-id mistral-large-3 \
       --out-dir docs/results
   ```

5. Dry-run the exact in-cluster benchmark.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service mistral-large-3-vllm-p5en \
       --protocol corpus --profile chat_typical \
       --instance p5en.48xlarge --model-tag mistral-large-3 \
       --config fp8 --quant fp8 \
       --model-id mistral-large-3 \
       --tokenizer mistralai/Mistral-Large-3-675B-Instruct-2512 \
       --out-dir docs/results \
       --reps 1 \
       --dry-run
   ```

6. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service mistral-large-3-vllm-p5en \
       --protocol corpus --profile chat_typical \
       --instance p5en.48xlarge --model-tag mistral-large-3 \
       --config fp8 --quant fp8 \
       --model-id mistral-large-3 \
       --tokenizer mistralai/Mistral-Large-3-675B-Instruct-2512 \
       --out-dir docs/results \
       --reps 1
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/mistral-large-3/fp8-p5en.48xlarge.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete pod nvme-raid0 --ignore-not-found
```

Scale the p5en node group to zero if nothing else uses it, and cancel any capacity reservation you created for this run. The checkpoint in S3 survives, so a later run skips Quickstart step 5.

---

## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **One copy across eight GPUs.** All experts stay resident, so `--tensor-parallel-size 8` splits every layer over the node's NVLink mesh. The server logs `WARNING: no NVLink pairs in topo matrix` at startup if `nvidia-smi topo -m` shows none; do not trust a result from that pod.
- **`FLASH_ATTN_MLA` was the fastest Hopper MLA backend tested.** `TRITON_MLA` and `FLASHMLA` also run on H200. `FLASHINFER_MLA` fails at startup on sm_90 with `compute capability not supported`. The deploy script fails unless the server logs `Using AttentionBackendEnum.FLASH_ATTN_MLA backend`.
- **The backend is pinned with `--attention-backend`, not `VLLM_ATTENTION_BACKEND`.** vLLM 0.29.0 does not read that environment variable. Set `ATTENTION_BACKEND` before running the deploy script to try another backend.
- **`--tokenizer-mode mistral --config-format mistral --load-format mistral`.** The checkpoint ships `params.json`, `tekken.json` and `consolidated-*.safetensors` shards, with no Hugging Face `config.json`.
- **No `--quantization` flag.** `params.json` declares a compressed-tensors FP8 export with block scales, and the loader resolves it. `VLLM_LOGGING_LEVEL=INFO` keeps the selected FP8 kernel in the server log.
- **16-bit KV cache.** The checkpoint ships no KV-cache scales (`kv_cache_scheme` is null), so `KV_CACHE_DTYPE=auto` keeps the cache at the model dtype.
- **`--no-enable-prefix-caching`** Held off for an uncached measurement. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`--stream-interval 10`** The server streams in 10-token chunks. That changes inter-token latency only, so compare ITL only against recipes with the same stream interval.
- **Weights come from S3 into `/mnt/nvme-raid0` through a hostPath.** The node's kubelet storage stays on the EBS root volume, which is too small for the checkpoint. The staging Job and the recipe both pin `GPU_NODE` and read the same directory, so the recipe's S3 sync finds the shards already local. A replaced node loses the array and the shards; rerun Quickstart step 4, and the sync downloads them again.
- **The `fetch-weights` init container fails** unless `/models` has 100 GiB free before the sync and holds 272 shards and at least 600 GiB after it.
- **The staging Job sends unauthenticated requests.** The repository is not gated, so the `hf-token` Secret is optional. A token raises the Hub's rate limits.
- **`--served-model-name mistral-large-3`** The server answers to that name rather than its path, so `--model-id` is `mistral-large-3`.
- **The chat template adds about 600 tokens.** Mistral's system prompt is part of every request, so `chat_typical` measures about 1,245 wire input tokens against a nominal 512.
- **`--reps 1`** One repetition, so the result has no spread to check. Raise `--reps` before quoting a figure.
- **Not comparable to a Blackwell run.** Results are labelled with the instance.
