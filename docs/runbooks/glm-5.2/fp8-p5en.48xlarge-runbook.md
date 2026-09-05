# GLM-5.2 FP8 on p5en.48xlarge

Serves [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) on p5en.48xlarge (8x H200, sm_90) with 8 GPUs per pod.

## Capacity sizing

GLM-5.2-FP8 at FP8 occupies 703.7 GiB of the 1,050.5 GiB on 8 H200 cards, leaving 234.3 GiB for KV cache. At the recipe's 32,768 token context that pool holds 85 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 753B total | `zai-org/GLM-5.2-FP8` card; the NVFP4 card reports packed bytes, not parameters |
| **Layers** | 78 | `config.json`, `num_hidden_layers` |
| **KV latent per layer** | kv_lora_rank 512 plus qk_rope 64 | `config.json` |
| **Checkpoint size, FP8** | 703.7 GiB | Model card safetensors metadata |
| **Instance** | p5en.48xlarge, 8x H200 at 141 GB | 1,128 GB nominal, 1,050.5 GiB |
| **Tensor parallel size** | 8 | `TP_SIZE` |
| **Weight precision** | FP8 | declared by the checkpoint's `quantization_config` |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `(kv_lora_rank + qk_rope_head_dim) x bytes_per_element`, which is 1.12 KiB here. Multi-head latent attention caches one compressed latent per layer instead of a key and value per head, so KV cost scales with layer count rather than head count. The 234.3 GiB pool is 1,050.5 GiB at 0.90 utilization, less 703.7 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `p5en.48xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- Label the nodes `role=p5en-glm`

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-glm-52-fp8-p5en-48xlarge
   export INSTANCE_TYPE=p5en.48xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm a `role=p5en-glm` node has the complete published FP8 checkpoint staged in the node directory `/mnt/glm-weights/glm-5.2-fp8`. The pod sees it at `/weights/glm-5.2-fp8`; this recipe has no downloader or quantization step.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" get nodes -l role=p5en-glm
   ```

4. Deploy and validate the model server.

   ```bash
   ./vllm-recipes/glm-5.2/fp8-p5en.48xlarge-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service glm-52-p5en \
       --mode fingerprint \
       --model-tag glm-5.2 --config fp8 \
       --model-id glm-5.2-fp8 \
       --thinking off \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service glm-52-p5en \
       --protocol corpus --profile legacy_corpus \
       --instance p5en.48xlarge --model-tag glm-5.2 \
       --config fp8 --quant fp8 \
       --model-id glm-5.2-fp8 \
       --tokenizer nvidia/GLM-5.2-NVFP4 \
       --thinking off \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service glm-52-p5en \
       --protocol corpus --profile legacy_corpus \
       --instance p5en.48xlarge --model-tag glm-5.2 \
       --config fp8 --quant fp8 \
       --model-id glm-5.2-fp8 \
       --tokenizer nvidia/GLM-5.2-NVFP4 \
       --thinking off \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/glm-5.2/fp8-p5en.48xlarge.yaml
```

---


## Settings

- **`--tensor-parallel-size 8`** --- parallelism: 8 H200 on one node, NVLink, no EFA needed -------

**`(end of file)`**

--- precision --------------------------------------------------- Weights are FP8 in the checkpoint and vLLM auto-detects that, so no

```
--quantization flag is needed.
```

NO --kv-cache-dtype. This is not an oversight, and it contradicts vLLM's own GLM-5.2 recipe, which passes `--kv-cache-dtype fp8` for 8xH200. On vLLM 0.26.0 that combination has no valid attention backend, because GLM-5.2 needs SPARSE MLA and every sparse backend rejects an FP8 KV cache:

```
FLASHMLA_SPARSE:       kv_cache_dtype not supported
FLASH_ATTN_MLA_SPARSE: supports only FP16/BF16 KV cache
```

FLASHMLA / TRITON_MLA / FLASH_ATTN_MLA: sparse not supported

```
FLASHINFER_MLA:        sparse not supported, compute capability not supported
```

The engine aborts at load with "No valid attention backend found for cuda". Leaving kv-cache-dtype at its default gives a 16-bit KV cache, which roughly doubles KV footprint versus the recipe's intent - so KV budget, not weights, is what bounds concurrency here. Record that with any result.

- **`--reasoning-parser glm45`** --- GLM-specific parsers (names per vLLM's GLM-5.2 recipe) -------

- **`--max-model-len 32768`** --- capacity ----------------------------------------------------

- **`--max-cudagraph-capture-size 512`** MUST be >= --max-num-seqs. See the note above.

- **`--no-enable-prefix-caching`** --- measurement hygiene ----------------------------------------- Off for an honest baseline; vLLM's recipe asks for this when benchmarking too.
