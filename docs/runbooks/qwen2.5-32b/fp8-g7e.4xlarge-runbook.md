# Qwen2.5-32B-Instruct FP8 on g7e.4xlarge

Serves [`Qwen/Qwen2.5-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) on g7e.4xlarge (1x RTX PRO 6000 Blackwell, sm_120) with 1 GPU per pod.

## Capacity sizing

Qwen2.5-32B-Instruct at FP8 occupies 31.4 GiB of the 89.4 GiB on 1 RTX PRO 6000 Blackwell card, leaving 43.4 GiB for KV cache. At the model maximum 32,768 token context that pool holds 10 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 32B | Model card |
| **Layers** | 64 | `config.json`, `num_hidden_layers` |
| **KV heads, head dimension** | 8, 128 | `config.json` |
| **Checkpoint size, FP8** | 31.4 GiB | 1 byte per parameter, derived from the BF16 card |
| **Instance** | g7e.4xlarge, 1x RTX PRO 6000 Blackwell at 96 GB | 96 GB nominal, 89.4 GiB |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | FP8 | `QUANTIZATION` |
| **KV cache precision** | FP8 (`fp8_e4m3`) | `--kv-cache-dtype fp8_e4m3` |
| **Context length** | 32,768, not pinned by the recipe | `config.json`, `max_position_embeddings` |
| **Memory fraction** | 0.92 | `GPU_MEM_UTIL` |
| **Max sequences** | 256 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 2 KiB here. Every layer attends to the full context, so per-sequence cost scales linearly with context length across all 64 layers. The 43.4 GiB pool is 89.4 GiB at 0.92 utilization, less 31.4 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.4xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Create the `vllm-sitecustomize` ConfigMap](../../../k8s/examples/configmap-vllm-sitecustomize.yaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen25-32b-fp8-g7e-4xlarge
   export INSTANCE_TYPE=g7e.4xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the persistent PVC, pre-quantized checkpoint, and sitecustomize setup expected by this recipe.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get configmap vllm-sitecustomize
   ```

4. Deploy and validate the model server mounted to `fsx-pvc`.

   ```bash
   ./vllm-recipes/qwen2.5-32b/fp8-g7e.4xlarge-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen32b-vllm-g7e-fp8 \
       --mode fingerprint \
       --model-tag qwen2.5-32b --config fp8 \
       --model-id /models/quantized/Qwen2.5-32B-Instruct-fp8 \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen32b-vllm-g7e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.4xlarge --model-tag qwen2.5-32b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-32B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-32B-Instruct \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen32b-vllm-g7e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.4xlarge --model-tag qwen2.5-32b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-32B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-32B-Instruct \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/qwen2.5-32b/fp8-g7e.4xlarge.yaml
```

---


## Settings

- **`nvidia.com/gpu: 1`** g7e.4xlarge = 1x Blackwell RTX PRO 6000, 96GB (sm_120)
- **`- --gpu-memory-utilization=0.92`** 32B FP8 ~32GB weights; leaves ~60GB for KV cache
- **`- --max-num-seqs=256`** Lower than 7B due to larger per-seq memory
- **`- --enforce-eager`** Workaround: CUDA illegal memory access during cudagraph warmup on g7e Blackwell (sm_120) with 32B model
- **`VLLM_ATTENTION_BACKEND=FLASHINFER`** Required for Blackwell sm_120
