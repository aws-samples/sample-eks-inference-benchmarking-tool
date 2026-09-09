# Qwen2.5-7B-Instruct FP8 on g6e.4xlarge

Serves [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on g6e.4xlarge (1x L40S, sm_89) with 1 GPU per pod.

## Capacity sizing

Qwen2.5-7B-Instruct at FP8 occupies 7.4 GiB of the 44.7 GiB on 1 L40S card, leaving 27.6 GiB for KV cache. At the model maximum 32,768 token context that pool holds 31 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 7B | Model card |
| **Layers** | 28 | `config.json`, `num_hidden_layers` |
| **KV heads, head dimension** | 4, 128 | `config.json` |
| **Checkpoint size, FP8** | 7.4 GiB | 1 byte per parameter, derived from the BF16 card |
| **Instance** | g6e.4xlarge, 1x L40S at 48 GB | 48 GB nominal, 44.7 GiB |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | FP8 | `QUANTIZATION` |
| **KV cache precision** | FP8 (`fp8_e4m3`) | `--kv-cache-dtype fp8_e4m3` |
| **Context length** | 32,768, not pinned by the recipe | `config.json`, `max_position_embeddings` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 1 KiB here. Every layer attends to the full context, so per-sequence cost scales linearly with context length across all 28 layers. The 27.6 GiB pool is 44.7 GiB at 0.95 utilization, less 7.4 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g6e.4xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Create the `vllm-sitecustomize` ConfigMap](../../../k8s/examples/configmap-vllm-sitecustomize.yaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen25-7b-fp8-g6e-4xlarge
   export INSTANCE_TYPE=g6e.4xlarge
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
   ./vllm-recipes/qwen2.5-7b/fp8-g6e.4xlarge-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-fp8 \
       --mode fingerprint \
       --model-tag qwen2.5-7b --config fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.4xlarge --model-tag qwen2.5-7b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.4xlarge --model-tag qwen2.5-7b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/qwen2.5-7b/fp8-g6e.4xlarge.yaml
```

---


## Settings

- **`nvidia.com/gpu: 1`** g6e.4xlarge = 1x L40S 48GB (sm_89)
