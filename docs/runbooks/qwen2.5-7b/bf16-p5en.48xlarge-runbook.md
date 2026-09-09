# Qwen2.5-7B-Instruct BF16 on p5en.48xlarge

Serves [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on p5en.48xlarge (8x H200, sm_90) with 1 GPU per pod.

## Capacity sizing

Qwen2.5-7B-Instruct at BF16 occupies 14.2 GiB of the 131.3 GiB on 1 H200 card, leaving 96.5 GiB for KV cache. At the model maximum 32,768 token context that pool holds 55 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 7B | Model card |
| **Layers** | 28 | `config.json`, `num_hidden_layers` |
| **KV heads, head dimension** | 4, 128 | `config.json` |
| **Checkpoint size, BF16** | 14.2 GiB | Model card safetensors metadata |
| **Instance** | p5en.48xlarge, 1 of 8x H200 at 141 GB | 141 GB for this pod, 131.3 GiB |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | BF16 | unquantized checkpoint |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 32,768, not pinned by the recipe | `config.json`, `max_position_embeddings` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 2 KiB here. Every layer attends to the full context, so per-sequence cost scales linearly with context length across all 28 layers. The 96.5 GiB pool is 131.3 GiB at 0.90 utilization, less 14.2 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `p5en.48xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen25-7b-bf16-p5en-48xlarge
   export INSTANCE_TYPE=p5en.48xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the persistent PVC, then stage the BF16 weights. A completed earlier download Job makes staging a no-op.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-downloads/qwen2.5-7b-bf16.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait --for=condition=complete job/download-qwen25-7b-bf16 --timeout=6h
   ```

4. Deploy and validate the model server mounted to `fsx-pvc`.

   ```bash
   ./vllm-recipes/qwen2.5-7b/bf16-p5en.48xlarge-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-p5en-bf16 \
       --mode fingerprint \
       --model-tag qwen2.5-7b --config bf16 \
       --model-id /models/Qwen2.5-7B-Instruct \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-p5en-bf16 \
       --protocol corpus --profile legacy_corpus \
       --instance p5en.48xlarge --model-tag qwen2.5-7b \
       --config bf16 --quant bf16 \
       --model-id /models/Qwen2.5-7B-Instruct \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-p5en-bf16 \
       --protocol corpus --profile legacy_corpus \
       --instance p5en.48xlarge --model-tag qwen2.5-7b \
       --config bf16 --quant bf16 \
       --model-id /models/Qwen2.5-7B-Instruct \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/qwen2.5-7b/bf16-p5en.48xlarge.yaml
```

---


## Settings

**`initContainers:`**

```
NOTE: the original run pinned this to a single node via kubernetes.io/hostname
```

because only that node could reach FSx (a security-group issue). Removed so this schedules on any p5en.48xlarge. Re-add a hostname selector if FSx access is still restricted to specific nodes in your cluster.
- **`node.kubernetes.io/instance-type: p5en.48xlarge`** H200 141GB HBM3e, sm_90 (Hopper)
- **`nvidia.com/gpu: 1`** p5en.48xlarge = 8x H200 141GB; using 1 for 7B model
