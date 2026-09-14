# Gemma 4 31B BF16 on g6e.12xlarge

Serves [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it) on g6e.12xlarge (4x L40S, sm_89) with 4 GPUs per pod.

## Capacity sizing

gemma-4-31b-it at BF16 occupies 58.2 GiB of the 178.8 GiB on 4 L40S cards, leaving 95.2 GiB for KV cache. At the recipe's 8,192 token context that pool holds 46 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 31B | Model card |
| **Layers** | 60, of which 10 full attention and 50 sliding at window 1024 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 16, 256 | `config.json` |
| **Checkpoint size, BF16** | 58.2 GiB | Model card safetensors metadata |
| **Instance** | g6e.12xlarge, 4x L40S at 48 GB | 192 GB nominal, 178.8 GiB |
| **Tensor parallel size** | 4 | `TP_SIZE` |
| **Weight precision** | BF16 | unquantized checkpoint |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 16 KiB here. Gemma 4 interleaves attention, so the sliding layers hold at most 1,024 tokens whatever the context length and per-sequence cost grows only through the 10 full attention layers. The 95.2 GiB pool is 178.8 GiB at 0.90 utilization, less 58.2 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.


## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a g6e.12xlarge node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Create the `vllm-sitecustomize` ConfigMap](../../../k8s/examples/configmap-vllm-sitecustomize.yaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-31b-bf16-g6e-12xlarge
   export INSTANCE_TYPE=g6e.12xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the FSx claim and sitecustomize ConfigMap exist in the namespace.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get configmap vllm-sitecustomize
   ```

4. Stage the weights. Re-running is a no-op once the checkpoint is complete.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-jobs/gemma4-31b/bf16-model-download.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait --for=condition=complete job/download-gemma4-31b-bf16 --timeout=6h
   ```

5. Deploy and validate the model server. The server mounts the staged weights from `fsx-pvc`.

   ```bash
   ./vllm-recipes/gemma4-31b/bf16-g6e.12xlarge-deploy.sh
   ```

6. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g6e-bf16 \
       --mode fingerprint \
       --model-tag gemma4-31b --config bf16 \
       --model-id /models/gemma-4-31b-it \
       --out-dir docs/results
   ```

7. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g6e-bf16 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.12xlarge --model-tag gemma4-31b \
       --config bf16 --quant bf16 \
       --model-id /models/gemma-4-31b-it \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

8. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g6e-bf16 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.12xlarge --model-tag gemma4-31b \
       --config bf16 --quant bf16 \
       --model-id /models/gemma-4-31b-it \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-31b/bf16-g6e.12xlarge.yaml
```

---


## Settings

- **`--tensor-parallel-size ${TP_SIZE}`** 31B BF16 (~62GB) does not fit on one 48GB L40S, so TP=4 shards it across all four GPUs. This is why g6e needs more than 1 GPU where g7e needs 1.
- **`nvidia.com/gpu: 4`** g6e.12xlarge = 4x NVIDIA L40S, 48GB each (sm_89)
- **`--gpu-memory-utilization ${GPU_MEM_UTIL}`** Left at 0.90 rather than tuned for this precision, so precision is the only variable in a comparison.
- **`--max-model-len ${MAX_MODEL_LEN}`** Capped well below the model's 262144, so context length is fixed rather than a variable in any measurement.
- **`VLLM_ATTENTION_BACKEND=FLASHINFER`** ships sm_89 cubins for L40S
- No `--enforce-eager`, so CUDA graphs stay on.
