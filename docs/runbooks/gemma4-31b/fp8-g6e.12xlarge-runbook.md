# Gemma 4 31B FP8 on g6e.12xlarge

Serves [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it) on g6e.12xlarge (4x L40S, sm_89) with 4 GPUs per pod.

## Capacity sizing

Gemma 4 31B at FP8 occupies 30.4 GiB of the 178.8 GiB on four L40S cards, leaving 123.0 GiB for KV cache. At the recipe's 8192 token context that pool holds 61 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 31B | Model card |
| **Layers** | 60, of which 10 full attention and 50 sliding at window 1024 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 16, 256 | `config.json`, `text_config` |
| **Checkpoint size, FP8** | 30.4 GiB | 1 byte per parameter, embeddings stay 16-bit |
| **Instance** | g6e.12xlarge, 4x L40S at 48 GB | 192 GB nominal, 178.8 GiB |
| **Tensor parallel size** | 4 | `TP_SIZE` |
| **Weight precision** | FP8 | `QUANTIZATION` |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 8192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 16 KiB here. Gemma 4 interleaves attention, so the sliding layers hold at most 1024 tokens whatever the context length and per-sequence cost grows only through the 10 full attention layers. The 123.0 GiB pool is 178.8 GiB at 0.90 utilization, less 30.4 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g6e.12xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Create the `vllm-sitecustomize` ConfigMap](../../../k8s/examples/configmap-vllm-sitecustomize.yaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-31b-fp8-g6e-12xlarge
   export INSTANCE_TYPE=g6e.12xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the FSx claim containing the quantized weights and the sitecustomize ConfigMap exist in the namespace.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get configmap vllm-sitecustomize
   ```

4. Deploy and validate the model server. The server mounts `/models/quantized/gemma-4-31b-it-fp8` from `fsx-pvc`; update the recipe first if your weights use a differently named PVC.

   ```bash
   ./vllm-recipes/gemma4-31b/fp8-g6e.12xlarge-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g6e-fp8 \
       --mode fingerprint \
       --model-tag gemma4-31b --config fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g6e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.12xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g6e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.12xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-31b/fp8-g6e.12xlarge.yaml
```

---

## Settings

- **`- --gpu-memory-utilization=0.90`** No FP8 KV cache: vLLM 0.21 + Gemma 4 raises the q_scale assertion.

- **`- --port=8000`** No --enforce-eager: CUDA graphs on (L40S/sm_89 is well supported).
- **`nvidia.com/gpu: 4`** g6e.12xlarge = 4x NVIDIA L40S, 48GB each (sm_89)
- **`- --tensor-parallel-size=4`** 31B sharded across all 4 L40S
- **`- --max-model-len=8192`** Capped well below the model's 262144, so context length is fixed rather than a variable in any measurement.
- **`- --max-num-seqs=512`** Queue depth, not a concurrency target. The KV pool tops out near 61 sequences at 8192 tokens, so a sweep past that measures queueing rather than the server.
- **`VLLM_ATTENTION_BACKEND=FLASHINFER`** ships sm_89 cubins for L40S
