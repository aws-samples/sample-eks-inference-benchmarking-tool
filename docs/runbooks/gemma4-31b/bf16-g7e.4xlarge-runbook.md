# Gemma 4 31B BF16 on g7e.4xlarge

Serves [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it) on g7e.4xlarge (1x RTX PRO 6000 Blackwell, sm_120) with 1 GPU per pod.

## Capacity sizing

gemma-4-31b-it at BF16 occupies 58.2 GiB of the 89.4 GiB on 1 RTX PRO 6000 Blackwell card, leaving 19.2 GiB for KV cache. At the recipe's 8,192 token context that pool holds 9 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 31B | Model card |
| **Layers** | 60, of which 10 full attention and 50 sliding at window 1024 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 16, 256 | `config.json` |
| **Checkpoint size, BF16** | 58.2 GiB | Model card safetensors metadata |
| **Instance** | g7e.4xlarge, 1x RTX PRO 6000 Blackwell at 96 GB | 96 GB nominal, 89.4 GiB |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | BF16 | unquantized checkpoint |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 16 KiB here. Gemma 4 interleaves attention, so the sliding layers hold at most 1,024 tokens whatever the context length and per-sequence cost grows only through the 10 full attention layers. The 19.2 GiB pool is 89.4 GiB at 0.95 utilization, less 58.2 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

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
   export NAMESPACE=vb-gemma4-31b-bf16-g7e-4xlarge
   export INSTANCE_TYPE=g7e.4xlarge
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

4. Stage the weights on the PVC.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-jobs/gemma4-31b/bf16-model-download.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait --for=condition=complete job/download-gemma4-31b-bf16 --timeout=6h
   ```

5. Deploy and validate the model server. The server mounts the staged weights from `fsx-pvc`; update the recipe first if your weights use a differently named PVC.

   ```bash
   ./vllm-recipes/gemma4-31b/bf16-g7e.4xlarge-deploy.sh
   ```

6. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7e-bf16 \
       --mode fingerprint \
       --model-tag gemma4-31b --config bf16 \
       --model-id /models/gemma-4-31b-it \
       --out-dir docs/results
   ```

7. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7e-bf16 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.4xlarge --model-tag gemma4-31b \
       --config bf16 --quant bf16 \
       --model-id /models/gemma-4-31b-it \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

8. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7e-bf16 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.4xlarge --model-tag gemma4-31b \
       --config bf16 --quant bf16 \
       --model-id /models/gemma-4-31b-it \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-31b/bf16-g7e.4xlarge.yaml
```

---


## Settings



**`image: public.ecr.aws/deep-learning-containers/vllm:0.21.0-gpu-py312`**

```
NOTE: same image as the FP8 deployment. Verify this DLC ships the Gemma 4
```

(gemma4) model module before relying on it; bump the tag if not.

- **`- --gpu-memory-utilization=0.95`** Left at 0.95 rather than tuned for this precision, so precision is the only variable in a comparison. No `--quantization` and no `--kv-cache-dtype`: this serves the unquantized checkpoint.

- **`- --max-model-len=8192`** Context capped at 8192, well below the model's 262144, so context length is fixed rather than a variable in any measurement.

**`- --port=8000`**

```
--enforce-eager intentionally NOT set: vLLM 0.21 V1 engine + precompiled
```

FlashInfer cubins handle cudagraph capture on sm_120. Eager mode disabled kernel fusion and was an unnecessary handicap.
- **`nvidia.com/gpu: 1`** g7e.4xlarge = 1x Blackwell RTX PRO 6000, 96GB (sm_120)
- **`- --max-num-seqs=512`** matched to FP8 and the Qwen g7e config
- **`VLLM_ATTENTION_BACKEND=FLASHINFER`** Required for Blackwell sm_120
