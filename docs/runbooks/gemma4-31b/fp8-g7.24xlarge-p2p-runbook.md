# Gemma 4 31B FP8, TP=4, NCCL P2P study on g7.24xlarge

Serves [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it) on g7.24xlarge (4x RTX PRO 4500 Blackwell, sm_120) with 4 GPUs per pod.

## Capacity sizing

gemma-4-31b-it at FP8 occupies 30.4 GiB of the 119.2 GiB on 4 RTX PRO 4500 Blackwell cards, leaving 69.4 GiB for KV cache. At the recipe's 8,192 token context that pool holds 34 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 31B | Model card |
| **Layers** | 60, of which 10 full attention and 50 sliding at window 1024 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 16, 256 | `config.json` |
| **Checkpoint size, FP8** | 30.4 GiB | 1 byte per parameter, embeddings stay 16-bit |
| **Instance** | g7.24xlarge, 4x RTX PRO 4500 Blackwell at 32 GB | 128 GB nominal, 119.2 GiB |
| **Tensor parallel size** | 4 | `TP_SIZE` |
| **Weight precision** | FP8 | `QUANTIZATION` |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 16 KiB here. Gemma 4 interleaves attention, so the sliding layers hold at most 1,024 tokens whatever the context length and per-sequence cost grows only through the 10 full attention layers. The 69.4 GiB pool is 119.2 GiB at 0.90 utilization, less 30.4 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7.24xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Create the `vllm-sitecustomize` ConfigMap](../../../k8s/examples/configmap-vllm-sitecustomize.yaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-31b-fp8-g7-24xlarge-p2p
   export INSTANCE_TYPE=g7.24xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm each remaining prerequisite exists.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get configmap vllm-sitecustomize
   ```

4. Bring up the server mounted to the weights on the PVC (`fsx-pvc`), setting NCCL_P2P_LEVEL first. If weights live on a different PVC (`fsx-pvc`), update the recipe before proceeding.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f vllm-recipes/gemma4-31b/fp8-g7.24xlarge-p2p.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" rollout status deploy/vllm-g7-tp4-p2p --timeout=3600s
   ```

5. Check whether the recipe's server is up and ready to serve.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec deploy/vllm-g7-tp4-p2p -c vllm -- \
     curl -s localhost:8000/v1/models
   ```

6. Read the KV cache the server allocated. Record the reported maximum concurrency with any result.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs deploy/vllm-g7-tp4-p2p -c vllm \
     | grep -E "GPU KV cache size|Maximum concurrency|model weights took"
   ```

7. Prove this recipe passes the flags it claims.

   ```bash
   python3 app/bench/src/main.py validate-manifests gemma4-31b/fp8-g7.24xlarge-p2p.yaml
   ```

8. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   python3 app/bench/src/main.py launch --capture-accuracy-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service vllm-g7-tp4-p2p \
       --mode fingerprint \
       --model-tag gemma4-31b --config p2p-pix \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --out-dir docs/results
   ```

9. Benchmark in-cluster. Change --config to match the arm: p2p-pix, p2p-phb, p2p-shm.

   ```bash
   python3 app/bench/src/main.py launch \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service vllm-g7-tp4-p2p \
       --protocol corpus --profile legacy_corpus \
       --instance g7.24xlarge --model-tag gemma4-31b \
       --config p2p-pix --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-31b/fp8-g7.24xlarge-p2p.yaml
```

---


## Settings

- **`- name: NCCL_P2P_LEVEL`** ---- the variable under test (change per run) ----

**`- name: NCCL_DEBUG`**

- name: NCCL_P2P_DISABLE

```
value: "1"          # <-- lower bound (force host staging)
```

---- diagnostics ----
- **`- /models/quantized/gemma-4-31b-it-fp8`** or a HF id vLLM can download
- **`- --quantization=modelopt`** drop if using a BF16/base model
- **`- --tensor-parallel-size=4`** forces the cross-GPU all-reduce we are measuring
- **`NCCL_P2P_LEVEL=PIX`** <-- PHB for the second run
- **`VLLM_ATTENTION_BACKEND=FLASHINFER`** required for Blackwell sm_120 (g7)
- **`VLLM_LOGGING_LEVEL=INFO`** INFO so vLLM prints whether custom all-reduce is used
- **`claimName: fsx-pvc`** The claim the weights are staged on. Change it here if yours is named differently.
