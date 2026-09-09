# Gemma 4 31B FP8 on g7e.4xlarge

Serves [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it) on g7e.4xlarge (1x RTX PRO 6000 Blackwell, sm_120) with 1 GPU per pod.

## Capacity sizing

gemma-4-31b-it at FP8 occupies 30.4 GiB of the 89.4 GiB on 1 RTX PRO 6000 Blackwell card, leaving 47.1 GiB for KV cache. At the recipe's 8,192 token context that pool holds 23 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 31B | Model card |
| **Layers** | 60, of which 10 full attention and 50 sliding at window 1024 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 16, 256 | `config.json` |
| **Checkpoint size, FP8** | 30.4 GiB | 1 byte per parameter, embeddings stay 16-bit |
| **Instance** | g7e.4xlarge, 1x RTX PRO 6000 Blackwell at 96 GB | 96 GB nominal, 89.4 GiB |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | FP8 | `QUANTIZATION` |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 16 KiB here. Gemma 4 interleaves attention, so the sliding layers hold at most 1,024 tokens whatever the context length and per-sequence cost grows only through the 10 full attention layers. The 47.1 GiB pool is 89.4 GiB at 0.95 utilization, less 30.4 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

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
   export NAMESPACE=vb-gemma4-31b-fp8-g7e-4xlarge
   export INSTANCE_TYPE=g7e.4xlarge
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

4. Deploy and validate the model server. The server mounts `/models/quantized/gemma-4-31b-it-fp8` from `fsx-pvc`; update `claimName` first if your weights use a differently named PVC.

   ```bash
   ./vllm-recipes/gemma4-31b/fp8-g7e.4xlarge-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7e-fp8 \
       --mode fingerprint \
       --model-tag gemma4-31b --config fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.4xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.4xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-31b/fp8-g7e.4xlarge.yaml
```

---


## Settings

**`image: public.ecr.aws/deep-learning-containers/vllm:0.21.0-gpu-py312`**

```
NOTE: Verify this DLC tag actually ships the Gemma 4 model code before running.
```

Gemma 4 released Apr 2026; if 0.21.0 predates the gemma4 model module, bump to a newer vLLM image that includes it (see vLLM Gemma 4 recipe).

**`- --gpu-memory-utilization=0.95`** Left at 0.95 rather than tuned for this precision, so precision is the only variable in a comparison.

```
NOTE: FP8 KV cache (--kv-cache-dtype=fp8_e4m3 --calculate-kv-scales) is removed.
```

On vLLM 0.21 + Gemma 4 it triggers: "AssertionError: A non 1.0 q_scale is not currently supported." Weights remain FP8 (the throughput lever); KV cache uses default precision. This also keeps the KV cache comparable to the BF16 baseline.

- **`- --max-model-len=8192`** Context capped at 8192, well below the model's 262144, so context length is fixed rather than a variable in any measurement.

**`- --port=8000`**

```
--enforce-eager intentionally NOT set: vLLM 0.21 V1 engine + precompiled
```

FlashInfer cubins handle cudagraph capture on sm_120. Eager mode disabled kernel fusion and was an unnecessary handicap.
- **`nvidia.com/gpu: 1`** g7e.4xlarge = 1x Blackwell RTX PRO 6000, 96GB (sm_120)
- **`- --max-num-seqs=512`** matched to BF16 and the Qwen g7e config
- **`VLLM_ATTENTION_BACKEND=FLASHINFER`** Required for Blackwell sm_120
- **`claimName: fsx-pvc`** A ReadWriteMany claim holding the weights. Change it here if yours is named differently.
