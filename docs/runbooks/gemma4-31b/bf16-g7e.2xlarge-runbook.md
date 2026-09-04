# Gemma 4 31B BF16 on g7e.2xlarge

Serves [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it) on g7e.2xlarge (1x RTX PRO 6000 Blackwell, sm_120) with 1 GPU per pod.

## Capacity sizing

gemma-4-31b-it at BF16 occupies 58.2 GiB of the 89.4 GiB on 1 RTX PRO 6000 Blackwell card, leaving 19.2 GiB for KV cache. At the recipe's 8,192 token context that pool holds 19 concurrent sequences, measured from `num_gpu_blocks=9845` at `block_size=16` on `/metrics` (157,520 tokens / 8,192). The estimate below this table, derived from the per-token KV cost alone, undercounts because it does not net out activation and CUDA graph memory as precisely as the allocator does; treat the measured figure as authoritative.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 31B | Model card |
| **Layers** | 60, of which 10 full attention and 50 sliding at window 1024 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 16, 256 | `config.json` |
| **Checkpoint size, BF16** | 58.2 GiB | Model card safetensors metadata |
| **Instance** | g7e.2xlarge, 1x RTX PRO 6000 Blackwell at 96 GB | 96 GB nominal, 89.4 GiB |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | BF16 | unquantized checkpoint |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 16 KiB here. Gemma 4 interleaves attention, so the sliding layers hold at most 1,024 tokens whatever the context length and per-sequence cost grows only through the 10 full attention layers. The 19.2 GiB pool is 89.4 GiB at 0.95 utilization, less 58.2 GiB of weights, less roughly 12 GiB for activations and CUDA graphs.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- Create a `g7e.2xlarge` node group in a subnet with free IPs; the AZ this instance's capacity reservation lands in may have none by default, so check first
- [Create the `vllm-sitecustomize` ConfigMap](../../../k8s/examples/configmap-vllm-sitecustomize.yaml)

No FSx PVC in this runbook. The recipe downloads its own weights into an `emptyDir`, the same pattern `glm-5.2/nvfp4-p6-b300.48xlarge.yaml` uses, because no FSx for Lustre filesystem exists in every AZ this instance type's capacity might land in.

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-31b-bf16-g7e-2xlarge
   export INSTANCE_TYPE=g7e.2xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the sitecustomize ConfigMap exists in the namespace.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get configmap vllm-sitecustomize
   ```

4. Deploy and validate the model server. Its init container downloads the weights into pod-local (`emptyDir`) storage before vLLM starts, so the first rollout takes as long as the download and pod replacement downloads them again.

   ```bash
   ./vllm-recipes/gemma4-31b/bf16-g7e.2xlarge-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   python3 app/cli.py launch --capture-accuracy-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7e-2xl-bf16 \
       --mode fingerprint \
       --model-tag gemma4-31b --config bf16 \
       --model-id /models/gemma-4-31b-it \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   python3 app/cli.py launch \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7e-2xl-bf16 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.2xlarge --model-tag gemma4-31b \
       --config bf16 --quant bf16 \
       --model-id /models/gemma-4-31b-it \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   python3 app/cli.py launch \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7e-2xl-bf16 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.2xlarge --model-tag gemma4-31b \
       --config bf16 --quant bf16 \
       --model-id /models/gemma-4-31b-it \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-31b/bf16-g7e.2xlarge.yaml
```

## Settings

**`image: public.ecr.aws/deep-learning-containers/vllm:0.21.0-gpu-py312`**

```
NOTE: same image as the g7e.4xlarge BF16 deployment. Verify this DLC ships
```

the Gemma 4 model module before relying on it; bump the tag if not.

- **`- --gpu-memory-utilization=0.95`** Left at 0.95 rather than tuned for this precision, so precision is the only variable in a comparison. No `--quantization` and no `--kv-cache-dtype`: this serves the unquantized checkpoint.
- **`- --max-model-len=8192`** Context capped at 8192, well below the model's 262144, so context length is fixed rather than a variable in any measurement.
- **`fetch-weights` init container** Downloads `google/gemma-4-31b-it` into the `model-storage` `emptyDir` before vLLM starts. Checks free space against `MIN_FREE_GIB` first, since an `emptyDir` lands wherever kubelet's root is rather than necessarily the instance store, and resumes rather than restarts if a partial checkpoint is already present.
- **`model-storage: emptyDir: {sizeLimit: 100Gi}`** Sized for the 58.2 GiB checkpoint plus the Hugging Face cache under the same mount, with headroom. Scaling this deployment to zero discards the download; a re-run downloads again.

```
--enforce-eager intentionally NOT set: vLLM 0.21 V1 engine + precompiled
```

FlashInfer cubins handle cudagraph capture on sm_120. Eager mode disabled kernel fusion and was an unnecessary handicap.
- **`nvidia.com/gpu: 1`** g7e.2xlarge = 1x Blackwell RTX PRO 6000, 96GB (sm_120), same GPU as g7e.4xlarge; only vCPU and host RAM differ between the two sizes.
- **`- --max-num-seqs=512`** matched to the g7e.4xlarge FP8 and BF16 configs.
- **`VLLM_ATTENTION_BACKEND=FLASHINFER`** Required for Blackwell sm_120.
