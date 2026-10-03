# Qwen2.5-7B-Instruct FP8 on g6e.12xlarge

Serves [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on g6e.12xlarge (4x L40S, sm_89) with 1 GPU per pod.

## Capacity sizing

Qwen2.5-7B-Instruct at ModelOpt FP8 occupies 8.1 GiB of the 44.4 GiB on 1 L40S card, leaving about 28.5 GiB for KV cache. At the model maximum 32,768 token context that pool holds 16 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 7B | Model card |
| **Layers** | 28 | `config.json`, `num_hidden_layers` |
| **KV heads, head dimension** | 4, 128 | `config.json` |
| **Checkpoint size, FP8** | 8.1 GiB | `model.safetensors` from the quantize step; embeddings and `lm_head` stay BF16 |
| **Instance** | g6e.12xlarge, 1 of 4x L40S at 48 GB | 44.4 GiB as CUDA reports it |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | ModelOpt FP8 | `QUANTIZATION` |
| **KV cache precision** | 16-bit | No `--kv-cache-dtype`; see Settings |
| **Context length** | 32,768, not pinned by the recipe | `config.json`, `max_position_embeddings` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 2 KiB here, or 56 KiB per token across all 28 layers. The 28.5 GiB pool is 44.4 GiB at 0.95 utilization, less 8.1 GiB of weights, less roughly 5.5 GiB for activations and CUDA graphs. Read `KV cache size` in the result file's provenance block for the allocation in force.

`legacy_corpus` sends 38 wire input tokens and asks for up to 150 output tokens, so 512 concurrent sequences hold under 100k tokens, about 5.1 GiB of KV. `--max-num-seqs 512`, not KV memory, caps the batch for this workload.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g6e.12xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Build and push the quantizer image](../../../k8s/examples/README.md#job-quantize-modelyaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export AWS_ACCOUNT_ID=<your-account-id>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen25-7b-fp8-g6e-12xlarge
   export INSTANCE_TYPE=g6e.12xlarge
   ```

2. Install the `eks-bench` CLI into a Python virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e '.[tokenizer]'
   ```

3. Configure the EKS context and namespace, install the NVIDIA device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

4. Quantize the model to ModelOpt FP8 and save the checkpoint to the `fsx-pvc` claim.
   ```bash
   ./model-jobs/qwen2.5-7b/fp8-model-quantize.sh
   ```

5. Deploy the vLLM inference server and confirm it serves the model.

   ```bash
   ./vllm-recipes/qwen2.5-7b/fp8-g6e.12xlarge-deploy.sh
   ```

6. Generate the accuracy gate (.JSON) that the benchmark checks against.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-12xl-fp8 \
       --mode fingerprint \
       --model-tag qwen2.5-7b --config fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --out-dir docs/results
   ```

7. Dry-run the exact in-cluster benchmark.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-12xl-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.12xlarge --model-tag qwen2.5-7b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

8. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-12xl-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.12xlarge --model-tag qwen2.5-7b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/qwen2.5-7b/fp8-g6e.12xlarge.yaml
```

---


## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **The checkpoint stays on the claim.** A later run can skip Quickstart step 4.
- **`QUANTIZATION=modelopt`** Serves the ModelOpt FP8 export the quantize step writes, using its `hf_quant_config.json` and calibrated input scales.
- **16-bit KV cache.** The checkpoint carries no KV scales (`kv_cache_quant_algo` is `None`), and with `--kv-cache-dtype fp8_e4m3` the accuracy gate rejects the server because answers collapse into repetition.
- **`--no-enable-prefix-caching`** `legacy_corpus` repeats five fixed prompts, so a cache-enabled server measures cache hits rather than prefill. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`--enable-chunked-prefill`** Breaks long prefills into chunks so they share a step with decodes.
- **`--gpu-memory-utilization=0.95`** Leaves the rest of the 48 GB card for activations and CUDA graphs.
- **`nvidia.com/gpu: 1`** The 7B model fits one of the four GPUs, so the pod takes one.
- **`strategy: Recreate`** Stops the old pod before starting its replacement, so a redeploy never holds a second GPU.
- **`VLLM_LOGGING_LEVEL=INFO`** Logs the KV cache size and prefix cache hit rate the provenance block reads.
