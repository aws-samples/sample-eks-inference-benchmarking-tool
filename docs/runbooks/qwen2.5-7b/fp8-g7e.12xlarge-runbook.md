# Qwen2.5-7B-Instruct FP8 on g7e.12xlarge

Serves [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on g7e.12xlarge (2x RTX PRO 6000 Blackwell, sm_120) with 1 GPU per pod.

## Capacity sizing

Qwen2.5-7B-Instruct at ModelOpt FP8 occupies 8.1 GiB of the 95.6 GiB on 1 RTX PRO 6000 Blackwell card. At 0.95 utilization that leaves at most 82.7 GiB, before activations and CUDA graphs, which is room for fewer than 47 sequences at the model maximum 32,768-token context.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 7B | Model card |
| **Layers** | 28 | `config.json`, `num_hidden_layers` |
| **KV heads, head dimension** | 4, 128 | `config.json` |
| **Checkpoint size, FP8** | 8.1 GiB | `model.safetensors` from the quantize step; embeddings and `lm_head` stay BF16 |
| **Instance** | g7e.12xlarge, 1 of 2x RTX PRO 6000 Blackwell at 96 GB | 97,887 MiB as `nvidia-smi` reports it |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | ModelOpt FP8 | `QUANTIZATION` |
| **KV cache precision** | 16-bit | No `--kv-cache-dtype`; see Settings |
| **Context length** | 32,768, not pinned by the recipe | `config.json`, `max_position_embeddings` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 2 KiB here, or 56 KiB per token across all 28 layers, so a full 32,768-token sequence holds 1.75 GiB. Read `KV cache size` in the result file's provenance block for the allocation in force.

`legacy_corpus` sends 38 wire input tokens and asks for up to 150 output tokens, so 512 concurrent sequences hold under 100k tokens, about 5.1 GiB of KV. `--max-num-seqs 512`, not KV memory, caps the batch for this workload.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.12xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Build and push the quantizer image](../../../k8s/examples/README.md#job-quantize-modelyaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export AWS_ACCOUNT_ID=<your-account-id>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen25-7b-fp8-g7e-12xlarge
   export INSTANCE_TYPE=g7e.12xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the claim is bound.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=jsonpath='{.status.phase}'=Bound pvc/fsx-pvc --timeout=2m
   ```

4. Quantize the checkpoint onto the claim at `/models/quantized/Qwen2.5-7B-Instruct-fp8`.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap quantizer-src \
       --from-file=app/quantize/src/quantize.py \
       --from-file=app/quantize/src/config.py \
       --from-file=app/quantize/src/fix_amax.py \
       --from-file=app/quantize/src/verify_checkpoint.py
   sed -e "s|012345678901.dkr.ecr.us-east-1|${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}|" \
       model-jobs/qwen2.5-7b/fp8-model-quantize.yaml \
     | kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f -
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/quantize-qwen25-7b-fp8 --timeout=1h
   ```

5. Deploy and validate the model server mounted to `fsx-pvc`.

   ```bash
   ./vllm-recipes/qwen2.5-7b/fp8-g7e.12xlarge-deploy.sh
   ```

6. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g7e \
       --mode fingerprint \
       --model-tag qwen2.5-7b --config fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --out-dir docs/results
   ```

7. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g7e \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen2.5-7b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

8. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g7e \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen2.5-7b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/qwen2.5-7b/fp8-g7e.12xlarge.yaml
```

---


## Settings

- **`QUANTIZATION=modelopt`** Serves the ModelOpt FP8 export the quantize step writes, using its `hf_quant_config.json` and calibrated input scales.
- **16-bit KV cache.** The checkpoint carries no KV scales (`kv_cache_quant_algo` is `None`), and with `--kv-cache-dtype fp8_e4m3` the accuracy gate rejects the server because answers collapse into repetition.
- **`--no-enable-prefix-caching`** `legacy_corpus` repeats five fixed prompts, so a cache-enabled server measures cache hits rather than prefill. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`--enable-chunked-prefill`** Breaks long prefills into chunks so they share a step with decodes.
- **`--gpu-memory-utilization=0.95`** Leaves the rest of the 96 GB card for activations and CUDA graphs.
- **`nvidia.com/gpu: 1`** The 7B model fits one of the two GPUs, so the pod takes one.
- **`VLLM_LOGGING_LEVEL=INFO`** Logs the KV cache size and prefix cache hit rate the provenance block reads.
