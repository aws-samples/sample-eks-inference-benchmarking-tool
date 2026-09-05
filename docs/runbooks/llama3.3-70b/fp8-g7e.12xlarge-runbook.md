# Llama 3.3 70B Instruct FP8 on g7e.12xlarge

Serves [`nvidia/Llama-3.3-70B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8) on g7e.12xlarge (2x RTX PRO 6000 Blackwell, sm_120) with 2 GPUs per pod.

## Capacity sizing

Llama 3.3 70B Instruct at FP8 occupies 67.7 GiB of the 178.8 GiB on two RTX PRO 6000 Blackwell cards, leaving roughly 81 GiB for 16-bit KV cache after the recipe's 0.90 memory fraction and an estimated 12 GiB runtime reserve. At the recipe's 8,192-token context that pool holds roughly 32 full-context sequences; the server's `/metrics` allocation captured in step 3 is authoritative.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 70B | Model card and safetensors metadata |
| **Layers** | 80 | `config.json`, `num_hidden_layers` |
| **KV heads, head dimension** | 8, 128 | `config.json` |
| **Checkpoint size, FP8** | 67.7 GiB | `model.safetensors.index.json`, `metadata.total_size` |
| **Instance** | g7e.12xlarge, 2x RTX PRO 6000 Blackwell at 96 GB | 192 GB nominal, 178.8 GiB |
| **Tensor parallel size** | 2 | `TP_SIZE` |
| **Weight precision** | FP8 | checkpoint `hf_quant_config.json`, `quant_algo` |
| **KV cache precision** | 16-bit BF16 | `KV_CACHE_DTYPE` |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 4 KiB with 16-bit KV here. Every layer attends to the full context, so one 8,192-token sequence consumes 2.5 GiB across both tensor-parallel ranks. Use aggregate memory with that whole-sequence cost; dividing both memory and KV by TP would count tensor parallelism twice.

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=llama33-70b-fp8-g7e12xlarge
   export INSTANCE_TYPE=g7e.12xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Deploy and validate the model server. The model checkpoint downloads into pod-local (`emptyDir`) storage and is downloaded again after pod replacement.

   ```bash
   ./vllm-recipes/llama3.3-70b/fp8-g7e.12xlarge-deploy.sh
   ```

4. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service llama33-70b-vllm-g7e-fp8 \
       --mode fingerprint \
       --model-tag llama3.3-70b --config fp8 \
       --model-id llama3.3-70b-fp8 \
       --out-dir docs/results
   ```

5. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service llama33-70b-vllm-g7e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag llama3.3-70b \
       --config fp8 --quant fp8 \
       --model-id llama3.3-70b-fp8 \
       --tokenizer nvidia/Llama-3.3-70B-Instruct-FP8 \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

6. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service llama33-70b-vllm-g7e-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag llama3.3-70b \
       --config fp8 --quant fp8 \
       --model-id llama3.3-70b-fp8 \
       --tokenizer nvidia/Llama-3.3-70B-Instruct-FP8 \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/llama3.3-70b/fp8-g7e.12xlarge.yaml
```

## Settings

- **`image: vllm/vllm-openai:v0.26.0`** Uses the repository's current upstream vLLM release rather than inheriting the older g7e DLC tag.
- **`nvidia/Llama-3.3-70B-Instruct-FP8`** NVIDIA's published checkpoint contains Model Optimizer FP8 weights. The [vLLM Llama 3.3 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Llama/Llama3.3-70B.html) names it for Blackwell and Hopper.
- **No `--quantization` flag** vLLM auto-detects the checkpoint's Model Optimizer metadata, matching the official launch command; forcing the generic runtime-FP8 loader would override a serialized quantization format unnecessarily.
- **`- --tensor-parallel-size=2`** Uses both 96 GB GPUs because the 67.7 GiB checkpoint leaves too little room for runtime state and KV cache on one 89.4 GiB card; TP2 is required for this g7e serving shape, not asserted as its measured optimum.
- **`- --kv-cache-dtype=bfloat16`** Overrides the checkpoint's FP8-KV metadata explicitly. On the live v0.26 g7e load, automatic FP8 KV warned that calibrated q/prob scales were unavailable and could affect accuracy, so only the weights remain FP8.
- **`- --gpu-memory-utilization=0.90`** Matches vLLM's safer default for the first live load of this 70B checkpoint and leaves headroom for CUDA graphs and activations; raise it only after the live allocation and startup are stable.
- **`- --max-model-len=8192`** Pins the context budget for comparable measurements rather than inheriting the model's 131,072-token maximum.
- **`- --max-num-seqs=512`** Aligns the scheduler admission ceiling with the highest benchmark concurrency. In the three-rep corpus sweep, replacing the initial 128-sequence ceiling raised c512 median output throughput and reduced p50 end-to-end latency.
- **`- --max-num-batched-tokens=8192`** Matches the official Blackwell recipe and remains equal to `MAX_MODEL_LEN`.
- **`- --enable-chunked-prefill`** Makes the prefill behavior explicit and matches every other single-node g7e recipe in this repository.
- **`- --async-scheduling`** Matches the official recipe and reduces host scheduling gaps. This recipe does not use the incompatible structured-output, speculative-decoding, or pipeline-parallel features.
- **No `--enforce-eager`** Keeps CUDA graphs enabled. The existing Qwen2.5-32B recipe is the measured g7e exception; do not copy its crash workaround here without reproducing that failure.
- **`- --no-enable-prefix-caching`** Keeps the benchmark baseline free of cache-hit inflation.
- **No `--attention-backend` override** v0.26 selects a Blackwell-compatible backend automatically; the older `VLLM_ATTENTION_BACKEND` environment variable is not recognized by this release.
- **`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,NET`** Records NCCL initialization and the selected intra-node transport for result provenance without enabling per-collective logging during the measurement.
- **`model-storage: emptyDir: {sizeLimit: 140Gi}`** Holds the 67.7 GiB checkpoint and its local Hugging Face cache with headroom; the pod requests 100 GiB of schedulable ephemeral storage and caps usage at 140 GiB.
- **Shared-cluster tolerations** Permit this pod to use GPU nodes carrying the repository's benchmark-workload and capacity-reservation taints; the instance-type selector still pins the hardware to `g7e.12xlarge`.
