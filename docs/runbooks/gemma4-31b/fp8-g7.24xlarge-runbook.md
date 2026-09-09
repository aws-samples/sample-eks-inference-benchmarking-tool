# Runbook for Gemma 4 31B on g7.24xlarge, FP8

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

- [Install the open NVIDIA kernel module if the node advertises no GPUs](../../../k8s/examples/pod-diagnostic-probe.yaml)
- [Create the `vllm-sitecustomize` ConfigMap](../../../k8s/examples/configmap-vllm-sitecustomize.yaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-31b-fp8-g7-24xlarge
   export INSTANCE_TYPE=g7.24xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the FSx claim containing the ModelOpt-quantized weights and the sitecustomize ConfigMap exist in the namespace.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get configmap vllm-sitecustomize
   ```

4. Deploy and validate the model server. The server mounts `/models/quantized/gemma-4-31b-it-fp8` from `fsx-pvc`; update `claimName` first if your weights use a differently named PVC. The recipe uses `Recreate`, so reapplying it causes a gap with no server rather than overlapping GPU claims.

   ```bash
   ./vllm-recipes/gemma4-31b/fp8-g7.24xlarge-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7-fp8 \
       --mode fingerprint \
       --model-tag gemma4-31b --config fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g7.24xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g7.24xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results
   ```

8. Dry-run the optional NVIDIA AIPerf cross-client calibration with identical arguments.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7-fp8 \
       --protocol corpus_aiperf \
       --instance g7.24xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

9. Run the optional NVIDIA AIPerf cross-client calibration. It uses the same server, profile, and canonical levels and requires the AIPerf extras.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7-fp8 \
       --protocol corpus_aiperf \
       --instance g7.24xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results
   ```

10. Dry-run the optional interactive capacity search with identical arguments. The full run can take several hours for 3 reps.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7-fp8 \
       --protocol server_qps --profile coding_completion \
       --instance g7.24xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

11. Run the optional interactive capacity search.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-31b-vllm-g7-fp8 \
       --protocol server_qps --profile coding_completion \
       --instance g7.24xlarge --model-tag gemma4-31b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-31b/fp8-g7.24xlarge.yaml
```

## Notes

* Step 7 runs at the shipped targets, TTFT p99 500 ms and TPOT p99 50 ms. `--ttft-target` and `--tpot-target` override them, and an overridden value is appended to the result filename so a run at different targets cannot overwrite one at the defaults. Read [Trusting the open-loop verdict](../../guides/REFERENCE.md#trusting-the-open-loop-verdict) before quoting a rate as sustainable.
**This recipe serves without `--enable-prefix-caching`, and `legacy_corpus` repeats five fixed prompts.** Those two facts go together: a repeating prompt set against a cache-enabled server measures cache hits rather than serving, so the flag is off here and the throughput figure reflects prefill actually being done. The other Gemma recipes, on g6e.12xlarge and g7e.4xlarge, still serve with the flag on, so their `corpus` numbers are not directly comparable to this recipe's. Add the flag back for a serving deployment, and leave it off for a measured one.
Check the `Prefix cache hit` line in the result file's provenance block regardless: anything other than 0% means prefill was skipped and throughput is inflated. On a pod that has been up for days that line rotates out of the log window and records as `(not found in server log)`, which is not the same as 0%, so a fresh server is worth more than it looks.

---


## Settings

- **`strategy:`** This pod claims all 4 GPUs and at one replica maxUnavailable rounds to 0, so Kubernetes waits on a new pod that can never schedule while the old one holds every GPU. Any edit to this recipe would deadlock.

- **`- --quantization=modelopt`** Real modelopt FP8 export (float8_e4m3fn weights + hf_quant_config.json + static input_scale). Served via vLLM's modelopt loader, NOT the vanilla "fp8" loader. The old "fp8" value mis-read the (previously mislabeled) checkpoint and produced garbage output.

- **`- --gpu-memory-utilization=0.90`** Left at 0.90 rather than tuned for this precision, so precision is the only variable in a comparison. No FP8 KV cache: vLLM 0.21 with Gemma 4 raises the q_scale assertion when the KV cache is quantized.

- **`- --enable-chunked-prefill`** No --enable-prefix-caching: the corpus protocols repeat a five-prompt set, so a warm cache skips prefill and measures caching rather than serving. Turn it on for a serving deployment, off for a measured one.

- **`- --port=8000`** No --enforce-eager: CUDA graphs on (Blackwell sm_120, FlashInfer cubins).
- **`nvidia.com/gpu: 4`** g7.24xlarge = 4x NVIDIA RTX PRO 4500 Blackwell, 32GB each (sm_120)
- **`- --tensor-parallel-size=4`** 31B sharded across all 4 GPUs (32GB each)
- **`- --max-model-len=8192`** Capped well below the model's 262144, so context length is fixed rather than a variable in any measurement.
- **`VLLM_ATTENTION_BACKEND=FLASHINFER`** required for Blackwell sm_120
