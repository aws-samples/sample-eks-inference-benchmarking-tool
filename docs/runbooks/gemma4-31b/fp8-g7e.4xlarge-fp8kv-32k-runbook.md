# Runbook for Gemma 4 31B on g7e.4xlarge, FP8 weights and FP8 KV cache at 32k context

Serves the locally quantized ModelOpt FP8 build of [`google/gemma-4-31b-it`](https://huggingface.co/google/gemma-4-31b-it) on one g7e.4xlarge (1x RTX PRO 6000 Blackwell, sm_120) with an FP8 KV cache and a 32,768-token context window, and measures it open-loop. The result is SLA-safe requests per second for one card, which is the number a single-replica fleet is sized from.

This is the long-context sibling of [`fp8-g7e.4xlarge-runbook.md`](./fp8-g7e.4xlarge-runbook.md). Four things differ, each explained under Settings: the vLLM image, `--kv-cache-dtype fp8`, `--max-model-len 32768`, and `--no-enable-prefix-caching`. Its numbers are not comparable to that runbook's.

## Capacity sizing

gemma-4-31b-it at FP8 occupies 30.4 GiB of the 89.4 GiB on 1 RTX PRO 6000 Blackwell card, leaving 47.1 GiB for KV cache. At FP8 KV precision and this recipe's 32,768 token context that pool holds 16 concurrent full-length sequences, and about 23 at the ~21.3k tokens a `coding_repo_context` request occupies. At 16-bit KV those figures would be 8 and 11, which is why the KV cache is FP8 here.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 31B | Model card |
| **Layers** | 60, of which 10 full attention and 50 sliding at window 1024 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 16, 256 | `config.json` |
| **Checkpoint size, FP8** | 30.4 GiB | 1 byte per parameter, embeddings stay 16-bit |
| **Instance** | g7e.4xlarge, 1x RTX PRO 6000 Blackwell at 96 GB | 96 GB nominal, 89.4 GiB; 16 vCPU, 128 GiB host RAM |
| **Tensor parallel size** | 1 | one GPU, no `--tensor-parallel-size` |
| **Weight precision** | FP8 E4M3, static input scales | `QUANTIZATION=modelopt`, `hf_quant_config.json` |
| **KV cache precision** | FP8 E4M3, uncalibrated scales | `KV_CACHE_DTYPE=fp8`; see Settings |
| **Attention backend** | FLASHINFER, pinned | `--attention-backend`, from `ATTENTION_BACKEND` |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 64 | `MAX_NUM_SEQS` |
| **Max batched tokens** | 32,768 | `MAX_NUM_BATCHED_TOKENS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 8 KiB at FP8 and 16 KiB at 16-bit. Gemma 4 interleaves attention, so the 50 sliding layers hold at most 1,024 tokens each whatever the context length (0.39 GiB per sequence at FP8) and per-sequence cost grows only through the 10 full attention layers (2.5 GiB at 32,768 tokens, 1.63 GiB at 21.3k). The 47.1 GiB pool is 89.4 GiB at 0.95 utilization, less 30.4 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs. Read `KV cache size` and `Max concurrency` in the result file's provenance block for the allocation the engine actually made.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.4xlarge` node group](../../../k8s/examples/eksctl-nodegroup-g7e-4xl-spot.yaml) (one subnet, in your FSx filesystem's availability zone)
- [Create a benchmark-client node group](../../../k8s/examples/eksctl-nodegroup-bench-client.yaml) (the measurement Job requests 8 dedicated CPUs, which an 8 vCPU node cannot supply)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Quantize the checkpoint to `/models/quantized/gemma-4-31b-it-fp8` on that claim](../../../k8s/examples/job-quantize-model.yaml) (ModelOpt FP8 with `MODEL_ID=google/gemma-4-31b-it`; the recipe's `modelopt` loader requires a real ModelOpt export)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-31b-fp8kv-32k-g7e-4xlarge
   export INSTANCE_TYPE=g7e.4xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the claim holding the quantized weights is bound.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=jsonpath='{.status.phase}'=Bound pvc/fsx-pvc --timeout=2m
   ```

4. Deploy and validate the model server. The script fails if the KV cache is not FP8 or the attention backend is not FLASHINFER, so a server that started on the wrong settings never reaches a result file.

   ```bash
   ./vllm-recipes/gemma4-31b/fp8-g7e.4xlarge-fp8kv-32k-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-31b-vllm-g7e-fp8kv-32k \
       --mode fingerprint \
       --model-tag gemma4-31b --config fp8kv-32k \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --out-dir docs/results
   ```

6. Dry-run the open-loop capacity search. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-31b-vllm-g7e-fp8kv-32k \
       --protocol server_qps --profile coding_repo_context \
       --qps-levels 0.5,0.75,1,1.5 --ttft-target 5000 \
       --instance g7e.4xlarge --model-tag gemma4-31b \
       --config fp8kv-32k --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-31b-vllm-g7e-fp8kv-32k -o jsonpath='{.items[0].metadata.name}')" \
       --reps 1 --out-dir docs/results \
       --dry-run
   ```

7. Run the open-loop capacity search. Expect roughly two and a half hours for one rep at these four levels; see Settings for why.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-31b-vllm-g7e-fp8kv-32k \
       --protocol server_qps --profile coding_repo_context \
       --qps-levels 0.5,0.75,1,1.5 --ttft-target 5000 \
       --instance g7e.4xlarge --model-tag gemma4-31b \
       --config fp8kv-32k --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-31b-vllm-g7e-fp8kv-32k -o jsonpath='{.items[0].metadata.name}')" \
       --reps 1 --out-dir docs/results
   ```

8. Once the first rep shows where the SLA boundary sits, rerun with three reps and levels that bracket it, so the quoted rate carries a spread.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-31b-vllm-g7e-fp8kv-32k \
       --protocol server_qps --profile coding_repo_context \
       --qps-levels <two levels below the boundary>,<the boundary>,<one level above> --ttft-target 5000 \
       --instance g7e.4xlarge --model-tag gemma4-31b \
       --config fp8kv-32k --quant fp8 \
       --model-id /models/quantized/gemma-4-31b-it-fp8 \
       --tokenizer google/gemma-4-31b-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-31b-vllm-g7e-fp8kv-32k -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/gemma4-31b/fp8-g7e.4xlarge-fp8kv-32k.yaml
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name g7e-4xl-spot \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name bench-client-cpu \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```

The checkpoint on `fsx-pvc` survives, so a later run needs no requantization.

---

## Settings

- **`image: vllm/vllm-openai:v0.29.0`, not the `vllm:0.21.0` DLC the other 31B recipes use.** FP8 KV on Gemma 4 fails on 0.21.0 with `AssertionError: A non 1.0 q_scale is not currently supported`, as recorded in the 8k sibling's Settings, and FP8 KV is the point of this recipe. 0.29.0 serves Gemma 4 with an FP8 KV cache (proven on the E4B recipes), needs no `vllm-sitecustomize` ConfigMap, and exposes the backend pin as a flag. The cost is comparability: nothing here should be read against the 0.21.0 runs.

- **`--quantization modelopt`.** The checkpoint is a real ModelOpt FP8 export with `float8_e4m3fn` weights, `hf_quant_config.json` and static `input_scale`. The vanilla `fp8` loader mis-reads it and produces garbage output; the g7.24xlarge runbook records the incident.

- **`--kv-cache-dtype fp8`, the KV precision under test.** At 30k tokens a sequence holds about 5.4 GiB of KV at 16-bit and 2.7 GiB at FP8, so on one card the KV cache is the binding constraint and FP8 doubles the sequences in flight. The scales are uncalibrated: vLLM 0.29.0 rejects `--calculate-kv-scales` and refuses to start, so the server logs `Using KV cache scaling factor 1.0 for fp8_e4m3` at startup. That bounds any accuracy claim taken from this recipe. Confirm `cache_dtype="fp8"` in `vllm:cache_config_info`; the deploy script fails if it is absent.

- **`--attention-backend FLASHINFER`, pinned, passed as a flag.** vLLM 0.29.0 does not read `VLLM_ATTENTION_BACKEND`; it logs `Unknown vLLM environment variable detected` and runs its auto-selected backend, so a pin passed that way records nothing. Pinned because vLLM's auto-priority tables are written for sm_10x B200 and this card is sm_120. FLASHINFER matches the 8k sibling and is the long-context kernel. Do not carry over the E4B finding that TRITON_ATTN decoded 8–13% faster: that was 96-token prompts on a 4B model. FLASH_ATTN cannot serve an FP8 KV cache on sm_120 at all; the engine requires FA3 on SM90 or FA4 on SM100. The deploy script asserts `Using AttentionBackendEnum.FLASHINFER backend` in the server log. To A/B, copy this recipe with `ATTENTION_BACKEND=TRITON_ATTN` and nothing else changed.

- **`--max-model-len 32768`, the context under test.** `coding_repo_context` measures 20,296 wire tokens median on this tokenizer (nominal 16,384; the profile runs +24% over) with prompts to 21,098, plus 1,024 output, so 24,576 would fit but leaves no room for a customer trace at the top of its range. 32,768 is the window the [pinned agentic trace](../../../app/bench/data/traces/README.md) was scaled for.

- **`--max-num-seqs 64` and `--max-cudagraph-capture-size 64`, down from the sibling's 512.** The pool holds 16 full-length sequences and about 23 at the profile's size, so 512 is unreachable and capturing graphs to 512 spends memory that could hold KV. 64 covers every reachable batch with margin for short requests.

- **`--max-num-batched-tokens 32768`.** A full 21k-token prompt prefills in one chunk. Held at the sibling's value.

- **`--gpu-memory-utilization 0.95`.** Held at the sibling's value so memory fraction is not a variable between them.

- **`--no-enable-prefix-caching`, as in every measured recipe here.** The 8k `fp8-g7e.4xlarge.yaml` sibling still serves with caching on, which is a second reason its numbers are not comparable. Confirm `Prefix cache hit` reads 0.0% in the provenance block. Production agentic traffic at 30k carries a large shared repository prefix, so a caching server would need materially less prefill; the fleet size this run produces is therefore a ceiling on what the workload needs, not a target.

- **`--async-scheduling`.** Overlaps scheduling with the forward pass. Fine on 0.29.0; the llama3.3-70b note about it conflicting with speculation does not apply here because nothing speculates.

- **Open-loop `server_qps`, not the closed-loop `corpus` sweep.** The question is fleet size for an arrival rate, and closed-loop concurrency understated SLA-safe capacity by 26% on this model. `server_qps` issues Poisson arrivals at each target rate and reports whether TTFT and TPOT p99 hold. Fleet size is then `arrival rate ÷ SLA-safe rate per card`, rounded up, times a peak factor. At the customer's 1.4M input tokens per minute and ~30k tokens per request that arrival rate is 0.78 requests per second.

- **`--qps-levels 0.5,0.75,1,1.5`, not the shipped `1,2,3,4,5,6,8,16`.** Those levels were set for `coding_completion` at 256 output tokens. Here a request holds its KV slot for roughly 22–33 seconds (a 20k-token prefill plus 1,024 decode steps at 20–30 ms), and about 23 slots exist, so the card saturates near 0.7–1.0 requests per second. The four levels bracket that with one saturated level to establish the ceiling. Move them once the first rep shows where the boundary actually is.

- **`--ttft-target 5000`, overriding the shipped 500 ms.** Prefilling 20k tokens on one card takes on the order of 1.5–2.5 seconds before any queueing, so the shipped target fails at every rate and measures nothing. 5,000 ms is a placeholder that admits one queued prefill ahead; substitute the customer's real TTFT SLA when known. The override is appended to the result filename so it cannot overwrite a run at the defaults. TPOT p99 stays at the shipped 50 ms.

- **Run time is set by `MIN_QUERY_COUNT=1500`, which the CLI cannot lower.** LoadGen issues at least 1,500 requests per level, so a level at 0.5 requests per second takes 50 minutes and any saturated level drains at the server's own rate regardless of target, about 30 minutes. Four levels at one rep is roughly two and a half hours; three reps at three levels is longer still. Budget the node accordingly.

- **The profile is directionally right, not a match.** `coding_repo_context` sends ~20.3k input tokens and caps output at 1,024. The customer's traffic backs out to ~25–30k input and ~5k output (250K output tokens per minute over 46.7 requests per minute). Residency, KV occupancy and per-card throughput all scale with output length, so the SLA-safe rate this run produces will overstate per-card capacity, plausibly by 3–4x, and the fleet size derived from it is a floor. A sibling profile with `output_tokens` near 5,120 would close the gap and is a protocol change: a new pinned file, never an edit to this one.

- **g7e.4xlarge capacity is thin and single-AZ.** FSx for Lustre fixes the zone, so the node group has one subnet. On-demand needs G/VT quota headroom of 16 vCPU; spot placement scores for this type ran 1/10 across the day, and a sibling type spent 26 minutes in `UnfulfillableCapacity` before a run was abandoned. Check `aws ec2 get-spot-placement-scores` first, and do not create the client node until the GPU node exists.
