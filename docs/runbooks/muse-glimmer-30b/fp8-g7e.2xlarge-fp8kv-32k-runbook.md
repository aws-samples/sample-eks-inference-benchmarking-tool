# Runbook for Muse-Glimmer-30B on g7e.2xlarge, FP8 weights and FP8 KV cache at 32k context

Serves a locally quantized ModelOpt FP8 build of [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) on one g7e.2xlarge (1x RTX PRO 6000 Blackwell, sm_120) with an FP8 KV cache and a 32,768-token context window, and measures the closed-loop throughput and latency tables on the long-context coding profile.

This recipe is the direct counterpart of [`gemma4-31b/fp8-g7e.2xlarge-fp8kv-32k-runbook.md`](../gemma4-31b/fp8-g7e.2xlarge-fp8kv-32k-runbook.md): same GPU, same context, same profile, same KV precision, same backend, so the two files read side by side compare the models and nothing else.

## Capacity sizing

Muse-Glimmer-30B is ~29.6B parameters (including a 1.8B perception encoder), ~32 GiB at FP8 with 16-bit embeddings. The deciding number is its KV geometry: **2 KV heads at head dimension 128** (GQA 16:1) across 52 layers is ~26 KiB per token at FP8 counting every layer uniformly, which is how the engine allocates — about 18x lighter than Gemma 4 31B's 480 KiB. A 32k sequence holds ~0.74 GiB of KV, so the ~46 GiB pool holds roughly **60 full-length sequences**, against 3.78 for Gemma on the same card. This model is compute-bound at long context, not KV-bound.

Read `KV cache size` and `Max concurrency` from the deploy script's output for the allocation actually in force, and correct this section if it disagrees; the Gemma sibling's formula-derived sizing was off by 4x until measured.

| Property | Value | Source |
|---|---|---|
| **Parameters** | ~29.6B, of which ~1.8B perception encoder | Model card |
| **Layers** | 52, of which 13 full attention and 39 sliding at window 2048 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 2, 128 | `config.json` |
| **Checkpoint size, FP8** | ~32 GiB | 1 byte per parameter, embeddings stay 16-bit |
| **Instance** | g7e.2xlarge, 1x RTX PRO 6000 Blackwell at 96 GB | 96 GB nominal, 89.4 GiB; 8 vCPU, 64 GiB host RAM |
| **Tensor parallel size** | 1 | one GPU; 2 KV heads make TP>1 replicate KV anyway |
| **Weight precision** | FP8 E4M3, static input scales | `QUANTIZATION=modelopt`, `hf_quant_config.json` |
| **KV cache precision** | FP8 E4M3, uncalibrated scales | `KV_CACHE_DTYPE=fp8`; see Settings |
| **Attention backend** | TRITON_ATTN, pinned | `--attention-backend`; see Settings |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 64 | `MAX_NUM_SEQS` |
| **Max batched tokens** | 32,768 | `MAX_NUM_BATCHED_TOKENS` |

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.2xlarge` node group](../../../k8s/examples/eksctl-nodegroup-g7e-2xl-spot.yaml) (one subnet, in your FSx filesystem's availability zone)
- [Create a benchmark-client node group](../../../k8s/examples/eksctl-nodegroup-bench-client.yaml) (the measurement Job requests 8 dedicated CPUs, which an 8 vCPU node cannot supply)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Quantize the checkpoint to `/models/quantized/Muse-Glimmer-30B-fp8` on that claim](../../../model-jobs/muse-glimmer-30b/fp8-model-quantize.yaml) (~45 GPU-minutes; the Job pins the serving instance type, so calibration runs on the GPU the result is measured on)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-muse-glimmer-30b-fp8kv-32k-g7e-2xlarge
   export INSTANCE_TYPE=g7e.2xlarge
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

4. Deploy and validate the model server. The script fails if the KV cache is not FP8 or the attention backend is not TRITON_ATTN, so a server that started on the wrong settings never reaches a result file.

   ```bash
   ./vllm-recipes/muse-glimmer-30b/fp8-g7e.2xlarge-fp8kv-32k-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-fp8kv-32k \
       --mode fingerprint \
       --model-tag muse-glimmer-30b --config fp8kv-32k \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-fp8kv-32k \
       --protocol corpus --profile coding_repo_context \
       --instance g7e.2xlarge --model-tag muse-glimmer-30b \
       --config fp8kv-32k --quant fp8 \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --tokenizer meta-models/Muse-Glimmer-30B \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=muse-glimmer-30b-vllm-g7e-fp8kv-32k -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Benchmark in-cluster. This produces the closed-loop throughput table and the streaming TTFT/TPOT/ITL table in one result file.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-fp8kv-32k \
       --protocol corpus --profile coding_repo_context \
       --instance g7e.2xlarge --model-tag muse-glimmer-30b \
       --config fp8kv-32k --quant fp8 \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --tokenizer meta-models/Muse-Glimmer-30B \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=muse-glimmer-30b-vllm-g7e-fp8kv-32k -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/muse-glimmer-30b/fp8-g7e.2xlarge-fp8kv-32k.yaml
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name g7e-2xl-spot \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name bench-client-cpu \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```

The checkpoint on `fsx-pvc` survives, so a later run needs no requantization.

---

## Settings

- **`image: vllm/vllm-openai:v0.29.0`.** The registry in this build lists `MuseGlimmerForConditionalGeneration` natively, so serving needs no image work. Note the quantize Job cannot use the repo's quantizer image for this model: that image's transformers (5.3.0) predates `model_type: muse_glimmer` and its frozen dependency set rejects an upgrade, so the Job quantizes inside this serving image instead; the Job file records the details.

- **`--quantization modelopt`.** Real ModelOpt FP8 export with static input scales, produced by [the committed quantize Job](../../../model-jobs/muse-glimmer-30b/fp8-model-quantize.yaml). The vanilla `fp8` loader mis-reads ModelOpt exports and produces garbage; the gemma4-31b g7.24xlarge runbook records the incident.

- **`--kv-cache-dtype fp8`, uncalibrated.** vLLM 0.29.0 rejects `--calculate-kv-scales`, so the server logs `Using KV cache scaling factor 1.0 for fp8_e4m3` at startup. Applies equally to the Gemma sibling, so it does not bias the model-to-model comparison, but it bounds accuracy claims from either.

- **`--attention-backend TRITON_ATTN`, pinned, passed as a flag.** 0.29.0 ignores `VLLM_ATTENTION_BACKEND` (logs `Unknown vLLM environment variable detected`). TRITON_ATTN matches the Gemma sibling, where the engine refused FLASHINFER for the multimodal token pattern and refuses FLASH_ATTN for FP8 KV on sm_120. This model also carries a vision encoder, so the same FLASHINFER refusal is expected; the deploy script asserts what actually served either way.

- **`--max-num-seqs 64`.** On the Gemma sibling this is an unreachable ceiling; here the pool genuinely holds ~60 full-length sequences, so 64 is a real limit and the closed-loop sweep's top level (512) will queue behind it. That is the intended shape: the serving unit under test is one card as configured, and 64 already exceeds the 13–22 in-flight the customer workload implies.

- **Speculative decoding is off, deliberately.** The model ships with a companion drafter (`DFlashMuseGlimmerAssistantModel` is in the vLLM registry, and the model card advertises it). Speculation is always its own labelled config in this repo, never folded into a baseline; a `-dflash` sibling recipe is the way to measure it.

- **`--no-enable-prefix-caching`, as in every measured recipe.** Confirm `Prefix cache hit` reads 0.0% in the provenance block.

- **The profile is directionally right for the reference workload, not a match.** `coding_repo_context` sends ~20.3k input tokens against the workload's ~25–30k, and caps output at 1,024 against its ~5k. Both gaps flatter per-card capacity. They apply identically to the Gemma sibling, so the model-to-model comparison stands; absolute fleet sizing from either file is a floor.
