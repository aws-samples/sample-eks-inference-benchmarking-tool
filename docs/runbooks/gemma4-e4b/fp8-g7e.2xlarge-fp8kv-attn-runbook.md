# Gemma 4 E4B FP8 with an FP8 KV cache on g7e.2xlarge: attention backend A/B

Serves the same locally quantized ModelOpt FP8 build of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) with an FP8 KV cache as [`fp8-g7e.2xlarge-fp8kv-runbook.md`](./fp8-g7e.2xlarge-fp8kv-runbook.md), across three pinned attention backends. Three recipes share this runbook, differing only in `--attention-backend`:

| Arm | Backend | Recipe | What to expect |
|---|---|---|---|
| 1 | `FLASHINFER` | [`fp8-g7e.2xlarge-fp8kv-attn-flashinfer.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-flashinfer.yaml) | Starts and serves. The CUDA fault in vLLM issue #54225 did not reproduce on this build. Its XQA decode path rejects `head_dim=512` and falls back to native FlashInfer decode, which the server logs |
| 2 | `FLASH_ATTN` | [`fp8-g7e.2xlarge-fp8kv-attn-flashattn.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-flashattn.yaml) | **Cannot run on this GPU.** The engine refuses at startup: `Selected backend AttentionBackendEnum.FLASH_ATTN is not valid for this configuration. Reason: ['kv_cache_dtype not supported', 'FP8 KV cache requires FA3 on SM90 or FA4 on SM100']`. This card is sm_120, so neither applies. Keep the arm; the refusal is the finding |
| 3 | `TRITON_ATTN` | [`fp8-g7e.2xlarge-fp8kv-attn-triton.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-triton.yaml) | Starts and serves. Portable path, and the only one of the three besides FlashInfer that accepts an FP8 KV cache here |

With FLASH_ATTN excluded by the engine, the A/B that can actually be measured on this GPU is two arms, not three.

`FLEX_ATTENTION` and `TORCH_SDPA` are valid values in this build but are not included. Both sit last in auto-priority and neither is a decode contender; add an arm if you want them for completeness.

## Capacity sizing

Every arm runs at `--gpu-memory-utilization 0.90`, not the 0.80 the [FP8-KV pair](./fp8-g7e.2xlarge-fp8kv-runbook.md) uses. **Compare arms against each other, not against that pair**, because two things differ between them. Read `KV capacity from` and `KV full-ctx seqs` in each result file's provenance block for the allocation in force.

The sizing that matters here is not the pool. `chat_terse` sends 96 wire input tokens with a 30-token output cap, so a sequence occupies about 126 tokens and the working set at concurrency 512 is a small fraction of the pool at either memory fraction. What the backend changes is the kernel doing the attention arithmetic, and with an FP8 KV cache this build performs that arithmetic in FP8 rather than converting back to 16-bit first, which puts the backend on the critical path.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 4B active, 8B on disk | Model card |
| **Layers** | 42, of which 7 full attention and 35 sliding at window 512 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 2; 256 sliding, 512 full attention | `config.json`, `head_dim` and `global_head_dim` |
| **Instance** | g7e.2xlarge, 1x RTX PRO 6000 Blackwell at 96 GB, sm_120 | 8 vCPU, 64 GiB host RAM |
| **Weight precision** | FP8 E4M3, static input scales | `hf_quant_config.json`, `quant_algo: FP8` |
| **KV cache precision** | FP8 E4M3, uncalibrated scales | `KV_CACHE_DTYPE=fp8`; see Settings |
| **Attention backend** | pinned per arm | `--attention-backend`, from `ATTENTION_BACKEND` |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.2xlarge` node group](../../../k8s/examples/eksctl-nodegroup-g7e-2xl-spot.yaml) (give it every subnet in your FSx filesystem's availability zone; a single-subnet group spent minutes on `InsufficientInstanceCapacity`)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- A node with more than 8 allocatable CPU for the benchmark client, which is not the GPU node (the Job requests 8 dedicated CPUs and an 8 vCPU node offers 7910m, so it stays Pending forever)
- The FP8 checkpoint at `/models/quantized/gemma-4-E4B-it-fp8` on that claim, produced by the quantize Job in [`fp8-g7e.2xlarge-fp8kv-runbook.md`](./fp8-g7e.2xlarge-fp8kv-runbook.md)
- [Export the checkpoint to S3](../../../model-jobs/gemma4-e4b/fp8-model-export.yaml) (only when the claim is in a region the checkpoint was not quantized in)
- [Load it onto the claim in that region](../../../model-jobs/gemma4-e4b/fp8-model-download.yaml) (copy rather than requantize; calibration is not bit-reproducible, so a fresh quantize is different weights)
- A Lustre filesystem at version 2.15 if you create one for this (a 2.10 filesystem is unmountable by the 2.15 client on AL2023 and fails with `mount.lustre ... Invalid argument`)

## Quickstart

Run one arm at a time in the order above. There is one GPU, so tear each arm down before deploying the next.

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-e4b-fp8kv-attn-g7e-2xlarge
   export INSTANCE_TYPE=g7e.2xlarge
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

4. Set the arm you are running, then repeat steps 5 to 9 per arm.
   ```bash
   export ARM=flashinfer          # then flashattn, then triton
   export BACKEND=FLASHINFER      # then FLASH_ATTN, then TRITON_ATTN
   ```

5. Deploy and validate the server for this arm. A start failure is a result for that backend, so record it rather than switching arms silently.
   ```bash
   ./vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-${ARM}-deploy.sh
   ```

6. Confirm the backend actually took effect. The deploy script in step 5 already fails if it did not, so this is for your own record. Expect one line reading `Using AttentionBackendEnum.${BACKEND} backend.`
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs \
       deployment/gemma4-e4b-vllm-g7e-2xl-fp8kv-${ARM} -c vllm \
       | grep -E "Using AttentionBackendEnum|Unknown vLLM environment variable"
   ```

7. Confirm the KV cache is still FP8 and read the pool this memory fraction produced.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec \
       deployment/gemma4-e4b-vllm-g7e-2xl-fp8kv-${ARM} -c vllm -- \
       sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
   ```

8. Generate the accuracy gate (.JSON) for this arm. Each arm is its own config tag, so each gets its own reference.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-g7e-2xl-fp8kv-${ARM} \
       --mode fingerprint \
       --model-tag gemma4-e4b --config fp8kv-attn-${ARM} \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

9. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-g7e-2xl-fp8kv-${ARM} \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8kv-attn-${ARM} --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8kv-${ARM} -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

10. Benchmark in-cluster, then tear the arm down and return to step 4 for the next one.
    ```bash
    eks-bench measure \
        --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
        --service gemma4-e4b-vllm-g7e-2xl-fp8kv-${ARM} \
        --protocol corpus --profile chat_terse \
        --instance g7e.2xlarge --model-tag gemma4-e4b \
        --config fp8kv-attn-${ARM} --quant fp8 \
        --model-id /models/quantized/gemma-4-E4B-it-fp8 \
        --tokenizer google/gemma-4-E4B-it \
        --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
            -l app=gemma4-e4b-vllm-g7e-2xl-fp8kv-${ARM} -o jsonpath='{.items[0].metadata.name}')" \
        --reps 3 --out-dir docs/results
    kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
        -f vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-${ARM}.yaml
    ```

## Tear down

```bash
for ARM in flashinfer flashattn triton; do
  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
      -f vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-${ARM}.yaml --ignore-not-found
done
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name g7e-2xl-spot \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```

Delete the benchmark-client node group too if you created one for this run. The checkpoint on `fsx-pvc` survives, so a later run needs no requantization.

---

## Settings

- **The backend is pinned with `--attention-backend`, not `VLLM_ATTENTION_BACKEND`.** vLLM 0.29.0 does not read that environment variable. It logs `Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND`, carries no such name in `vllm.envs`, and proceeds on its auto-selected backend, so a pin passed that way produces arms that differ in label only. The flag lives under `AttentionConfig` and takes the names in `vllm.v1.attention.backends.registry.AttentionBackendEnum`. The deploy scripts assert the chosen backend appears in the server log and exit non-zero if it does not.

- **The backend is pinned rather than auto-selected at all** because vLLM's auto-priority tables are written for sm_10x B200. This card is sm_120, so which kernel path the engine would pick is not a known default, and leaving it unset makes the backend an unrecorded variable.

- **`FLASH_ATTN` cannot serve an FP8 KV cache on this GPU.** The engine refuses at startup with `Reason: ['kv_cache_dtype not supported', 'FP8 KV cache requires FA3 on SM90 or FA4 on SM100']`. sm_120 is neither SM90 nor SM100, so the arm cannot be measured. The recipe is kept because the refusal, and its reason, is the answer for that backend.

- **`FLASHINFER` may refuse to start, and that would be a result.** vLLM issue #54225 reports a CUDA illegal memory access with the FlashInfer backend on sm_120 with an FP8 KV cache, crashing on a request as small as 16 tokens, with `TRITON_ATTN` noted as unaffected. It did not reproduce on this build: the arm started, served a full sweep, and the container never restarted. Its XQA decode path does reject `head_dim=512` and log a fallback to native FlashInfer decode, so the kernel actually serving the full-attention layers is not XQA.

- **`--gpu-memory-utilization 0.90`, against 0.80 on the FP8-KV pair.** Two things therefore differ between this study and that pair, so a difference against it is not attributable. Within this study the fraction is constant, so arm-to-arm comparisons are clean.

- **KV scales are uncalibrated.** vLLM 0.29.0 rejects `--calculate-kv-scales` and refuses to start, so no scaling factors are supplied and the server logs `Using KV cache scaling factor 1.0 for fp8_e4m3` alongside `Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention`. This bounds any quality comparison, and it applies equally to all three arms so it does not bias the A/B.

- **`--no-enable-prefix-caching`** Held off, as in every recipe here. Confirm `Prefix cache hit` reads 0.0% in each provenance block before publishing any figure as cache-free.

- **g7e.2xlarge spot capacity is thin.** A single-subnet node group spent several minutes returning `InsufficientInstanceCapacity`, and the error text named a different availability zone as having capacity. Because this study needs the node held across three arms, check `aws ec2 get-spot-placement-scores` first and consider on-demand if the score is low.

- **The benchmark client needs its own node.** `measure` runs the client as a Job requesting 8 dedicated CPUs. An 8 vCPU instance advertises 7910m allocatable, so the Job cannot schedule on the GPU node or on an `m5.2xlarge` and sits Pending without an obvious cause.
