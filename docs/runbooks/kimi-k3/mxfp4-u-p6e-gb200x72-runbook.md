# Kimi K3 (2.8T MoE) MXFP4 on u-p6e-gb200x72

Serves [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) on u-p6e-gb200x72 (GB200, sm_100) with 16 GPUs per pod.

## Capacity sizing

This recipe serves a locally quantized checkpoint on 16 GB200 GPUs. Its size is not derivable from the model card, and the recipe enforces 1,500 GiB free at `/models` before it starts. KV cost per sequence is exact and appears below; the pool and concurrency are read from the server.

| Property | Value | Source |
|---|---|---|
| **Layers** | 93 | `config.json`, `num_hidden_layers` |
| **KV latent per layer** | kv_lora_rank 512 plus qk_rope 64 | `config.json` |
| **Checkpoint size** | not derivable, 1,500 GiB free enforced at startup | `MIN_FREE_GIB` |
| **Instance** | u-p6e-gb200x72, GB200 | Per-GPU VRAM not recorded in this repo |
| **Tensor parallel size** | 16 | `TP_SIZE` |
| **Weight precision** | MXFP4 | declared by the checkpoint's `quantization_config` |
| **KV cache precision** | FP8 (`fp8`) | `--kv-cache-dtype fp8` |
| **Context length** | 1,048,576, not pinned by the recipe | `config.json`, `max_position_embeddings` |
| **Memory fraction** | 0.9 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `(kv_lora_rank + qk_rope_head_dim) x bytes_per_element`, which is 0.56 KiB here. Multi-head latent attention caches one compressed latent per layer instead of a key and value per head, so KV cost scales with layer count rather than head count.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `u-p6e-gb200x72` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- The locally quantized main checkpoint already staged in the Hugging Face cache at `/models/.hfcache` on `fsx-pvc`, where the manifest's `HF_HOME` expects it when resolving `moonshotai/Kimi-K3`. This repository does not ship the source-specific staging Job; stage the checkpoint before deployment.

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-kimi-k3-mxfp4-u-p6e-gb200x72
   export INSTANCE_TYPE=u-p6e-gb200x72
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm each remaining prerequisite exists.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   ```

4. Bring up the server. It mounts the checkpoint from the `fsx-pvc` claim; if yours has another name, update `claimName` in your copy of the recipe first.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f vllm-recipes/kimi-k3/mxfp4-u-p6e-gb200x72.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" rollout status deploy/kimi-k3-gb200 --timeout=3600s
   ```

5. Check whether the recipe's server is up and ready to serve.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec deploy/kimi-k3-gb200 -c vllm -- \
     curl -s localhost:8000/v1/models
   ```

6. Read the KV cache the server allocated. Record the reported maximum concurrency with any result.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs deploy/kimi-k3-gb200 -c vllm \
     | grep -E "GPU KV cache size|Maximum concurrency|model weights took"
   ```

7. Prove this recipe passes the flags it claims.

   ```bash
   eks-bench validate-manifests kimi-k3/mxfp4-u-p6e-gb200x72.yaml
   ```

8. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-gb200 \
       --mode fingerprint \
       --model-tag kimi-k3 --config mxfp4 \
       --model-id kimi-k3 \
       --thinking off \
       --out-dir docs/results
   ```

9. Benchmark in-cluster. If needed, repeat per protocol, changing --protocol and --profile.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-gb200 \
       --protocol corpus --profile legacy_corpus \
       --instance u-p6e-gb200x72 --model-tag kimi-k3 \
       --config mxfp4 --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/kimi-k3/mxfp4-u-p6e-gb200x72.yaml
```

---


## Settings

- **`node.kubernetes.io/instance-type: u-p6e-gb200x72`** [FILL] UltraServer instance type / node-group label.

- **`image: vllm/vllm-openai:kimi-k3`** Dedicated K3 tag. Docker-only for now: the build pulls pre-release FlashInfer, so there is no pip path. Do not substitute a generic vllm tag.

- **`(end of file)`** Expect NV# between all pairs. If this shows PIX/PHB/NODE we are NOT in a single NVLink domain and the MoE all-to-all is on the wrong transport.

**`ports:`**

FLAG NOTES (all from the official recipe):

```
--trust-remote-code           REQUIRED - KDA/AttnRes/LatentMoE are custom.
--load-format fastsafetensors REQUIRED in practice: 1560 GB load.
--moe-backend flashinfer_trtllm   recommended for TP > 1.
```

(use deep_gemm_mega_moe for DEP setups)

```
--all2all-backend flashinfer_nvlink_one_sided  <- NVLink. Use deepep_v2
```

only for RDMA/EFA (see the b200 manifest).

```
--kv-cache-dtype fp8          requires the use_prefill_query_quantization
```

attention-config flag, set above.

```
NO --enable-prefix-caching    deliberate. Off by default for K3; leaving it
```

off keeps throughput numbers honest. Speculative decoding (DSpark) is a SEPARATE labelled config - it ~3x's single-stream decode and must never be mixed into a baseline. Add:

```
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark",
```

"num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA", "draft_sample_method":"probabilistic","rejection_sample_method":"block"}'

- **`- name: VLLM_ENGINE_READY_TIMEOUT_S`** ---- required/recommended engine env from the recipe ----

- **`- name: INSTANCE_TYPE`** ---- provenance for the result record ----

- **`nvidia.com/gpu: "16"`** [FILL] must match TP_SIZE (x DP if used).
- **`TP_SIZE=16`** [VERIFY] 8 or 16 per the recipe / GPU count
- **`MAX_MODEL_LEN=auto`** recipe uses auto; pin a value for long-context runs
- **`VLLM_ENGINE_READY_TIMEOUT_S=3600`** engine init on a 2.8T model far exceeds the default
- **`VLLM_USE_RUST_FRONTEND=1`** fully supports K3; needed for the render/parse path
- **`failureThreshold: 240`** 60 min; matches VLLM_ENGINE_READY_TIMEOUT_S intent
