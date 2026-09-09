# Kimi K3 (2.8T MoE) MXFP4, lws-dspark on p6-b200.48xlarge

Serves [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) on p6-b200.48xlarge (8x B200, sm_100) with 8 GPUs per pod.

## Capacity sizing

This recipe serves a locally quantized checkpoint on 16 B200 GPUs. Its size is not derivable from the model card, and the recipe enforces 1,500 GiB free at `/models` before it starts. KV cost per sequence is exact and appears below; the pool and concurrency are read from the server.

| Property | Value | Source |
|---|---|---|
| **Layers** | 93 | `config.json`, `num_hidden_layers` |
| **KV latent per layer** | kv_lora_rank 512 plus qk_rope 64 | `config.json` |
| **Checkpoint size** | not derivable, 1,500 GiB free enforced at startup | `MIN_FREE_GIB` |
| **Instance** | p6-b200.48xlarge x2, 16x B200 at 180 GB | 2,880 GB across 2 nodes, 2,682.2 GiB |
| **Tensor parallel size** | 16 | `TP_SIZE` |
| **Weight precision** | MXFP4 | declared by the checkpoint's `quantization_config` |
| **KV cache precision** | FP8 (`fp8`) | `--kv-cache-dtype fp8` |
| **Context length** | 1,048,576, not pinned by the recipe | `config.json`, `max_position_embeddings` |
| **Memory fraction** | 0.9 | `GPU_MEM_UTIL` |
| **Max sequences** | 32 | `MAX_NUM_SEQS` |

KV cache per layer per token is `(kv_lora_rank + qk_rope_head_dim) x bytes_per_element`, which is 0.56 KiB here. Multi-head latent attention caches one compressed latent per layer instead of a key and value per head, so KV cost scales with layer count rather than head count.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `p6-b200.48xlarge` node group with capacity for two EFA-capable nodes](../../../k8s/examples/eksctl-nodegroup-gpu-efa.yaml)
- [Install the EFA device plugin (`vpc.amazonaws.com/efa: 8`)](../../../k8s/examples/eksctl-nodegroup-gpu-efa.yaml)

  ```bash
  helm repo add eks https://aws.github.io/eks-charts
  helm repo update
  helm install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin -n kube-system
  ```

- [Install the LeaderWorkerSet controller](https://github.com/kubernetes-sigs/lws)

  ```bash
  kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/latest/download/manifests.yaml
  ```

- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- The locally quantized main checkpoint already staged in the Hugging Face cache at `/models/.hfcache` on `fsx-pvc`, where the manifest's `HF_HOME` expects it when resolving `moonshotai/Kimi-K3`
- The DSpark draft already staged at `/models/kimi-k3-dspark` on `fsx-pvc`. This repository does not ship the source-specific staging Job for either checkpoint; stage both before deployment.

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-kimi-k3-mxfp4-p6-b200-lws-dspark
   export INSTANCE_TYPE=p6-b200.48xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm each remaining prerequisite exists.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" get nodes -l node.kubernetes.io/instance-type=p6-b200.48xlarge
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   kubectl --context "$EKS_CLUSTER_NAME" get crd leaderworkersets.leaderworkerset.x-k8s.io
   ```

4. Bring up the server. It mounts the checkpoint from the `fsx-pvc` claim; if yours has another name, update `claimName` in your copy of the recipe first.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f vllm-recipes/kimi-k3/mxfp4-p6-b200.48xlarge-lws-dspark.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" rollout status statefulset/kimi-k3-b200-latency-lws --timeout=3600s
   ```

5. Check whether the recipe's server is up and ready to serve.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec statefulset/kimi-k3-b200-latency-lws -c vllm-leader -- \
     curl -s localhost:8000/v1/models
   ```

6. Read the KV cache the server allocated. Record the reported maximum concurrency with any result.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs deploy/kimi-k3-b200-latency-lws -c vllm \
     | grep -E "GPU KV cache size|Maximum concurrency|model weights took"
   ```

7. Prove this recipe passes the flags it claims.

   ```bash
   eks-bench validate-manifests kimi-k3/mxfp4-p6-b200.48xlarge-lws-dspark.yaml
   ```

8. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b200-latency-lws \
       --mode fingerprint \
       --model-tag kimi-k3 --config lws-dspark \
       --model-id kimi-k3 \
       --thinking off \
       --out-dir docs/results
   ```

9. Benchmark in-cluster. If needed, repeat per protocol, changing --protocol and --profile.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b200-latency-lws \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b200.48xlarge --model-tag kimi-k3 \
       --config lws-dspark --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/kimi-k3/mxfp4-p6-b200.48xlarge-lws-dspark.yaml
```

---


## Settings

- **`if [ ! -d /models/kimi-k3-dspark ]; then`** Fail early and loudly if the draft model is missing - otherwise vLLM spends ~an hour loading 1560 GB before erroring on the speculator.

**`ports:`**

LATENCY LEVERS, and what each one is for:

```
--enable-prefix-caching
```

Off by default for K3 while the hybrid KDA/full-attention cache design settles. K3's strength is long-horizon agentic coding, where prefix reuse is heavy, so leaving it off UNDERSTATES realistic agentic TTFT. Caches over both paged KV and recurrent KDA state.

```
--speculative-config (DSpark)
```

Block-diffusion speculator, MLA-native so draft and target share KV layout. Biggest single latency lever available (~3x single-stream). Sweep NUM_SPEC_TOKENS: more drafts raise potential acceptance but cost more verification work per step. 7 is the published setting.

```
--max-num-seqs 32 (vs 512 baseline)
```

THE latency/throughput dial. Capping in-flight sequences keeps queueing delay out of TTFT and inter-token gaps, at the cost of aggregate tok/s. This is the batch-vs-real-time tradeoff made explicit - sweep it to produce the perf-at-SLO curve rather than trusting one point.

Optional, try as its own variant (adds variance, so isolate it):

```
--enable-eplb   Expert-Parallel Load Balancing. Evens compute across
```

EP ranks; can help when router load is skewed.

- **`- name: TP_SIZE`** Left untuned, so the topology this recipe changes is the only variable in a comparison.

- **`- name: MAX_NUM_SEQS`** ---- the latency levers ----

- **`- name: VLLM_ENGINE_READY_TIMEOUT_S`** multi-turn conversation. Larger values add periodic checkpoints (e.g. every 32K tokens) at more cache cost - worth sweeping for long agentic sessions. ---- engine env (same as baseline) ----

- **`- name: FI_PROVIDER`** ---- EFA / RDMA ----

- **`- name: INSTANCE_TYPE`** ---- provenance: config name must make the levers obvious in the record ----

- **`vllm serve moonshotai/Kimi-K3 \`** Worker must mirror the leader's parallelism AND speculative config.
- **`MAX_NUM_SEQS=32`** sweep 8 / 16 / 32 / 64 vs baseline 512
- **`NUM_SPEC_TOKENS=7`** published setting; sweep 3 / 5 / 7
- **`VLLM_USE_V2_MODEL_RUNNER=1`** v2 fully supports K3
- **`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`** 0 = retain prompt-end KDA states only, which suits
- **`failureThreshold: 280`** 70 min: main weights + draft model load
