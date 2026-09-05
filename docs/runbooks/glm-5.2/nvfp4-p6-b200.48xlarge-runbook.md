# GLM-5.2 NVFP4 on p6-b200.48xlarge

Serves [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) on p6-b200.48xlarge (8x B200, sm_100) with 8 GPUs per pod. Weights are read from the shared filesystem rather than downloaded into the pod, so a restart costs a mount instead of 433 GiB.

## Capacity sizing

GLM-5.2-NVFP4 at NVFP4 occupies 432.9 GiB of the 1,341.1 GiB on 8 B200 cards, 271 GiB less than the FP8 checkpoint. All of that saving goes to KV cache, because KV precision is the same in both recipes: measured 95.48 GiB of KV per rank here against 61.35 GiB on the FP8 sibling, and 65.59x concurrency against 42.14x.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 753B total | `zai-org/GLM-5.2-FP8` card; the NVFP4 card reports packed bytes, not parameters |
| **Layers** | 78 | `config.json`, `num_hidden_layers` |
| **KV latent per layer** | kv_lora_rank 512 plus qk_rope 64 | `config.json` |
| **Checkpoint size, NVFP4** | 432.9 GiB, 47 shards, 424 GiB on disk | Hub API, confirmed on download |
| **Instance** | p6-b200.48xlarge, 8x B200 at 183 GB | 1,466.9 GB nominal, 1,341.1 GiB |
| **Tensor parallel size** | 8 | `TP_SIZE` |
| **Weight precision** | NVFP4 | declared by the checkpoint's `quantization_config` |
| **KV cache precision** | FP8 (`fp8_e4m3`), auto-selected | server log `kv_cache_dtype=fp8_e4m3`, with no `--kv-cache-dtype` passed |
| **KV pool, measured** | 95.48 GiB per rank, 2,149,312 tokens | step 6 on this hardware |
| **Weights, measured** | 55.42 GiB per rank, 572.6 s to load | step 6 on this hardware |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

## Before you begin

- A `p6-b200.48xlarge` node group, or a capacity-block node group if on-demand B200 is unavailable
- An `fsx-pvc` claim whose filesystem is reachable from the node's subnet. FSx Lustre only accepts clients its security group admits: a node in a different subnet from the filesystem needs an ingress rule for TCP 988 from that subnet's CIDR, and without it the mount fails with `mount.lustre ... Invalid argument`, which reads like a client problem and is not.

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-glm-52-nvfp4-p6-b200-48xlarge
   export INSTANCE_TYPE=p6-b200.48xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the claim.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   ```

4. Stage the weights on the PVC. A completed earlier run makes this a no-op. Run it on a node in the filesystem's own AZ so the write path stays intra-AZ.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-downloads/glm-5.2-nvfp4.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait --for=condition=complete job/download-glm-52-nvfp4 --timeout=6h
   ```

5. Bring up the server.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f vllm-recipes/glm-5.2/nvfp4-p6-b200.48xlarge.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" rollout status deploy/glm-52-b200-nvfp4 --timeout=3600s
   ```

6. Read the KV cache the server allocated. Record the reported maximum concurrency with any result.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs deploy/glm-52-b200-nvfp4 -c vllm \
     | grep -E "GPU KV cache size|Maximum concurrency|Available KV cache memory|Model loading took"
   ```

7. Prove this recipe passes the flags it claims.

   ```bash
   eks-bench validate-manifests glm-5.2/nvfp4-p6-b200.48xlarge.yaml
   ```

8. Generate the accuracy gate (.JSON). `--thinking off` is required, not optional: see Settings.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service glm-52-b200-nvfp4 \
       --mode fingerprint \
       --model-tag glm-5.2 --config nvfp4 \
       --model-id glm-5.2-nvfp4 \
       --thinking off \
       --out-dir docs/results
   ```

9. Benchmark in-cluster with the reasoning protocol, which is the right one for this model: it budgets 1024 output tokens, counts reasoning tokens in the total, and warms the server before measuring.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service glm-52-b200-nvfp4 \
       --protocol reasoning --profile legacy_reasoning \
       --instance p6-b200.48xlarge --model-tag glm-5.2 \
       --config nvfp4 --quant nvfp4 \
       --model-id glm-5.2-nvfp4 \
       --tokenizer nvidia/GLM-5.2-NVFP4 \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/glm-5.2/nvfp4-p6-b200.48xlarge.yaml
```

---

## Settings

- **`persistentVolumeClaim: { claimName: fsx-pvc }`** Weights live on the shared filesystem, not in an `emptyDir`. This is the deliberate difference from the B300 recipe, whose own runbook records the cost of the emptyDir approach: the checkpoint is deleted with the pod and re-downloaded on every deploy, measured at 2h15m twice. A PVC survives pod deletion, so `kubectl delete` costs nothing but the mount. It also removes the free-space check the B300 initContainer needs, because the claim's capacity is known rather than inherited from wherever kubelet's root happens to live.

- **`initContainers: wait-for-model`** Refuse to start on a partial checkpoint. Counts shards against `model.safetensors.index.json` and blocks on any `.incomplete` file, because a truncated checkpoint fails deep inside the loader with an error that does not name the cause.

- **`--tensor-parallel-size 8`** All 8 GPUs on one node. Intra-node GPU-to-GPU runs over NVLink and needs no EFA: `nvidia-smi topo -m` on this instance shows NV18 between every pair, 18 bonded links each. EFA only matters once a second node is involved.

- **no `--kv-cache-dtype`, yet KV is FP8 anyway** The flag is omitted to match the B300 NVFP4 recipe, but omitting it does not mean 16-bit. This server reports `kv_cache_dtype=fp8_e4m3` with no flag passed, and its measured KV cost per token, 46.58 KiB, is identical to the FP8 sibling that sets the flag explicitly. vLLM is selecting FP8 KV from the checkpoint. The B300 runbook's capacity table describes this configuration as "16-bit default" and derives 87.75 KiB per token from that, which is twice the real figure. Anything sized on that assumption is wrong in the safe direction, but it is still wrong. If you want KV precision pinned rather than inferred, pass the flag and say so, because an auto-selected value can change with the vLLM version and nothing in the recipe would record it.

- **`--reasoning-parser glm45` and `--tool-call-parser glm47`** Parser names follow vLLM's own GLM-5.2 recipe. These are what make the reasoning trace and tool calls arrive as structured fields rather than raw text in `content`.

- **`--max-cudagraph-capture-size 512`** MUST be >= `--max-num-seqs`. A batch wider than the largest captured graph falls back to eager execution and loses most of the graph speedup. Setting it below `--max-num-seqs` is the most expensive single misconfiguration available here.

- **`--no-enable-prefix-caching`** An honest baseline. Every result file records the server's prefix cache hit rate, and any value above 0% on a non-cache-seeking protocol means prefill was skipped and throughput is inflated.

- **`VLLM_DEEP_GEMM_WARMUP=skip`** Faster startup, at the cost of moving first-kernel work into the first requests. Combined with a protocol that does no warmup this produces a large cold-start artefact, so it interacts with the note below.

- **`--thinking off` on the accuracy gate**  Required, and the gate fails without it. GLM-5.2 emits a reasoning trace, the fingerprint tier caps output at 256 tokens, and the trace consumes the entire budget, so responses arrive with empty content and `finish_reason=length` and the gate refuses to write a reference from what looks like a broken server. The flag reaches `accuracy.py` through `launch`'s unknown-argument passthrough, so it is absent from `launch --help` yet forwarded correctly. Off is right for the gate, which asks whether the server is configured as claimed rather than whether the model reasons well. Measure the reasoning behaviour with `--protocol reasoning`, as step 9 does.

- **`--protocol reasoning` rather than `corpus`**  Corpus caps output at 150 tokens, which truncates a reasoning model mid-trace, and it pins `WARMUP_REQUESTS=0`. Reasoning budgets 1024 tokens and warms with 8 requests, which matters here: on the sibling FP8 recipe a cold 3-rep corpus run understated throughput by 2.85x at concurrency 128 while every request succeeded and the gate passed. If you do run corpus on this recipe for continuity, treat the first sweep as a warmup and quote the second.

- **`startupProbe: failureThreshold: 120`** Allows 40 minutes. Weight load plus CUDA graph capture over 51 shapes in two passes is the long pole, and a probe firing mid-capture restarts the pod. The FP8 sibling on this hardware took 820.7 seconds to load and about 22 minutes from pod start to ready.
