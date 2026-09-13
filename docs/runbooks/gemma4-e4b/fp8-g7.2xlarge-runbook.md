# Gemma 4 E4B FP8 on g7.2xlarge

Serves a locally quantized ModelOpt FP8 build of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) on g7.2xlarge (1x RTX PRO 4500 Blackwell, sm_120) with 1 GPU per pod. Two recipes share this runbook, differing only in `--max-num-batched-tokens` and `--stream-interval`: [`fp8-g7.2xlarge-throughput.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7.2xlarge-throughput.yaml) and [`fp8-g7.2xlarge-latency.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7.2xlarge-latency.yaml). The BF16 pair for the same instance is in [`bf16-g7.2xlarge-runbook.md`](./bf16-g7.2xlarge-runbook.md), and the two sets are identical apart from precision so a comparison isolates it.

## Capacity sizing

The FP8 checkpoint occupies 11.09 GiB against BF16's 14.89 GiB, a 25% reduction, and the 3.8 GiB it frees goes to KV cache. Only text-side Linear layers convert: embeddings, `lm_head`, and the vision and audio towers stay BF16, which is why the saving is a quarter rather than a half. The KV cache itself is **not** quantized, so per-token cache cost is unchanged from BF16 and the larger pool is purely the freed weight memory. Read `KV capacity from` and `KV full-ctx seqs` in the result file's provenance block for the allocation in force.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 4B active, 8B on disk | Model card |
| **Layers** | 42, of which 7 full attention and 35 sliding at window 512 | `config.json`, `text_config.layer_types` |
| **KV-owning layers** | 24; the last 18 share KV with earlier layers | `config.json`, `text_config.num_kv_shared_layers` |
| **KV heads, head dimension** | 2; 256 sliding, 512 full attention | `config.json`, `head_dim` and `global_head_dim` |
| **Checkpoint size, FP8** | 11.09 GiB, one safetensors file | quantize Job output |
| **Quantized modules** | text-side Linear only | `hf_quant_config.json`, `exclude_modules` |
| **Instance** | g7.2xlarge, 1x RTX PRO 4500 Blackwell at 32 GB | 32 GB nominal, 31.9 GiB |
| **Tensor parallel size** | 1 | unset, single GPU |
| **Weight precision** | FP8 E4M3, static input scales | `hf_quant_config.json`, `quant_algo: FP8` |
| **KV cache precision** | BF16, 16-bit | `kv_cache_quant_algo: null`; `KV_CACHE_DTYPE=auto` |
| **Context length** | 8,192 | `MAX_MODEL_LEN`, model supports 131,072 |
| **Memory fraction** | 0.80 | `GPU_MEM_UTIL` |
| **Max sequences** | 1,024 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, unchanged from the BF16 recipe at 2 KiB on a sliding layer and 4 KiB on a full attention layer. FP8 weights do not shrink the cache; they shrink what the weights occupy, leaving more room for it. The reason to expect a gain here is decode bandwidth rather than concurrency: decode re-reads the weights every step, and FP8 cuts that by about a quarter.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7.2xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml) (in a subnet with free IPs; this instance type's AZ may have none by default)
- [Apply the `nvidia-open-driver-healer` DaemonSet](../../../k8s/examples/daemonset-nvidia-open-driver.yaml) (without it the node advertises no GPU indefinitely)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)

No RAID0 launch template, unlike the BF16 node group. That template puts a weight-download `emptyDir` on the NVMe instance store, and these recipes read the checkpoint from the claim and download nothing.

## Quickstart

Steps 1 to 6 are shared. Then run the throughput test, the latency test, or both. There is one GPU, so tear down one config before deploying the other.

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-e4b-fp8-g7-2xlarge
   export INSTANCE_TYPE=g7.2xlarge
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

4. Overlay this repository's quantizer source onto the image, since an image built earlier takes an export path that produces an invalid checkpoint.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap quantizer-src \
       --from-file=app/quantize/src/quantize.py \
       --from-file=app/quantize/src/config.py \
       --from-file=app/quantize/src/fix_amax.py \
       --from-file=app/quantize/src/verify_checkpoint.py
   ```

5. Quantize the checkpoint to `/models/quantized/gemma-4-E4B-it-fp8`. Roughly five minutes, and it needs the GPU, so tear down any server first.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-quantize/gemma4-e4b-fp8.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/quantize-gemma4-e4b-fp8 --timeout=30m
   ```

6. Confirm the checkpoint is really FP8. Expect ~11 GiB; a failed Job names each problem and means do not serve it.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap checkpoint-verifier \
       --from-file=verify_checkpoint.py=app/quantize/src/verify_checkpoint.py
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-quantize/verify-checkpoint.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/verify-checkpoint --timeout=5m
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs job/verify-checkpoint
   ```

## Throughput test

1. Deploy and validate the model server. It mounts the checkpoint from `fsx-pvc`, so there is no download and startup is the model load only.

   ```bash
   ./vllm-recipes/gemma4-e4b/fp8-g7.2xlarge-throughput-deploy.sh
   ```

2. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7-2xl-fp8-throughput \
       --mode fingerprint \
       --model-tag gemma4-e4b --config fp8-throughput \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

3. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7-2xl-fp8-throughput \
       --protocol corpus --profile chat_terse \
       --instance g7.2xlarge --model-tag gemma4-e4b \
       --config fp8-throughput --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7-2xl-fp8-throughput -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

4. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7-2xl-fp8-throughput \
       --protocol corpus --profile chat_terse \
       --instance g7.2xlarge --model-tag gemma4-e4b \
       --config fp8-throughput --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7-2xl-fp8-throughput -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results
   ```

## Latency test

1. Deploy and validate the model server.

   ```bash
   ./vllm-recipes/gemma4-e4b/fp8-g7.2xlarge-latency-deploy.sh
   ```

2. Generate the accuracy gate (.JSON) required for the next step. It is a separate file from the throughput gate, since the path is `docs/accuracy/<model-tag>-<config>.json`.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7-2xl-fp8-latency \
       --mode fingerprint \
       --model-tag gemma4-e4b --config fp8-latency \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

3. Dry-run the in-cluster benchmark.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7-2xl-fp8-latency \
       --protocol corpus --profile chat_terse \
       --instance g7.2xlarge --model-tag gemma4-e4b \
       --config fp8-latency --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7-2xl-fp8-latency -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

4. Benchmark in-cluster.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7-2xl-fp8-latency \
       --protocol corpus --profile chat_terse \
       --instance g7.2xlarge --model-tag gemma4-e4b \
       --config fp8-latency --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7-2xl-fp8-latency -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-e4b/fp8-g7.2xlarge-throughput.yaml
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-e4b/fp8-g7.2xlarge-latency.yaml
```

The checkpoint on `fsx-pvc` survives, so a later run needs no requantization.

---

## Settings

- **`--quantization modelopt`** Real ModelOpt FP8: `float8_e4m3fn` weights plus static `input_scale` tensors and an `hf_quant_config.json`. Served through vLLM's modelopt loader, not the vanilla `fp8` loader, which mis-reads this layout. The startup log confirms it with `Detected ModelOpt fp8 checkpoint (quant_algo=FP8)`.

- **`--dtype bfloat16`** The compute dtype, unchanged from the BF16 recipes. FP8 applies to stored weights; activations and the KV cache remain BF16. Gemma's activations exceed float16's ceiling, so this is the only safe 16-bit choice regardless of weight precision.

- **`--kv-cache-dtype auto`** Resolves to BF16 through `--dtype`, and `hf_quant_config.json` records `kv_cache_quant_algo: null`, so the cache is not quantized. Keep it that way: the `gemma4-31b` FP8 runbooks record that vLLM 0.21 with Gemma 4 raises a `q_scale` assertion when the KV cache is quantized. Pinning the literal `bfloat16` is also wrong here, for the reason given in the BF16 runbook.

- **`--max-num-seqs 1024`** Above the canonical sweep's top level of 512, so a swept row measures the server rather than its own admission limit. FP8 frees enough weight memory to enlarge the KV pool by roughly 40%, which widens the margin further.

- **`--max-num-batched-tokens`** `16384` for throughput, `2048` for latency, matching the BF16 pair. Measured on the BF16 build this knob made no material difference at a 96-token prompt; it is held at the same values here so precision is the only variable between the two runbooks.

- **`--stream-interval`** `10` for throughput, `1` for latency, matching the BF16 pair. Compare ITL only within one config.

- **`--no-enable-prefix-caching`** Off in both, as in the BF16 recipes: the pinned profiles cycle a small prompt pool, and a warm cache skips prefill and inflates throughput. Check `Prefix cache hit` reads 0.0% in the provenance block.

- **`model-storage: persistentVolumeClaim: fsx-pvc`** The one structural difference from the BF16 recipes, which download into an instance-store `emptyDir`. A quantized checkpoint has to outlive the Job that produced it, so it lives on the claim and both FP8 recipes mount it read-write at `/models`. There is no init container and no download.

- **`nvidia.com/gpu: 1`** g7.2xlarge = 1x Blackwell RTX PRO 4500, 32 GB (sm_120).

- **`MODEL_LOADER=image_text_to_text`** in the quantize Job. The default `causal_lm` auto-class cannot load this checkpoint.
- **The audio exclude patterns are specific to E4B**, which carries an audio tower as well as a vision one. Calibration prompts are text-only, so neither tower would see calibration data.
- **An invalid FP8 checkpoint is silent, which is why step 6 exists.** Weights can stay BF16 behind a `quantization_config` announcing FP8; vLLM then reads those bytes as FP8 and serves nonsense rather than refusing to start. The Job exits 0, all four FP8 recipes inherit the path, and a gate captured against the same checkpoint agrees with itself. `quantize.py` now fails on its own bad export, but a stale image runs its own copy, so step 6 checks from outside.
- **The `quantizer-src` overlay in step 4 is a workaround, not the design.** `scripts/push-image.sh` builds the image from `app/quantize/src/`, so a freshly built image agrees with this repository by construction; drop the `src` volume and its mount once it does.

- **Quote `req_per_sec`, not `tokens_per_sec`, on this profile.** `tokens_per_sec` counts output tokens only, and `chat_terse` puts 96 of its 126 tokens on the input side. The BF16 runbook's note on reading that metric applies unchanged.
