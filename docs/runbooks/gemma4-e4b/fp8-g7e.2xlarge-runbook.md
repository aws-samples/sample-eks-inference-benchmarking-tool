# Gemma 4 E4B FP8 on g7e.2xlarge

Serves a locally quantized ModelOpt FP8 build of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) on g7e.2xlarge (1x RTX PRO 6000 Blackwell Server Edition, sm_120) with 1 GPU per pod. Two recipes share this runbook, differing only in `--max-num-batched-tokens` and `--stream-interval`: [`fp8-g7e.2xlarge-throughput.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-throughput.yaml) and [`fp8-g7e.2xlarge-latency.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-latency.yaml). The same pair on the 32 GiB g7.2xlarge is in [`fp8-g7.2xlarge-runbook.md`](./fp8-g7.2xlarge-runbook.md); every flag is held at the g7.2xlarge value so instance is the only variable between them.

## Capacity sizing

The FP8 checkpoint occupies 11.09 GiB against BF16's 14.89 GiB, a 25% reduction. Only text-side Linear layers convert: embeddings, `lm_head`, and the vision and audio towers stay BF16, which is why the saving is a quarter rather than a half. The KV cache itself is **not** quantized, so per-token cache cost is unchanged from BF16. Read `KV capacity from` and `KV full-ctx seqs` in the result file's provenance block for the allocation in force.

This card has 96 GiB against the g7.2xlarge's 32 GiB, so at the same `--gpu-memory-utilization` the KV pool is roughly three times larger. The `chat_terse` profile used about a third of the g7.2xlarge pool at concurrency 512, so the extra capacity is not expected to change throughput on this workload. It is retained rather than tuned away so the two instances stay comparable.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 4B active, 8B on disk | Model card |
| **Layers** | 42, of which 7 full attention and 35 sliding at window 512 | `config.json`, `text_config.layer_types` |
| **KV-owning layers** | 24; the last 18 share KV with earlier layers | `config.json`, `text_config.num_kv_shared_layers` |
| **KV heads, head dimension** | 2; 256 sliding, 512 full attention | `config.json`, `head_dim` and `global_head_dim` |
| **Checkpoint size, FP8** | 11.09 GiB, one safetensors file | quantize Job output |
| **Quantized modules** | text-side Linear only | `hf_quant_config.json`, `exclude_modules` |
| **Instance** | g7e.2xlarge, 1x RTX PRO 6000 Blackwell at 96 GB | 8 vCPU, 64 GiB host RAM, 1,900 GB NVMe |
| **Tensor parallel size** | 1 | unset, single GPU |
| **Weight precision** | FP8 E4M3, static input scales | `hf_quant_config.json`, `quant_algo: FP8` |
| **KV cache precision** | BF16, 16-bit | `kv_cache_quant_algo: null`; `KV_CACHE_DTYPE=auto` |
| **Context length** | 8,192 | `MAX_MODEL_LEN`, model supports 131,072 |
| **Memory fraction** | 0.80 | `GPU_MEM_UTIL` |
| **Max sequences** | 1,024 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, 2 KiB on a sliding layer and 4 KiB on a full attention layer. FP8 weights do not shrink the cache; they shrink what the weights occupy, leaving more room for it. The reason to expect a gain from FP8 is decode bandwidth rather than concurrency: decode re-reads the weights every step, and FP8 cuts that by about a quarter.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.2xlarge` node group](../../../k8s/examples/eksctl-nodegroup-g7e-2xl-spot.yaml) (single subnet, in the same availability zone as your FSx filesystem)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)

No `nvidia-open-driver-healer` DaemonSet, unlike the g7.2xlarge node group. That healer builds the NVIDIA open kernel module on nodes whose AMI shipped the closed one, which g7e does not require. Confirm the node advertises a GPU before deploying:

```bash
kubectl --context "$EKS_CLUSTER_NAME" get nodes -l node.kubernetes.io/instance-type=g7e.2xlarge \
    -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
```

No RAID0 launch template either. That template puts a weight-download `emptyDir` on the NVMe instance store, and these recipes read the checkpoint from the claim and download nothing.

`g7e.2xlarge` is offered in a subset of a region's availability zones. Check offerings before choosing a subnet, since FSx for Lustre is single-AZ and fixes the zone:

```bash
aws ec2 describe-instance-type-offerings --region "$AWS_REGION" \
    --location-type availability-zone \
    --filters "Name=instance-type,Values=g7e.2xlarge" \
    --query 'InstanceTypeOfferings[].Location' --output text
```

## Quickstart

Steps 1 to 5 are shared. Then run the throughput test, the latency test, or both. There is one GPU, so tear down one config before deploying the other.

The checkpoint lives on `fsx-pvc` and outlives the Job that produced it, so steps 3 and 4 are skipped whenever an earlier run on any instance type already wrote it. Step 5 always runs.

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-e4b-fp8-g7e-2xlarge
   export INSTANCE_TYPE=g7e.2xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret. Then confirm the claim is bound and check whether the checkpoint is already present.

   ```bash
   ./scripts/install-prerequisites.sh
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=jsonpath='{.status.phase}'=Bound pvc/fsx-pvc --timeout=2m
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" run ckpt-check --rm -it --restart=Never \
       --image=public.ecr.aws/docker/library/busybox:1.36 \
       --overrides='{"spec":{"containers":[{"name":"c","image":"public.ecr.aws/docker/library/busybox:1.36","command":["sh","-c","ls -la /models/quantized/gemma-4-E4B-it-fp8 2>/dev/null || echo ABSENT"],"volumeMounts":[{"name":"m","mountPath":"/models","readOnly":true}]}],"volumes":[{"name":"m","persistentVolumeClaim":{"claimName":"fsx-pvc"}}]}}'
   ```

   A listing showing `model.safetensors` and `hf_quant_config.json` means skip to step 5. `ABSENT` means run steps 3 and 4 first.

3. **First run only.** Overlay this repository's quantizer source onto the image, since an image built earlier takes an export path that produces an invalid checkpoint.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap quantizer-src \
       --from-file=app/quantize/src/quantize.py \
       --from-file=app/quantize/src/config.py \
       --from-file=app/quantize/src/fix_amax.py \
       --from-file=app/quantize/src/verify_checkpoint.py
   ```

4. **First run only.** Quantize the checkpoint to `/models/quantized/gemma-4-E4B-it-fp8`. Roughly five minutes, and it needs the GPU, so tear down any server first.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-jobs/gemma4-e4b/fp8-model-quantize.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/quantize-gemma4-e4b-fp8 --timeout=30m
   ```

5. Confirm the checkpoint is really FP8. Run this on every new instance type, including when reusing a checkpoint, because it needs no GPU and catches an invalid export that the producing Job reported as successful. Expect ~11 GiB; a failed Job names each problem and means do not serve it.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap checkpoint-verifier \
       --from-file=verify_checkpoint.py=app/quantize/src/verify_checkpoint.py
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-jobs/gemma4-e4b/fp8-model-verify.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/verify-checkpoint --timeout=5m
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs job/verify-checkpoint
   ```

## Throughput test

1. Deploy and validate the model server. It mounts the checkpoint from `fsx-pvc`, so there is no download and startup is the model load only.

   ```bash
   ./vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-throughput-deploy.sh
   ```

2. Generate the accuracy gate (.JSON) required for the next step. The gate path is `docs/accuracy/<model-tag>-<config>.json` and carries no instance, so this is the same file the g7.2xlarge run uses.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8-throughput \
       --mode fingerprint \
       --model-tag gemma4-e4b --config fp8-throughput \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

3. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8-throughput \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8-throughput --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8-throughput -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

4. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8-throughput \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8-throughput --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8-throughput -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results
   ```

## Latency test

1. Deploy and validate the model server.

   ```bash
   ./vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-latency-deploy.sh
   ```

2. Generate the accuracy gate (.JSON) required for the next step. It is a separate file from the throughput gate, since the path is `docs/accuracy/<model-tag>-<config>.json`.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8-latency \
       --mode fingerprint \
       --model-tag gemma4-e4b --config fp8-latency \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

3. Dry-run the in-cluster benchmark.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8-latency \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8-latency --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8-latency -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

4. Benchmark in-cluster.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8-latency \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8-latency --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8-latency -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-throughput.yaml
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-latency.yaml
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name g7e-2xl-spot \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```

The checkpoint on `fsx-pvc` survives, so a later run needs no requantization.

---

## Settings

- **`--quantization modelopt`** Real ModelOpt FP8: `float8_e4m3fn` weights plus static `input_scale` tensors and an `hf_quant_config.json`. Served through vLLM's modelopt loader, not the vanilla `fp8` loader, which mis-reads this layout. The startup log confirms it with `Detected ModelOpt fp8 checkpoint (quant_algo=FP8)`.

- **`--dtype bfloat16`** The compute dtype. FP8 applies to stored weights; activations and the KV cache remain BF16. Gemma's activations exceed float16's ceiling, so this is the only safe 16-bit choice regardless of weight precision.

- **`--kv-cache-dtype auto`** Resolves to BF16 through `--dtype`, and `hf_quant_config.json` records `kv_cache_quant_algo: null`, so the cache is not quantized. Pinning the literal `bfloat16` passes argument parsing and then aborts the engine during KV initialisation with `unsupported kv_cache_dtype (str), got bfloat16`, because that value is only valid for a checkpoint whose weights default to FP8.

- **`--gpu-memory-utilization 0.80`** Held at the g7.2xlarge value, where 0.85 and 0.90 both aborted during CUDA graph capture on a 32 GiB card. This card has 96 GiB and would tolerate more, but raising it here would make instance and memory fraction move together and the comparison would no longer isolate the instance.

- **`--max-num-seqs 1024`** Above the canonical sweep's top level of 512, so a swept row measures the server rather than its own admission limit.

- **`--max-num-batched-tokens`** `16384` for throughput, `2048` for latency. Measured on g7.2xlarge this knob made no material difference at a 96-token prompt, within 3% on `req_per_sec`; it is held at the same values here. Do not carry that null result to a long-prompt profile.

- **`--stream-interval`** `10` for throughput, `1` for latency. Compare ITL only within one config.

- **`--no-enable-prefix-caching`** Off in both. The pinned profiles cycle a small prompt pool, and a warm cache skips prefill and inflates throughput. Check `Prefix cache hit` reads 0.0% in the provenance block.

- **`model-storage: persistentVolumeClaim: fsx-pvc`** A quantized checkpoint has to outlive the Job that produced it, so it lives on the claim and both recipes mount it read-write at `/models`. There is no init container and no download. Because FSx for Lustre is single-AZ, the node group is pinned to one subnet.

- **`nvidia.com/gpu: 1`** g7e.2xlarge = 1x Blackwell RTX PRO 6000 Server Edition, 96 GB (sm_120).

- **No driver healer on this instance type.** The g7.2xlarge runbook applies `nvidia-open-driver-healer` because its AMI shipped the closed kernel module and the node otherwise advertises `nvidia.com/gpu: 0` indefinitely. g7e does not need it. The `gemma4-31b` g7e recipes also run without it.

- **`MODEL_LOADER=image_text_to_text`** in the quantize Job. The default `causal_lm` auto-class cannot load this checkpoint.

- **The audio exclude patterns are specific to E4B**, which carries an audio tower as well as a vision one. Calibration prompts are text-only, so neither tower would see calibration data.

- **An invalid FP8 checkpoint is silent, which is why step 6 exists.** Weights can stay BF16 behind a `quantization_config` announcing FP8; vLLM then reads those bytes as FP8 and serves nonsense rather than refusing to start. The Job exits 0. `quantize.py` fails on its own bad export, but a stale image runs its own copy, so step 6 checks from outside.

- **The `quantizer-src` overlay in step 4 is a workaround, not the design.** `scripts/push-image.sh` builds the image from `app/quantize/src/`, so a freshly built image agrees with this repository by construction; drop the `src` volume and its mount once it does.

- **Two further recipes add GPU n-gram speculative decoding**, `fp8-g7e.2xlarge-ngram-throughput.yaml` and `fp8-g7e.2xlarge-ngram-latency.yaml`. They are identical to the pair above apart from `--speculative-config`, so running them back to back against it attributes the difference to speculation alone. On g7.2xlarge speculation reduced throughput at every concurrency level.

- **Quote `req_per_sec`, not `tokens_per_sec`, on this profile.** `tokens_per_sec` counts output tokens only, and `chat_terse` puts 96 of its 126 tokens on the input side.
