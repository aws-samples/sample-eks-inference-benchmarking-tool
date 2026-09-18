# Gemma 4 E4B FP8 with an FP8 KV cache on g7e.2xlarge

Serves the same locally quantized ModelOpt FP8 build of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) as [`fp8-g7e.2xlarge-runbook.md`](./fp8-g7e.2xlarge-runbook.md), with the KV cache at FP8 instead of 16-bit. Two recipes share this runbook, differing only in `--max-num-batched-tokens` and `--stream-interval`: [`fp8-g7e.2xlarge-fp8kv-throughput.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-throughput.yaml) and [`fp8-g7e.2xlarge-fp8kv-latency.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-latency.yaml).

Every flag is held at the 16-bit-KV pair's value apart from `--kv-cache-dtype`, so running the two pairs back to back attributes the difference to KV precision alone.

## Capacity sizing

FP8 halves the bytes each cached token occupies, so at the same `--gpu-memory-utilization` the KV pool roughly doubles. Read `KV capacity from` and `KV full-ctx seqs` in the result file's provenance block for the allocation actually in force, and expect close to twice the figure the 16-bit pair reports.

Doubling the pool only helps when the pool is the constraint. The `chat_terse` profile sends 96 wire input tokens with a 30-token output cap, so a sequence occupies about 126 tokens and the working set at concurrency 512 is a small fraction of even the 16-bit pool on this 96 GiB card. On this workload the reason to expect a change is decode bandwidth, not capacity: attention re-reads the cache every step and FP8 halves those bytes.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 4B active, 8B on disk | Model card |
| **Layers** | 42, of which 7 full attention and 35 sliding at window 512 | `config.json`, `text_config.layer_types` |
| **KV-owning layers** | 24; the last 18 share KV with earlier layers | `config.json`, `text_config.num_kv_shared_layers` |
| **KV heads, head dimension** | 2; 256 sliding, 512 full attention | `config.json`, `head_dim` and `global_head_dim` |
| **Checkpoint size, FP8** | 11.09 GiB, one safetensors file | quantize Job output |
| **Instance** | g7e.2xlarge, 1x RTX PRO 6000 Blackwell at 96 GB | 8 vCPU, 64 GiB host RAM, 1,900 GB NVMe |
| **Tensor parallel size** | 1 | unset, single GPU |
| **Weight precision** | FP8 E4M3, static input scales | `hf_quant_config.json`, `quant_algo: FP8` |
| **KV cache precision** | FP8 E4M3, uncalibrated scales | `KV_CACHE_DTYPE=fp8`; see Settings |
| **Context length** | 8,192 | `MAX_MODEL_LEN`, model supports 131,072 |
| **Memory fraction** | 0.80 | `GPU_MEM_UTIL` |
| **Max sequences** | 1,024 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which at FP8 is 1 KiB on a sliding layer and 2 KiB on a full attention layer, half the 16-bit figures. Only the 4 cache-owning full attention layers grow with context; the 20 sliding layers hold at most 512 tokens whatever the context length.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.2xlarge` node group](../../../k8s/examples/eksctl-nodegroup-g7e-2xl-spot.yaml) (give it every subnet in your FSx filesystem's availability zone; a single-subnet group failed to fulfil on spot for several minutes)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- A node with more than 8 allocatable CPU for the benchmark client, which is not the GPU node (the Job requests 8 dedicated CPUs and an 8 vCPU node offers 7910m, so it stays Pending forever)

No `nvidia-open-driver-healer` DaemonSet and no RAID0 launch template, unlike the g7.2xlarge node group. See Settings for why neither applies here.

## Quickstart

Steps 1 to 5 are shared. Then run the throughput test, the latency test, or both. There is one GPU, so tear down one config before deploying the other.

The checkpoint outlives the Job that produced it, so step 4 is skipped when an earlier run on any instance type already wrote it to the claim. Step 5 runs either way, since it needs no GPU and catches an invalid export that the producing Job reported as successful.

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-e4b-fp8kv-g7e-2xlarge
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

4. Quantize the checkpoint to `/models/quantized/gemma-4-E4B-it-fp8`. Roughly five minutes, and it needs the GPU.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap quantizer-src \
       --from-file=app/quantize/src/quantize.py \
       --from-file=app/quantize/src/config.py \
       --from-file=app/quantize/src/fix_amax.py \
       --from-file=app/quantize/src/verify_checkpoint.py
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-jobs/gemma4-e4b/fp8-model-quantize.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/quantize-gemma4-e4b-fp8 --timeout=30m
   ```

5. Confirm the checkpoint is really FP8. Expect ~11 GiB; a failed Job names each problem.
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
   ./vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-throughput-deploy.sh
   ```

2. Confirm the KV cache is really FP8 before spending GPU time on a sweep. Expect `cache_dtype="fp8"` and roughly twice the 16-bit pair's `kv_cache_size_tokens`.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec \
       deployment/gemma4-e4b-vllm-g7e-2xl-fp8kv-throughput -c vllm -- \
       sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
   ```

3. Generate the accuracy gate (.JSON) required for the next step. The config tag differs from the 16-bit pair's, so this is its own reference file.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8kv-throughput \
       --mode fingerprint \
       --model-tag gemma4-e4b --config fp8kv-throughput \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

4. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8kv-throughput \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8kv-throughput --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8kv-throughput -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

5. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8kv-throughput \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8kv-throughput --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8kv-throughput -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results
   ```

## Latency test

1. Tear down the throughput server first. There is one GPU.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-throughput.yaml
   ```

2. Deploy and validate the latency server.
   ```bash
   ./vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-latency-deploy.sh
   ```

3. Confirm the KV cache is really FP8, as in the throughput test.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec \
       deployment/gemma4-e4b-vllm-g7e-2xl-fp8kv-latency -c vllm -- \
       sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
   ```

4. Generate the accuracy gate for this config.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8kv-latency \
       --mode fingerprint \
       --model-tag gemma4-e4b --config fp8kv-latency \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

5. Dry-run the in-cluster benchmark.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8kv-latency \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8kv-latency --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8kv-latency -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

6. Benchmark in-cluster.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service gemma4-e4b-vllm-g7e-2xl-fp8kv-latency \
       --protocol corpus --profile chat_terse \
       --instance g7e.2xlarge --model-tag gemma4-e4b \
       --config fp8kv-latency --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=gemma4-e4b-vllm-g7e-2xl-fp8kv-latency -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-throughput.yaml
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-latency.yaml
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name g7e-2xl-spot \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```

Delete the benchmark-client node group too if you created one for this run. The checkpoint on `fsx-pvc` survives, so a later run needs no requantization.

---

## Settings

- **`--kv-cache-dtype fp8`** The only difference from the 16-bit pair. It halves bytes per cached token, which both doubles the pool and halves what attention reads each decode step. `"bfloat16"` is rejected by the engine and `"auto"` resolves through `DTYPE` to BF16, which is what the sibling pair serves.

- **KV scales are uncalibrated, and this is a caveat on accuracy.** vLLM 0.29.0 rejects `--calculate-kv-scales` with `unrecognized arguments` and refuses to start, so no scaling factors are supplied. The server logs `Using KV cache scaling factor 1.0 for fp8_e4m3` and `Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention` at startup. Read those lines before trusting a quality comparison against the 16-bit pair. Calibrated scales would have to come from the checkpoint, produced by a calibration pass rather than a serving flag.

- **No attention backend is pinned,** so the engine selects the sm_120 default and no result file records which one served. That matters more here than on the 16-bit pair, because this build runs the attention arithmetic itself in FP8 rather than converting back first, putting the backend on the critical path for the thing being measured. Pin it explicitly before comparing against another environment's numbers.

- **`--no-enable-prefix-caching`** Held off, as in every recipe here. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure as cache-free.

- **`--gpu-memory-utilization 0.80`** Held at the 16-bit pair's value rather than raised to use the doubled pool, so instance and memory fraction do not move together between the two runs.

- **g7e.2xlarge spot capacity is thin.** A single-subnet node group spent several minutes returning `InsufficientInstanceCapacity` before fulfilling, and the error text named a different availability zone as having capacity. Give the node group every subnet in the filesystem's zone, and check `aws ec2 get-spot-placement-scores` before planning around a deadline.

- **The benchmark client needs its own node.** `measure` runs the client as a Job requesting 8 dedicated CPUs. An 8 vCPU instance advertises 7910m allocatable, so the Job cannot schedule on the GPU node or on an `m5.2xlarge` and sits Pending without an obvious cause. Keep a larger CPU node group at zero and scale it for the run.

- **No `nvidia-open-driver-healer` DaemonSet.** The RTX PRO 6000 Blackwell in this instance registers with the AL2023 NVIDIA AMI's shipped driver, unlike the g7.2xlarge, so nothing needs healing.

- **No RAID0 launch template.** That template exists so a weight-download `emptyDir` lands on the NVMe instance store. These recipes read the checkpoint from the claim and download nothing, so the 1,900 GB instance store goes unused.
