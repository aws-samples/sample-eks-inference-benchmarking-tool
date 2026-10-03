# Gemma 4 E4B FP8 with an FP8 KV cache on p5.4xlarge, Triton attention

Serves a locally quantized ModelOpt FP8 build of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) on one H100 with an FP8 KV cache and the attention backend pinned to `TRITON_ATTN`, using [`fp8-p5.4xlarge-fp8kv-attn-triton.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton.yaml). Its g7e sibling, [`fp8-g7e.2xlarge-fp8kv-attn-triton.yaml`](../../../vllm-recipes/gemma4-e4b/fp8-g7e.2xlarge-fp8kv-attn-triton.yaml), holds every serving flag identical, so the pair isolates the GPU.

## Capacity sizing

The recipe runs at `--gpu-memory-utilization 0.90` on an 80 GB card. Read `KV capacity from` and `KV full-ctx seqs` in the result file's provenance block for the allocation in force.

The pool is not the constraint. `chat_terse` sends 96 wire input tokens with a 30-token output cap, so a sequence occupies about 126 tokens and the working set at concurrency 512 is a small fraction of the pool. The backend is the kernel doing the attention arithmetic, and with an FP8 KV cache that arithmetic runs in FP8, which puts the backend on the critical path.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 4B active, 8B on disk | Model card |
| **Layers** | 42, of which 7 full attention and 35 sliding at window 512 | `config.json`, `text_config.layer_types` |
| **KV heads, head dimension** | 2; 256 sliding, 512 full attention | `config.json`, `head_dim` and `global_head_dim` |
| **Checkpoint size, FP8** | 11.09 GiB, one safetensors file | `fetch-weights` checks 1 shard and at least 10 GiB |
| **Instance** | p5.4xlarge, 1x H100 at 80 GB, sm_90 | Instance type |
| **Weight precision** | FP8 E4M3, static input scales | `hf_quant_config.json`, `quant_algo: FP8` |
| **KV cache precision** | FP8 E4M3, uncalibrated scales | `KV_CACHE_DTYPE=fp8`; see Settings |
| **Attention backend** | `TRITON_ATTN` | `ATTENTION_BACKEND` |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Max sequences** | 1,024 | `MAX_NUM_SEQS` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Stream interval** | 10 tokens | `STREAM_INTERVAL` |

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `p5.4xlarge` Capacity Block node pool](../../../k8s/examples/karpenter-capacity-block-nodepool.yaml) (it mounts the instance store as RAID0)
- [Create a benchmark-client node group](../../../k8s/examples/eksctl-nodegroup-bench-client.yaml) (the client Job does not tolerate the GPU taint)
- [Install the EKS Pod Identity Agent add-on](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-agent-setup.html)
- [Create an IAM role for Pod Identity that reads the S3 checkpoint prefix](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-role.html)
- [Build and push the quantizer image](../../../k8s/examples/README.md#job-quantize-modelyaml) (only if the checkpoint is not in S3 yet)
- [Create a `g7.2xlarge` node group for quantization](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml) (only if the checkpoint is not in S3 yet)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (only if the checkpoint is not in S3 yet)
- [Grant S3 write access to the staging Job](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-association.html) (only if the checkpoint is not in S3 yet)

## Quickstart

Skip steps 4 to 7 when an earlier run already put the checkpoint in S3. Run step 8 either way.

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export AWS_ACCOUNT_ID=<your-account-id>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-gemma4-e4b-fp8kv-triton-p5-4xlarge
   export INSTANCE_TYPE=p5.4xlarge
   export WEIGHTS_BUCKET=<your-weights-bucket>
   export WEIGHTS_REGION=<your-weights-bucket-region>
   export GEMMA_S3_ROLE_ARN=<your-gemma-s3-read-role-arn>
   ```

2. Configure the EKS context and namespace, install the NVIDIA device plugin, and create the `hf-token` secret.
   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Create the `gemma4-31b-bench` ServiceAccount the recipe names and give it the S3 read role.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create serviceaccount gemma4-31b-bench \
       --dry-run=client -o yaml | kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f -
   aws eks create-pod-identity-association --cluster-name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" \
       --namespace "$NAMESPACE" --service-account gemma4-31b-bench --role-arn "$GEMMA_S3_ROLE_ARN"
   ```

4. Confirm the claim is bound.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=jsonpath='{.status.phase}'=Bound pvc/fsx-pvc --timeout=2m
   ```

5. Quantize the checkpoint onto the claim at `/models/quantized/gemma-4-E4B-it-fp8`, which takes about five minutes.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap quantizer-src \
       --from-file=app/quantize/src/quantize.py \
       --from-file=app/quantize/src/config.py \
       --from-file=app/quantize/src/fix_amax.py \
       --from-file=app/quantize/src/verify_checkpoint.py
   sed -e "s|012345678901.dkr.ecr.us-east-1|${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}|" \
       model-jobs/gemma4-e4b/fp8-model-quantize.yaml \
     | kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f -
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/quantize-gemma4-e4b-fp8 --timeout=30m
   ```

6. Confirm the checkpoint is really FP8; expect about 11 GiB, and a failed Job names each problem.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap checkpoint-verifier \
       --from-file=verify_checkpoint.py=app/quantize/src/verify_checkpoint.py
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-jobs/gemma4-e4b/fp8-model-verify.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/verify-checkpoint --timeout=5m
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs job/verify-checkpoint
   ```

7. Copy the checkpoint from the claim to S3; the Job fails on a shard-count mismatch.
   ```bash
   sed -e "s|value: SUBSTITUTE-weights-bucket|value: ${WEIGHTS_BUCKET}|" \
       -e "/name: AWS_REGION/{n;s|value: .*|value: ${WEIGHTS_REGION}|;}" \
       model-jobs/gemma4-e4b/fp8-model-fsx-to-s3.yaml \
     | kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f -
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/gemma4-e4b-fp8-fsx-to-s3 --timeout=1h
   ```

8. Confirm the checkpoint in S3 has one safetensors file and `hf_quant_config.json`.
   ```bash
   aws s3 ls "s3://${WEIGHTS_BUCKET}/gemma4-e4b-fp8-quantized/" --region "$WEIGHTS_REGION" \
       | grep -E 'hf_quant_config\.json$|\.safetensors$'
   ```

## Throughput test

1. Set the weights bucket and its region in the recipe; the deploy script refuses the placeholder.
   ```bash
   sed -i.bak -e "s|SUBSTITUTE-weights-bucket|${WEIGHTS_BUCKET}|" \
       -e "/name: AWS_REGION/{n;s|value: .*|value: \"${WEIGHTS_REGION}\"|;}" \
       vllm-recipes/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton.yaml \
     && rm vllm-recipes/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton.yaml.bak
   ```

2. Deploy the server; the script fails if the engine did not select `TRITON_ATTN`.
   ```bash
   ./vllm-recipes/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton-deploy.sh
   ```

3. Confirm `/models` is the NVMe RAID0 array, not the EBS root volume.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec \
       deployment/gemma4-e4b-vllm-p5-4xl-fp8kv-triton -c vllm -- df -h /models
   ```

4. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
       --mode fingerprint \
       --model-tag gemma4-e4b --config p5-fp8kv-attn-triton \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --out-dir docs/results
   ```

5. Dry-run the in-cluster benchmark, which renders the Job without creating it.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
       --protocol corpus --profile chat_terse \
       --instance p5.4xlarge --model-tag gemma4-e4b \
       --config p5-fp8kv-attn-triton --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

6. Benchmark in-cluster.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service gemma4-e4b-vllm-p5-4xl-fp8kv-triton \
       --protocol corpus --profile chat_terse \
       --instance p5.4xlarge --model-tag gemma4-e4b \
       --config p5-fp8kv-attn-triton --quant fp8 \
       --model-id /models/quantized/gemma-4-E4B-it-fp8 \
       --tokenizer google/gemma-4-E4B-it \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/gemma4-e4b/fp8-p5.4xlarge-fp8kv-attn-triton.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete nodepool bench-capacity-block --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete ec2nodeclass bench-capacity-block --ignore-not-found
```

Scale the quantization and benchmark-client node groups to zero too if you created them for this run. The checkpoint in S3 survives, so a later run skips steps 4 to 7.

---

## Settings

- **Every serving flag matches the g7e.2xlarge sibling.** Context, batch limits, memory fraction, stream interval, KV precision and backend are identical, so a difference between the two results is the GPU and the weight source, not the configuration.
- **The backend is pinned with `--attention-backend`, not `VLLM_ATTENTION_BACKEND`.** vLLM 0.29.0 does not read that environment variable. It logs `Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND` and proceeds on its auto-selected backend. The deploy script asserts `TRITON_ATTN` appears in the server log and exits non-zero if it does not.
- **The backend is pinned rather than auto-selected** because vLLM would pick a different kernel on sm_90 than on the g7e sibling's sm_120, which would make the backend a second variable in the pair.
- **`--config p5-fp8kv-attn-triton`** The `p5-` prefix keeps this server's accuracy gate separate from the g7e sibling's. The gate file is named by model tag and config, and the same checkpoint on a different GPU produces a different fingerprint.
- **KV scales are uncalibrated.** vLLM 0.29.0 rejects `--calculate-kv-scales`, so the server logs `Using KV cache scaling factor 1.0 for fp8_e4m3`. This bounds any quality comparison.
- **`--no-enable-prefix-caching`** Held off for an uncached measurement. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`--stream-interval 10`** The server streams in 10-token chunks. That changes inter-token latency only, so never compare ITL against a recipe with a different stream interval.
- **`MAX_CUDAGRAPH=512`.** Matches the top concurrency level of the sweep, so every measured batch size is graph-captured and nothing is captured above it.
- **Weights come from S3 into an `emptyDir`, not from the claim.** A pod restart re-downloads 11 GiB. The `fetch-weights` init container fails unless it finds `hf_quant_config.json`, exactly one shard and at least 10 GiB.
- **The instance store must back the `emptyDir`.** The Capacity Block node class sets `instanceStorePolicy: RAID0`; without it the sync lands on the EBS root volume. Throughput step 3 confirms where `/models` is.
- **`serviceAccountName: gemma4-31b-bench`** The recipe reuses the Gemma 4 31B ServiceAccount name for its pod-scoped S3 read through EKS Pod Identity. Any ServiceAccount with read access to the prefix works if you change the name in the recipe.
- **`WEIGHTS_BUCKET` and `AWS_REGION` are placeholders in git.** Throughput step 1 fills them in your working copy; keep that edit out of commits. The deploy script refuses to apply while the bucket placeholder remains.
- **The S3 staging Job sets no ServiceAccount.** It runs as the namespace's `default` ServiceAccount, so that is the identity the write role must be associated with.
- **Copy the checkpoint rather than requantizing it.** Calibration is not bit-reproducible, so a fresh quantize gives different weights and makes results incomparable with the g7e sibling. Serve the S3 copy of the checkpoint the sibling reads from the claim.
