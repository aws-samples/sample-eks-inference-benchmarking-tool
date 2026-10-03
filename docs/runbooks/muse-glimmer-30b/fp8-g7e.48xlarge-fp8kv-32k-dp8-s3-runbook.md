# Muse-Glimmer-30B FP8 with an FP8 KV cache on g7e.48xlarge, eight replicas, Triton attention

Serves a locally quantized ModelOpt FP8 build of [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) as eight independent single-GPU replicas behind one Service, using [`fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml`](../../../vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml). This is the `TRITON_ATTN` arm of an attention-backend pair. [`fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer.yaml`](../../../vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer.yaml) is identical apart from the backend.

## Capacity sizing

Muse-Glimmer-30B is dense and its KV cache is light. 2 KV heads at head size 128 across 52 layers cost 26 KiB per token at FP8, so a full 32,768-token sequence holds 0.81 GiB. The 32 GiB checkpoint fits one 96 GB card with room for roughly 60 full-length sequences, which is why each replica takes one GPU rather than splitting a copy across PCIe. Read `KV capacity from` and `KV full-ctx seqs` in the result file's provenance block for the allocation in force.

`coding_repo_context` sends about 14.5k wire input tokens and asks for up to 1,024 output tokens, so a request occupies about half a full-length sequence. Prompt processing, not KV memory, is the limit for this workload.

| Property | Value | Source |
|---|---|---|
| **Parameters** | ~29.6B dense, including a 1.8B image encoder | `config.json`; no expert or router settings |
| **Layers** | 52 | `config.json` |
| **KV heads, head dimension** | 2, 128 | `config.json` |
| **KV cache per token, FP8** | 26 KiB | `2 x 2 x 128 x 1 byte x 52 layers` |
| **Checkpoint size, FP8** | ~32 GiB, 4 safetensors shards | `fetch-weights` init container checks |
| **Instance** | g7e.48xlarge, 8x RTX PRO 6000 Blackwell at 96 GB, sm_120, PCIe, no NVLink | Instance type |
| **Serving layout** | 8 replicas x 1 GPU, one Service | `replicas: 8`, `nvidia.com/gpu: 1` |
| **Weight precision** | FP8 E4M3, ModelOpt export | `QUANTIZATION` |
| **KV cache precision** | FP8 E4M3, uncalibrated scales | `KV_CACHE_DTYPE`; see Settings |
| **Attention backend** | `TRITON_ATTN` | `ATTENTION_BACKEND` |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Max sequences** | 64 per replica | `MAX_NUM_SEQS` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.48xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml) (set the instance type)
- [Mount its instance store as RAID0 for kubelet storage](../../../k8s/examples/nodeconfig-local-nvme-raid0.yaml) (eight replicas stage 256 GiB)
- [Create a benchmark-client node group](../../../k8s/examples/eksctl-nodegroup-bench-client.yaml) (the client Job does not tolerate the GPU taint)
- [Install the EKS Pod Identity Agent add-on](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-agent-setup.html)
- [Create an IAM role for Pod Identity that reads the S3 checkpoint prefix](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-role.html)
- [Create a `g7e.2xlarge` node group for quantization](../../../k8s/examples/eksctl-nodegroup-g7e-2xl-spot.yaml) (only if the checkpoint is not in S3 yet)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (only if the checkpoint is not in S3 yet)
- [Grant S3 write access to the staging Job](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-association.html) (only if the checkpoint is not in S3 yet)

## Quickstart

Skip steps 4 to 6 when an earlier run already put the checkpoint in S3. Run step 7 either way.

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-muse-glimmer-30b-fp8kv-32k-dp8-g7e-48xlarge
   export INSTANCE_TYPE=g7e.48xlarge
   export WEIGHTS_BUCKET=<your-weights-bucket>
   export WEIGHTS_REGION=<your-weights-bucket-region>
   export MUSE_S3_ROLE_ARN=<your-muse-s3-read-role-arn>
   ```

2. Configure the EKS context and namespace, install the NVIDIA device plugin, and create the `hf-token` secret.
   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Create the `muse-glimmer-30b-bench` ServiceAccount and give it the S3 read role.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create serviceaccount muse-glimmer-30b-bench \
       --dry-run=client -o yaml | kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f -
   aws eks create-pod-identity-association --cluster-name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" \
       --namespace "$NAMESPACE" --service-account muse-glimmer-30b-bench --role-arn "$MUSE_S3_ROLE_ARN"
   ```

4. Confirm the claim is bound.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=jsonpath='{.status.phase}'=Bound pvc/fsx-pvc --timeout=2m
   ```

5. Quantize the checkpoint onto the claim at `/models/quantized/Muse-Glimmer-30B-fp8`, on one g7e.2xlarge GPU.
   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create configmap quantizer-src \
       --from-file=app/quantize/src/quantize.py \
       --from-file=app/quantize/src/config.py \
       --from-file=app/quantize/src/fix_amax.py \
       --from-file=app/quantize/src/verify_checkpoint.py
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-jobs/muse-glimmer-30b/fp8-model-quantize.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/quantize-muse-glimmer-30b-fp8 --timeout=3h
   ```

6. Copy the checkpoint from the claim to S3; the Job fails on a shard-count mismatch.
   ```bash
   sed -e "s|value: SUBSTITUTE-weights-bucket|value: ${WEIGHTS_BUCKET}|" \
       -e "/name: AWS_REGION/{n;s|value: .*|value: ${WEIGHTS_REGION}|;}" \
       model-jobs/muse-glimmer-30b/fp8-model-fsx-to-s3.yaml \
     | kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f -
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=condition=complete job/muse-glimmer-30b-fp8-fsx-to-s3 --timeout=4h
   ```

7. Confirm the checkpoint in S3 has 4 safetensors shards.
   ```bash
   aws s3 ls "s3://${WEIGHTS_BUCKET}/muse-glimmer-30b-fp8-quantized/" --region "$WEIGHTS_REGION" \
       | grep -c '\.safetensors$'
   ```

## Throughput test

1. Set the weights bucket and its region in the recipe; the deploy script refuses the placeholder.
   ```bash
   sed -i.bak -e "s|SUBSTITUTE-weights-bucket|${WEIGHTS_BUCKET}|" \
       -e "/name: AWS_REGION/{n;s|value: .*|value: \"${WEIGHTS_REGION}\"|;}" \
       vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml \
     && rm vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml.bak
   ```

2. Deploy the eight replicas; the script fails unless the server reports `TRITON_ATTN`.
   ```bash
   ./vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-deploy.sh
   ```

3. Record each replica's completed-request count before the run.
   ```bash
   APP=muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3
   BALANCE=docs/results/muse-glimmer-30b-fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-replica-balance.txt
   echo "== before measurement $(date -u '+%F %T UTC') ==" | tee "$BALANCE"
   for pod in $(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods -l app="$APP" -o name); do
     printf '  %s %s\n' "${pod#pod/}" "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec "$pod" -c vllm \
       -- curl -s localhost:8000/metrics | awk '/^vllm:request_success_total/ {s+=$NF} END {print s+0}')"
   done | tee -a "$BALANCE"
   ```

4. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
       --mode fingerprint \
       --model-tag muse-glimmer-30b --config fp8kv-32k-dp8-s3 \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --out-dir docs/results
   ```

5. Dry-run the in-cluster benchmark, which renders the Job without creating it.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
       --protocol corpus --profile coding_repo_context \
       --instance g7e.48xlarge --model-tag muse-glimmer-30b \
       --config fp8kv-32k-dp8-s3 --quant fp8 \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --tokenizer meta-models/Muse-Glimmer-30B \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

6. Benchmark in-cluster.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
       --protocol corpus --profile coding_repo_context \
       --instance g7e.48xlarge --model-tag muse-glimmer-30b \
       --config fp8kv-32k-dp8-s3 --quant fp8 \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --tokenizer meta-models/Muse-Glimmer-30B \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 -o jsonpath='{.items[0].metadata.name}')" \
       --reps 3 --out-dir docs/results
   ```

7. Record each replica's count again; an uneven split belongs beside the result.
   ```bash
   echo "== after measurement $(date -u '+%F %T UTC') ==" | tee -a "$BALANCE"
   for pod in $(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods -l app="$APP" -o name); do
     printf '  %s %s\n' "${pod#pod/}" "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec "$pod" -c vllm \
       -- curl -s localhost:8000/metrics | awk '/^vllm:request_success_total/ {s+=$NF} END {print s+0}')"
   done | tee -a "$BALANCE"
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml --ignore-not-found
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name gpu-spot-1node \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```

Scale the quantization and benchmark-client node groups to zero too if you created them for this run. The checkpoint in S3 survives, so a later run skips steps 4 to 6.

---

## Settings

- **Eight copies rather than one split copy.** The model fits one GPU, and g7e GPUs talk over PCIe with no NVLink, so tensor parallelism pays a transfer cost on every layer. Independent replicas trade that for one Service balancing requests between them.
- **The Service balances per connection, not per request.** The benchmark client holds one persistent connection per in-flight request, so at low concurrency some replicas can sit idle. Throughput steps 3 and 7 record `vllm:request_success_total` per pod so the split is on file beside the result.
- **`--pod` names one replica for provenance.** `measure` reads the image, KV pool and server log from that pod. Every replica runs the same spec.
- **The backend is pinned with `--attention-backend`, not `VLLM_ATTENTION_BACKEND`.** vLLM 0.29.0 does not read that environment variable and falls back to its auto-selected backend. The deploy script fails unless the server logs `Using AttentionBackendEnum.TRITON_ATTN backend`.
- **Why `TRITON_ATTN`.** It is portable across GPU generations and is the baseline the FlashInfer arm is read against. `FLASH_ATTN` refuses an FP8 KV cache on sm_120. Results from the two arms are separate labelled configs; never fold one into the other.
- **KV scales are uncalibrated.** vLLM 0.29.0 rejects `--calculate-kv-scales`, so each server logs `Using KV cache scaling factor 1.0 for fp8_e4m3`. This bounds any quality comparison.
- **`--no-enable-prefix-caching`** Held off for an uncached measurement. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`MAX_NUM_SEQS=64` and `MAX_CUDAGRAPH=64`.** Held at the Gemma 4 31B 32k recipe's values so the batch ceiling is not a variable between the two models. The pool would allow roughly 60 full-length sequences, and more at this workload's length.
- **`MAX_NUM_BATCHED_TOKENS=32768`.** A full `coding_repo_context` prompt prefills in one chunk.
- **Weights come from S3 into an `emptyDir`, not from the claim.** Each replica syncs its own copy, so a pod restart re-downloads 32 GiB. The `fetch-weights` init container fails unless it finds `config.json`, 4 shards and at least 28 GiB.
- **`serviceAccountName: muse-glimmer-30b-bench`** Supplies the pod-scoped S3 read identity through EKS Pod Identity. Node roles on clusters with an IMDS hop limit of 1 are unreachable from pods.
- **`WEIGHTS_BUCKET` and `AWS_REGION` are placeholders in git.** Throughput step 1 fills them in your working copy; keep that edit out of commits. The deploy script refuses to apply while the bucket placeholder remains.
- **The S3 staging Job sets no ServiceAccount.** It runs as the namespace's `default` ServiceAccount, so that is the identity the write role must be associated with.
- **The quantize Job runs in the serving image.** The quantizer image's transformers predates `model_type: muse_glimmer`, so the Job installs ModelOpt and a newer transformers into `vllm/vllm-openai:v0.29.0` at start. It keeps the image encoder, projector, embeddings and head in 16-bit.
