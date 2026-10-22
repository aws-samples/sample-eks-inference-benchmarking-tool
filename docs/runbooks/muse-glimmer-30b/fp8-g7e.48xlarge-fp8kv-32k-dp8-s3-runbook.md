# Muse-Glimmer-30B FP8 with an FP8 KV cache on g7e.48xlarge, eight replicas, Triton attention

Serves a locally quantized ModelOpt FP8 build of [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) as eight independent single-GPU replicas behind one Service, using [`fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml`](../../../vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml).

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
- [Associate an S3 read role with the `muse-glimmer-30b-bench` ServiceAccount in your namespace through Pod Identity](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-association.html)
- [Create a `g7e.2xlarge` node group for quantization](../../../k8s/examples/eksctl-nodegroup-g7e-2xl-spot.yaml) (only if the checkpoint is not in S3 yet)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (only if the checkpoint is not in S3 yet)
- [Grant S3 write access to the staging Job](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-association.html) (only if the checkpoint is not in S3 yet)

## Quickstart

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-muse-glimmer-30b-fp8kv-32k-dp8-g7e-48xlarge
   export INSTANCE_TYPE=g7e.48xlarge
   ```

2. Install the `eks-bench` CLI into a Python virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e '.[tokenizer]'
   ```

3. Configure the EKS context and namespace, install the NVIDIA device plugin, create the `hf-token` secret, and grant ECR push permissions.
   ```bash
   ./scripts/install-prerequisites.sh
   ```

4. Quantize the model to ModelOpt FP8 and save the checkpoint to the `fsx-pvc` claim.
   ```bash
   ./model-jobs/muse-glimmer-30b/fp8-model-quantize.sh
   ```

5. Copy the quantized checkpoint from the claim to your S3 bucket.
   ```bash
   ./model-jobs/muse-glimmer-30b/fp8-model-fsx-to-s3.sh
   ```

## Throughput test

1. Deploy eight vLLM inference server replicas and confirm they serve the model.
   ```bash
   ./vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-deploy.sh
   ```

2. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3 \
       --mode fingerprint \
       --model-tag muse-glimmer-30b --config fp8kv-32k-dp8-s3 \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --out-dir docs/results
   ```

3. Dry-run the exact in-cluster benchmark.
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
       --out-dir docs/results \
       --reps 3 \
       --dry-run
   ```

4. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).
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
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3.yaml --ignore-not-found
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name gpu-spot-1node \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```

Scale the quantization and benchmark-client node groups to zero too if you created them for this run. The checkpoint in S3 survives, so a later run skips Quickstart steps 4 and 5.

---

## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **Eight copies rather than one split copy.** The model fits one GPU, and g7e GPUs talk over PCIe with no NVLink, so tensor parallelism pays a transfer cost on every layer. Independent replicas trade that for one Service balancing requests between them.
- **The Service balances per connection, not per request.** The benchmark client holds one persistent connection per in-flight request, so at low concurrency some replicas can sit idle.
- **`--pod` names one replica for provenance.** `measure` reads the image, KV pool and server log from that pod. Every replica runs the same spec.
- **The backend is pinned with `--attention-backend`, not `VLLM_ATTENTION_BACKEND`.** vLLM 0.29.0 does not read that environment variable and falls back to its auto-selected backend. The deploy script fails unless the server logs `Using AttentionBackendEnum.TRITON_ATTN backend`.
- **Why `TRITON_ATTN`.** It serves an FP8 KV cache on sm_120, where `FLASH_ATTN` refuses one.
- **KV scales are uncalibrated.** vLLM 0.29.0 rejects `--calculate-kv-scales`, so each server logs `Using KV cache scaling factor 1.0 for fp8_e4m3`. This bounds any quality comparison.
- **`--no-enable-prefix-caching`** Held off for an uncached measurement. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`MAX_NUM_SEQS=64` and `MAX_CUDAGRAPH=64`.** Held at the Gemma 4 31B 32k recipe's values so the batch ceiling is not a variable between the two models. The pool would allow roughly 60 full-length sequences, and more at this workload's length.
- **`MAX_NUM_BATCHED_TOKENS=32768`.** A full `coding_repo_context` prompt prefills in one chunk.
- **Weights come from S3 into an `emptyDir`, not from the claim.** Each replica syncs its own copy, so a pod restart re-downloads 32 GiB. The `fetch-weights` init container fails unless it finds `config.json`, 4 shards and at least 28 GiB.
- **`serviceAccountName: muse-glimmer-30b-bench`** Supplies the pod-scoped S3 read identity through EKS Pod Identity. Node roles on clusters with an IMDS hop limit of 1 are unreachable from pods.
- **The weights bucket is `muse-glimmer-30b-model-weights-<account-id>-<region>`.** The staging script creates it in `AWS_REGION` if it does not exist. The staging and deploy scripts fill the bucket name and `AWS_REGION` into each manifest as they apply it, so the files in git stay untouched.
- **The S3 staging Job sets no ServiceAccount.** It runs as the namespace's `default` ServiceAccount, so that is the identity the write role must be associated with.
- **The quantize Job runs in the serving image.** The quantizer image's transformers predates `model_type: muse_glimmer`, so the Job installs ModelOpt and a newer transformers into `vllm/vllm-openai:v0.29.0` at start. It keeps the image encoder, projector, embeddings and head in 16-bit.
