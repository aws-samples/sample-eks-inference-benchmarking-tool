# Muse-Glimmer-30B FP8 with an FP8 KV cache on g7e.48xlarge, eight replicas, FlashInfer attention

Serves a locally quantized ModelOpt FP8 build of [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) as eight independent single-GPU replicas behind one Service, using [`fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer.yaml`](../../../vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer.yaml).

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
| **Attention backend** | `FLASHINFER` | `ATTENTION_BACKEND` |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Max sequences** | 64 per replica | `MAX_NUM_SEQS` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |

## Before you begin

Complete the following tasks before you perform the steps in this runbook.

- [Create an EKS cluster with Karpenter with `./scripts/create-cluster.sh`](../../../scripts/create-cluster.sh)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem)

## Quickstart

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-muse-glimmer-30b-fp8kv-32k-dp8-flashinfer-g7e-48xlarge
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

4. Provision a GPU node pool with NVMe RAID0 for the instance type.
   ```bash
   ./scripts/provision-capacity.sh "$INSTANCE_TYPE"
   ```

5. Provision an on-demand GPU node pool with NVMe RAID0 for the quantizer.
   ```bash
   ./scripts/provision-capacity.sh g7e.2xlarge
   ```

6. Install the Pod Identity Agent and grant the `muse-glimmer-30b-bench` ServiceAccount S3 access.
   ```bash
   ./scripts/setup-pod-identity.sh muse-glimmer-30b-bench
   ```

7. Quantize the model to ModelOpt FP8 and save the checkpoint to the `fsx-pvc` claim.
   ```bash
   ./model-jobs/muse-glimmer-30b/fp8-model-quantize.sh
   ```

8. Copy the quantized checkpoint from the claim to your S3 bucket.
   ```bash
   ./model-jobs/muse-glimmer-30b/fp8-model-fsx-to-s3.sh
   ```

## Throughput test

1. Deploy eight vLLM inference server replicas and confirm they serve the model.
   ```bash
   ./vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer-deploy.sh
   ```

2. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3-flashinfer \
       --mode fingerprint \
       --model-tag muse-glimmer-30b --config fp8kv-32k-dp8-s3-flashinfer \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --out-dir docs/results
   ```

3. Dry-run the exact in-cluster benchmark.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3-flashinfer \
       --protocol corpus --profile coding_repo_context \
       --instance g7e.48xlarge --model-tag muse-glimmer-30b \
       --config fp8kv-32k-dp8-s3-flashinfer --quant fp8 \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --tokenizer meta-models/Muse-Glimmer-30B \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3-flashinfer -o jsonpath='{.items[0].metadata.name}')" \
       --out-dir docs/results \
       --reps 3 \
       --dry-run
   ```

4. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3-flashinfer \
       --protocol corpus --profile coding_repo_context \
       --instance g7e.48xlarge --model-tag muse-glimmer-30b \
       --config fp8kv-32k-dp8-s3-flashinfer --quant fp8 \
       --model-id /models/quantized/Muse-Glimmer-30B-fp8 \
       --tokenizer meta-models/Muse-Glimmer-30B \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=muse-glimmer-30b-vllm-g7e-48xl-fp8kv-32k-dp8-s3-flashinfer -o jsonpath='{.items[0].metadata.name}')" \
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/muse-glimmer-30b/fp8-g7e.48xlarge-fp8kv-32k-dp8-s3-flashinfer.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-g7e-48xlarge ec2nodeclass/bench-g7e-48xlarge --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-g7e-2xlarge ec2nodeclass/bench-g7e-2xlarge --ignore-not-found
```

Deleting the server leaves the node empty, and Karpenter scales both the serving and quantize pools to zero. The checkpoint in S3 survives, so a later run skips weights staging in previous steps.
