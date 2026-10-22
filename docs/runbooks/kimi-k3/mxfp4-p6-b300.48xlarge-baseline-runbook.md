# Kimi K3 (2.8T MoE) MXFP4, baseline on p6-b300.48xlarge

Serves [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) on p6-b300.48xlarge (8x B300-SXM6-AC, sm_100) with 8 GPUs per pod.

## Capacity sizing

This recipe serves the published MXFP4 checkpoint [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) on 8 B300-SXM6-AC GPUs. The checkpoint is about 1,454 GiB, and the recipe enforces 1,500 GiB free at `/models` before it starts. KV cost per sequence is exact and appears below; the pool and concurrency are read from the server.

| Property | Value | Source |
|---|---|---|
| **Layers** | 93 | `config.json`, `num_hidden_layers` |
| **KV latent per layer** | kv_lora_rank 512 plus qk_rope 64 | `config.json` |
| **Checkpoint size** | not derivable, 1,500 GiB free enforced at startup | `MIN_FREE_GIB` |
| **Instance** | p6-b300.48xlarge, 8x B300-SXM6-AC at 275 GB | 2,200 GB nominal, 2,048.9 GiB |
| **Tensor parallel size** | 8 | `--tensor-parallel-size 8` |
| **Weight precision** | MXFP4 | declared by the checkpoint's `quantization_config` |
| **KV cache precision** | FP8 (`fp8`) | `--kv-cache-dtype fp8` |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.9 | `--gpu-memory-utilization 0.9` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `(kv_lora_rank + qk_rope_head_dim) x bytes_per_element`, which is 0.56 KiB here. Multi-head latent attention caches one compressed latent per layer instead of a key and value per head, so KV cost scales with layer count rather than head count. The deploy helper reads the server's live cache configuration.

## Before you begin

- [Create a `p6-b300.48xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Create an ECR repository in your cluster's region](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) for the custom vLLM image
- [Build and push the custom vLLM image as `vllm-kimi-k3-efa:nccl2.29.7-efa1.49.0`](../../../k8s/examples/job-build-image-kaniko.yaml) (builds in-cluster on a CPU node)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export AWS_ACCOUNT_ID=<your-account-id>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-kimi-k3-mxfp4-p6-b300-baseline
   export INSTANCE_TYPE=p6-b300.48xlarge
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

4. Install the Pod Identity Agent and grant the `kimi-k3-bench` ServiceAccount S3 access.
   ```bash
   ./scripts/setup-pod-identity.sh kimi-k3-bench
   ```

5. Copy the published checkpoint from Hugging Face to your S3 bucket.
   ```bash
   ./model-jobs/kimi-k3/mxfp4-model-download.sh
   ```

6. Deploy the vLLM inference server and confirm it serves the model.
   ```bash
   ./vllm-recipes/kimi-k3/mxfp4-p6-b300.48xlarge-baseline-deploy.sh
   ```

7. Generate the accuracy gate (.JSON) that the benchmark checks against.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300 \
       --mode fingerprint \
       --model-tag kimi-k3 --config baseline \
       --model-id kimi-k3 \
       --thinking off \
       --out-dir docs/results
   ```

8. Dry-run the exact in-cluster benchmark.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300 \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b300.48xlarge --model-tag kimi-k3 \
       --config baseline --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --thinking off \
       --out-dir docs/results \
       --reps 3 \
       --dry-run
   ```

9. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300 \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b300.48xlarge --model-tag kimi-k3 \
       --config baseline --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --thinking off \
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/kimi-k3/mxfp4-p6-b300.48xlarge-baseline.yaml
```

## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **The checkpoint stays in S3.** A later run can skip Quickstart step 4; rerunning only copies files that are missing.
- **`serviceAccountName: kimi-k3-bench`** Supplies the pod-scoped S3 identity used by the init container.
- **`aws configure set default.s3.max_concurrent_requests 64`** Increases S3 transfer concurrency for the checkpoint sync.
- **`test "$SZ" -ge 1400`** Fails before model loading if the staged checkpoint is incomplete.
- **`--max-model-len 32768`** Pins the served context length.
- **`--max-cudagraph-capture-size 512`** Covers the configured scheduler ceiling.
- **`--no-enable-prefix-caching`** Turns off prefix caching, which vLLM enables by default. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`memory: 3000Gi`** Constrains page-cache pressure inside the pod cgroup while leaving node headroom.
- **`emptyDir: { sizeLimit: 2000Gi }`** Is pod-local ephemeral storage rooted wherever kubelet stores `emptyDir`; the free-space check does not prove NVMe placement. A replaced pod syncs the checkpoint again.
- **`vllm-kimi-k3-efa:nccl2.29.7-efa1.49.0`** Carries matched NCCL and `aws-ofi-nccl` versions, recorded in the tag; this single-node recipe does not use EFA.
- **The weights bucket is `kimi-k3-model-weights-<account-id>-<region>`.** The staging script creates it in `AWS_REGION` if it does not exist. The staging and deploy scripts fill the bucket name, `AWS_REGION` and the image registry from `AWS_ACCOUNT_ID` into each manifest as they apply it, so the files in git stay untouched.
