# Kimi K3 (2.8T MoE) MXFP4, sts on p6-b200.48xlarge

Serves [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) on p6-b200.48xlarge (8x B200, sm_100) with 8 GPUs per pod.

## Capacity sizing

This recipe serves the published MXFP4 checkpoint [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) on 16 B200 GPUs. The checkpoint is about 1,454 GiB, and the recipe enforces 1,500 GiB free at `/models` before it starts. KV cost per sequence is exact and appears below; the pool and concurrency are read from the server.

| Property | Value | Source |
|---|---|---|
| **Layers** | 93 | `config.json`, `num_hidden_layers` |
| **KV latent per layer** | kv_lora_rank 512 plus qk_rope 64 | `config.json` |
| **Checkpoint size** | not derivable, 1,500 GiB free enforced at startup | `MIN_FREE_GIB` |
| **Instance** | p6-b200.48xlarge x2, 16x B200 at 180 GB | 2,880 GB across 2 nodes, 2,682.2 GiB |
| **Tensor parallel size** | 8 | `TP_SIZE` |
| **Pipeline parallel size** | 2 | `PP_SIZE` |
| **Weight precision** | MXFP4 | declared by the checkpoint's `quantization_config` |
| **KV cache precision** | FP8 (`fp8`) | `--kv-cache-dtype fp8` |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.9 | `--gpu-memory-utilization 0.9` |
| **Max sequences** | 128 | `MAX_NUM_SEQS` |

KV cache per layer per token is `(kv_lora_rank + qk_rope_head_dim) x bytes_per_element`, which is 0.56 KiB here. Multi-head latent attention caches one compressed latent per layer instead of a key and value per head, so KV cost scales with layer count rather than head count. The deploy helper reads the server's live cache configuration.

## Before you begin

- [Create two `p6-b200.48xlarge` nodes with EFA interfaces](../../../k8s/examples/eksctl-nodegroup-gpu-efa.yaml)
- [Create an ECR repository in your cluster's region](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) for the custom vLLM image
- [Build and push the custom vLLM image as `vllm-kimi-k3-efa:nccl2.29.7-efa1.49.0`](../../../k8s/examples/job-build-image-kaniko.yaml) (builds in-cluster on a CPU node)
- [Install the EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) so each node advertises `vpc.amazonaws.com/efa: 8`
- [Create an IAM role configured for EKS IRSA with read access to `kimi-k3-model-weights-<account-id>-<region>`](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html) (and write access, if the checkpoint is not in S3 yet)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export AWS_ACCOUNT_ID=<your-account-id>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-kimi-k3-mxfp4-p6-b200-sts
   export INSTANCE_TYPE=p6-b200.48xlarge
   export KIMI_S3_ROLE_ARN=<your-kimi-s3-read-role-arn>
   ```

2. Install the `eks-bench` CLI into a Python virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e '.[tokenizer]'
   ```

3. Configure the EKS context and namespace, install the NVIDIA device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

4. Copy the published checkpoint from Hugging Face to your S3 bucket.
   ```bash
   ./model-jobs/kimi-k3/mxfp4-model-download.sh
   ```

5. Deploy the vLLM inference server and confirm it serves the model.
   ```bash
   ./vllm-recipes/kimi-k3/mxfp4-p6-b200.48xlarge-sts-deploy.sh
   ```

6. Generate the accuracy gate (.JSON) that the benchmark checks against.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b200-api \
       --mode fingerprint \
       --model-tag kimi-k3 --config sts \
       --model-id kimi-k3 \
       --thinking off \
       --out-dir docs/results
   ```

7. Dry-run the exact in-cluster benchmark.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b200-api \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b200.48xlarge --model-tag kimi-k3 \
       --config sts --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --thinking off \
       --out-dir docs/results \
       --reps 3 \
       --dry-run
   ```

8. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b200-api \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b200.48xlarge --model-tag kimi-k3 \
       --config sts --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --thinking off \
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/kimi-k3/mxfp4-p6-b200.48xlarge-sts.yaml
```

## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **The checkpoint stays in S3.** A later run can skip Quickstart step 4; rerunning only copies files that are missing.
- **`publishNotReadyAddresses: true`** Publishes rank DNS before readiness and avoids a circular rendezvous wait.
- **`Service/kimi-k3-b200-api`** Selects only `kimi-k3-b200-0`, the rank that owns the API server.
- **`RANK="${HOSTNAME##*-}"`** Uses the StatefulSet ordinal as the distributed node rank.
- **`--headless`** Keeps the follower from starting another API server or driving collective RPC.
- **`TP_SIZE=8`, `PP_SIZE=2`** Keeps tensor parallelism within each NVLink node and crosses EFA only at pipeline boundaries.
- **`--no-enable-prefix-caching`** Turns off prefix caching, which vLLM enables by default. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`MAX_NUM_SEQS=128`** Pins the scheduler ceiling used by this recipe.
- **`memory: 1500Gi`** Constrains page-cache pressure inside the pod cgroup.
- **`emptyDir: { sizeLimit: 2000Gi }`** Is pod-local ephemeral storage rooted wherever kubelet stores `emptyDir`; the free-space check does not prove NVMe placement. A replaced pod syncs the checkpoint again.
- **`NCCL_DEBUG=INFO`** Records whether NCCL selected OFI/EFA rather than sockets.
- **`add: ["IPC_LOCK"]`** Supplies the capability required by EFA/libfabric.
- **`vllm-kimi-k3-efa:nccl2.29.7-efa1.49.0`** Carries matched NCCL and `aws-ofi-nccl` versions, recorded in the tag. Without the plugin NCCL can fall back to TCP while the server still starts.
- **The weights bucket is `kimi-k3-model-weights-<account-id>-<region>`.** The staging script creates it in `AWS_REGION` if it does not exist. The staging and deploy scripts fill the bucket name, `AWS_REGION` and the image registry from `AWS_ACCOUNT_ID` into each manifest as they apply it, so the files in git stay untouched.
