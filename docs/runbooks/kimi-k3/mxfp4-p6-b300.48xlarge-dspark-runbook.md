# Kimi K3 (2.8T MoE) MXFP4, dspark on p6-b300.48xlarge

Serves [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) on p6-b300.48xlarge (8x B300-SXM6-AC, sm_100) with 8 GPUs per pod.

## Capacity sizing

This recipe serves a locally quantized checkpoint and DSpark draft on 8 B300-SXM6-AC GPUs. The main checkpoint size is not derivable from the model card, and the recipe enforces 1,500 GiB free at `/models` before staging. KV cost per sequence is exact and appears below; the pool and concurrency are read from the server.

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
| **Speculative tokens** | 7 | `NUM_SPEC_TOKENS` |

KV cache per layer per token is `(kv_lora_rank + qk_rope_head_dim) x bytes_per_element`, which is 0.56 KiB here. Multi-head latent attention caches one compressed latent per layer instead of a key and value per head, so KV cost scales with layer count rather than head count. The deploy helper reads the server's live cache configuration.

## Before you begin

- [Create a `p6-b300.48xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the NVIDIA device plugin](../../../README.md)
- [Create an IAM role configured for EKS IRSA with read access to the main and DSpark S3 checkpoint prefixes](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html)

## Setup

This recipe uses a custom vLLM image and carries a placeholder ECR account and region in `image:`. The image includes matched NCCL and `aws-ofi-nccl` versions for a reproducible Kimi K3 serving environment; the EFA plugin is not required for this single-node deployment.

1. [Create an ECR repository](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) for the image.

2. Build and push the custom vLLM image. [`job-build-image-kaniko.yaml`](../../../k8s/examples/job-build-image-kaniko.yaml) can build it in-cluster on a CPU node when local Docker is unavailable.

3. Replace account `012345678901` and region `us-east-1` in the recipe's `image:` reference with the image you pushed. The deploy helper refuses to apply a manifest that still contains the placeholder account.

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-kimi-k3-mxfp4-p6-b300-dspark
   export INSTANCE_TYPE=p6-b300.48xlarge
   export KIMI_S3_ROLE_ARN=<your-kimi-s3-read-role-arn>
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Idempotently create the namespace-local `kimi-k3-bench` ServiceAccount and annotate it for EKS IRSA.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" create serviceaccount kimi-k3-bench --dry-run=client -o yaml | kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f -
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" annotate serviceaccount kimi-k3-bench eks.amazonaws.com/role-arn="$KIMI_S3_ROLE_ARN" --overwrite
   ```

4. Confirm the namespace, Ready GPU node, created IRSA ServiceAccount, and both S3 prefixes are available.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" get namespace "$NAMESPACE"
   kubectl --context "$EKS_CLUSTER_NAME" get nodes -l node.kubernetes.io/instance-type=p6-b300.48xlarge
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get serviceaccount kimi-k3-bench
   aws s3 ls s3://kimi-k3-mxfp4-weights/kimi-k3-mxfp4/ | head
   aws s3 ls s3://kimi-k3-mxfp4-weights/kimi-k3-dspark/ | head
   ```

5. Deploy and validate the server. The init container syncs the main checkpoint and DSpark draft from S3 into pod-local `emptyDir` storage before vLLM starts; Kubernetes does not guarantee that storage is instance NVMe, and replacing the pod repeats both syncs. The helper waits for the Deployment, checks `/v1/models` from its `vllm` container, reads live cache evidence, and validates the manifest.

   ```bash
   ./vllm-recipes/kimi-k3/mxfp4-p6-b300.48xlarge-dspark-deploy.sh
   ```

6. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   python3 app/cli.py launch --capture-accuracy-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300-dspark \
       --mode fingerprint \
       --model-tag kimi-k3 --config dspark \
       --model-id kimi-k3 \
       --thinking off \
       --out-dir docs/results
   ```

7. Dry-run the exact in-cluster benchmark.

   ```bash
   python3 app/cli.py launch \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300-dspark \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b300.48xlarge --model-tag kimi-k3 \
       --config dspark --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --thinking off \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

8. Run the identical benchmark without `--dry-run`.

   ```bash
   python3 app/cli.py launch \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300-dspark \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b300.48xlarge --model-tag kimi-k3 \
       --config dspark --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --thinking off \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/kimi-k3/mxfp4-p6-b300.48xlarge-dspark.yaml
```

## Settings

- **`serviceAccountName: kimi-k3-bench`** Supplies the pod-scoped S3 identity used by the init container.
- **`test -n "$(ls -A /models/kimi-k3-dspark 2>/dev/null)"`** Fails before model loading when the staged draft prefix is empty.
- **`--speculative-config`** Loads `/models/kimi-k3-dspark` with the `dspark` method, seven speculative tokens, probabilistic draft sampling, and block rejection sampling.
- **`--max-model-len 32768`** Pins the served context length.
- **`--max-cudagraph-capture-size 512`** Covers the configured scheduler ceiling.
- **`memory: 3000Gi`** Constrains page-cache pressure inside the pod cgroup while leaving node headroom.
- **`emptyDir: { sizeLimit: 2000Gi }`** Is pod-local ephemeral storage rooted wherever kubelet stores `emptyDir`; the free-space check does not prove NVMe placement.
- **`failureThreshold: 280`** Allows up to 70 minutes for both syncs and server startup.
