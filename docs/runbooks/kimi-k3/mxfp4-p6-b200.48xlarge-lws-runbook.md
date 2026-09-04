# Kimi K3 (2.8T MoE) MXFP4, lws on p6-b200.48xlarge

Serves [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) on p6-b200.48xlarge (8x B200, sm_100) with 8 GPUs per pod.

## Capacity sizing

This recipe serves a locally quantized checkpoint on 16 B200 GPUs. Its size is not derivable from the model card, and the recipe enforces 1,500 GiB free at `/models` before it starts. KV cost per sequence is exact and appears below; the pool and concurrency are read from the server.

| Property | Value | Source |
|---|---|---|
| **Layers** | 93 | `config.json`, `num_hidden_layers` |
| **KV latent per layer** | kv_lora_rank 512 plus qk_rope 64 | `config.json` |
| **Checkpoint size** | not derivable, 1,500 GiB free enforced at startup | `MIN_FREE_GIB` |
| **Instance** | p6-b200.48xlarge x2, 16x B200 at 180 GB | 2,880 GB across 2 nodes, 2,682.2 GiB |
| **Tensor parallel size** | 16 | `TP_SIZE` |
| **Weight precision** | MXFP4 | declared by the checkpoint's `quantization_config` |
| **KV cache precision** | FP8 (`fp8`) | `--kv-cache-dtype fp8` |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.9 | `--gpu-memory-utilization 0.9` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `(kv_lora_rank + qk_rope_head_dim) x bytes_per_element`, which is 0.56 KiB here. Multi-head latent attention caches one compressed latent per layer instead of a key and value per head, so KV cost scales with layer count rather than head count. The deploy helper reads the server's live cache configuration.

## Before you begin

- [Create two `p6-b200.48xlarge` nodes with EFA interfaces](../../../k8s/examples/eksctl-nodegroup-gpu-efa.yaml)
- [Install the NVIDIA device plugin](../../../README.md)
- [Install the EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) so each node advertises `vpc.amazonaws.com/efa: 8`
- [Install the LeaderWorkerSet controller](https://github.com/kubernetes-sigs/lws)
- [Create an IAM role configured for EKS IRSA with read access to the S3 checkpoint prefix](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html)

## Setup

This recipe requires a custom vLLM image containing a matched `aws-ofi-nccl` plugin and uses a placeholder ECR account and region in `image:`. Without the plugin, NCCL can fall back to TCP while the server still starts, invalidating the measurement. The image tag records the NCCL and EFA versions so a mismatch remains visible.

1. [Create an ECR repository](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) for the image.

2. Build and push the custom vLLM image with `aws-ofi-nccl` installed. [`job-build-image-kaniko.yaml`](../../../k8s/examples/job-build-image-kaniko.yaml) can build it in-cluster on a CPU node when local Docker is unavailable.

3. Replace account `012345678901` and region `us-east-1` in the recipe's `image:` reference with the image you pushed. The deploy helper refuses to apply a manifest that still contains the placeholder account.

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-kimi-k3-mxfp4-p6-b200-lws
   export INSTANCE_TYPE=p6-b200.48xlarge
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

4. Confirm the namespace, two Ready GPU nodes, LeaderWorkerSet CRD, EFA resources, created IRSA ServiceAccount, and S3 checkpoint are available.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" get namespace "$NAMESPACE"
   kubectl --context "$EKS_CLUSTER_NAME" get nodes -l node.kubernetes.io/instance-type=p6-b200.48xlarge
   kubectl --context "$EKS_CLUSTER_NAME" get crd leaderworkersets.leaderworkerset.x-k8s.io
   kubectl --context "$EKS_CLUSTER_NAME" get nodes -l node.kubernetes.io/instance-type=p6-b200.48xlarge -o custom-columns='NAME:.metadata.name,EFA:.status.allocatable.vpc\.amazonaws\.com/efa'
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get serviceaccount kimi-k3-bench
   aws s3 ls s3://kimi-k3-mxfp4-weights/kimi-k3-mxfp4/ | head
   ```

5. Deploy and validate the server. Each LWS pod independently syncs the S3 checkpoint into its own pod-local `emptyDir` before vLLM starts; Kubernetes does not guarantee that storage is instance NVMe, and replacing either pod requires another sync. The helper discovers the leader through stable `app=kimi-k3-b200,role=leader` labels, waits for that Pod to become Ready, checks `/v1/models`, reads live cache evidence, and validates the manifest.

   ```bash
   ./vllm-recipes/kimi-k3/mxfp4-p6-b200.48xlarge-lws-deploy.sh
   ```

6. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   python3 app/cli.py launch --capture-accuracy-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b200-lws \
       --mode fingerprint \
       --model-tag kimi-k3 --config lws \
       --model-id kimi-k3 \
       --thinking off \
       --out-dir docs/results
   ```

7. Dry-run the exact in-cluster benchmark.

   ```bash
   python3 app/cli.py launch \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b200-lws \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b200.48xlarge --model-tag kimi-k3 \
       --config lws --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --thinking off \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

8. Run the identical benchmark without `--dry-run`.

   ```bash
   python3 app/cli.py launch \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b200-lws \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b200.48xlarge --model-tag kimi-k3 \
       --config lws --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --thinking off \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/kimi-k3/mxfp4-p6-b200.48xlarge-lws.yaml
```

## Settings

- **`aws configure set default.s3.max_concurrent_requests 64`** Increases S3 transfer concurrency for the checkpoint sync.
- **`test "$SZ" -ge 1400`** Fails before model loading if the staged checkpoint is incomplete.
- **`--all2all-backend deepep_v2`** Selects the recipe's cross-node expert communication backend.
- **No `--enable-prefix-caching`** Keeps prefix caching disabled for this configuration.
- **`emptyDir: { sizeLimit: 2000Gi }`** Is pod-local ephemeral storage rooted wherever kubelet stores `emptyDir`; the startup free-space check prevents an unsafe sync, but does not prove NVMe placement.
- **`NCCL_DEBUG=INFO`** Records the selected NCCL transport so EFA use can be verified with the run provenance.
- **`vpc.amazonaws.com/efa: "8"`** Requires eight EFA devices on each pod's node.
