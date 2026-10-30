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
| **Image** | `vllm/vllm-openai:v0.31.0` (upstream) | `image` |

KV cache per layer per token is `(kv_lora_rank + qk_rope_head_dim) x bytes_per_element`, which is 0.56 KiB here. Multi-head latent attention caches one compressed latent per layer instead of a key and value per head, so KV cost scales with layer count rather than head count. The deploy helper reads the server's live cache configuration.

## Before you begin

- [Create an EKS cluster with Karpenter with `./scripts/create-cluster.sh`](../../../scripts/create-cluster.sh)
- [Purchase a p6-b300.48xlarge Capacity Block for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-kimi-k3-mxfp4-p6-b300-baseline
   export INSTANCE_TYPE=p6-b300.48xlarge
   export RESERVATION_ID=<your-capacity-block-id>
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

4. Provision a Capacity Block GPU node pool with NVMe RAID0 for the instance type.
   ```bash
   ./scripts/provision-capacity-block.sh "$INSTANCE_TYPE" "$RESERVATION_ID"
   ```

5. Install the Pod Identity Agent and grant the `kimi-k3-bench` ServiceAccount S3 access.
   ```bash
   ./scripts/setup-pod-identity.sh kimi-k3-bench
   ```

6. Copy the published checkpoint from Hugging Face to your S3 bucket.
   ```bash
   ./model-jobs/kimi-k3/mxfp4-model-download.sh
   ```

7. Deploy the inference server and confirm it serves the model.
   ```bash
   ./vllm-recipes/kimi-k3/mxfp4-p6-b300.48xlarge-baseline-deploy.sh
   ```

8. Generate the accuracy gate (.JSON) that the benchmark checks against.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300 \
       --mode fingerprint \
       --model-tag kimi-k3 --config baseline \
       --model-id kimi-k3 \
       --thinking off \
       --out-dir docs/results
   ```

9. Dry-run the exact in-cluster benchmark.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300 \
       --protocol corpus --profile chat_typical \
       --instance p6-b300.48xlarge --model-tag kimi-k3 \
       --config baseline --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --out-dir docs/results \
       --reps 1 \
       --dry-run
   ```

10. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300 \
       --protocol corpus --profile chat_typical \
       --instance p6-b300.48xlarge --model-tag kimi-k3 \
       --config baseline --quant mxfp4 \
       --model-id kimi-k3 \
       --tokenizer moonshotai/Kimi-K3 \
       --out-dir docs/results \
       --reps 1
   ```

## (Optional) Open-loop capacity test

Measures the real-time capacity under Poisson arrivals: the request rate at which TTFT p99 stays within 500 ms and TPOT p99 within 50 ms. Open-loop figures are not comparable to the in-cluster concurrency sweep. Requires the server from step 7 to be running.

1. (Optional) Dry-run the open-loop capacity test.

   ```bash
   ./vllm-recipes/kimi-k3/mxfp4-p6-b300.48xlarge-baseline-server-qps.sh --dry-run
   ```

2. (Optional) Run the open-loop capacity test.

   ```bash
   ./vllm-recipes/kimi-k3/mxfp4-p6-b300.48xlarge-baseline-server-qps.sh
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/kimi-k3/mxfp4-p6-b300.48xlarge-baseline.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-cb-p6-b300-48xlarge ec2nodeclass/bench-cb-p6-b300-48xlarge --ignore-not-found
```

Deleting the server leaves the node empty, and Karpenter scales it to zero.
