# GLM-5.2 NVFP4 on p6-b300.48xlarge

Serves [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) on p6-b300.48xlarge (8x B300-SXM6-AC, sm_100) with 8 GPUs per pod.

## Capacity sizing

GLM-5.2-NVFP4 at NVFP4 occupies 432.9 GiB of the 2,048.9 GiB on 8 B300-SXM6-AC cards, leaving 1,403.7 GiB for KV cache. At the recipe's 32,768 token context that pool holds 511 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 753B total | `zai-org/GLM-5.2-FP8` card; the NVFP4 card reports packed bytes, not parameters |
| **Layers** | 78 | `config.json`, `num_hidden_layers` |
| **KV latent per layer** | kv_lora_rank 512 plus qk_rope 64 | `config.json` |
| **Checkpoint size, NVFP4** | 432.9 GiB | Model card safetensors metadata |
| **Instance** | p6-b300.48xlarge, 8x B300-SXM6-AC at 275 GB | 2,200 GB nominal, 2,048.9 GiB |
| **Tensor parallel size** | 8 | `TP_SIZE` |
| **Weight precision** | NVFP4 | declared by the checkpoint's `quantization_config` |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 32,768 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `(kv_lora_rank + qk_rope_head_dim) x bytes_per_element`, which is 1.12 KiB here. Multi-head latent attention caches one compressed latent per layer instead of a key and value per head, so KV cost scales with layer count rather than head count. The 1,403.7 GiB pool is 2,048.9 GiB at 0.90 utilization, less 432.9 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

| Context per request | KV per sequence, per rank | Concurrent sequences |
|---|---|---|
| 4,096 tokens | 0.17 GiB | 337 |
| 8,192 tokens | 0.34 GiB | 168 |
| 32,768 tokens, this recipe's maximum | 1.37 GiB | 42 |

Derived from the 1,380,864-token pool this server actually allocated, divided by the context length. The deploy helper prints the live cache configuration on your hardware, and that value is the one to record.

## Before you begin

Complete the following tasks before you perform the steps in this runbook.

- [Create an EKS cluster with Karpenter, Karpenter discovery tags and node role](../../../scripts/create-cluster.sh)
- [Purchase a p6-b300.48xlarge Capacity Block for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-glm-52-nvfp4-p6-b300-48xlarge
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

5. Deploy the inference server and confirm it serves the model.

   ```bash
   ./vllm-recipes/glm-5.2/nvfp4-p6-b300.48xlarge-deploy.sh
   ```

6. Generate the accuracy gate (.JSON) that the benchmark checks against.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service glm-52-b300-nvfp4 \
       --mode fingerprint \
       --model-tag glm-5.2 --config nvfp4 \
       --model-id glm-5.2-nvfp4 \
       --thinking off \
       --out-dir docs/results
   ```

7. Dry-run the exact in-cluster benchmark.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service glm-52-b300-nvfp4 \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b300.48xlarge --model-tag glm-5.2 \
       --config nvfp4 --quant nvfp4 \
       --model-id glm-5.2-nvfp4 \
       --tokenizer nvidia/GLM-5.2-NVFP4 \
       --thinking off \
       --out-dir docs/results \
       --reps 3 \
       --dry-run
   ```

8. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service glm-52-b300-nvfp4 \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b300.48xlarge --model-tag glm-5.2 \
       --config nvfp4 --quant nvfp4 \
       --model-id glm-5.2-nvfp4 \
       --tokenizer nvidia/GLM-5.2-NVFP4 \
       --thinking off \
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/glm-5.2/nvfp4-p6-b300.48xlarge.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-cb-p6-b300-48xlarge ec2nodeclass/bench-cb-p6-b300-48xlarge --ignore-not-found
```

Deleting the server leaves the node empty, and Karpenter scales it to zero.
