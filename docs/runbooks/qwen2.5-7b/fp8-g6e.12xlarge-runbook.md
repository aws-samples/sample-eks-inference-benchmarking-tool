# Qwen2.5-7B-Instruct FP8 on g6e.12xlarge

Serves [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on g6e.12xlarge (4x L40S, sm_89) with 1 GPU per pod.

## Capacity sizing

Qwen2.5-7B-Instruct at ModelOpt FP8 occupies 8.1 GiB of the 44.4 GiB on 1 L40S card, leaving about 28.5 GiB for KV cache. At the model maximum 32,768 token context that pool holds 16 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 7B | Model card |
| **Layers** | 28 | `config.json`, `num_hidden_layers` |
| **KV heads, head dimension** | 4, 128 | `config.json` |
| **Checkpoint size, FP8** | 8.1 GiB | `model.safetensors` from the quantize step; embeddings and `lm_head` stay BF16 |
| **Instance** | g6e.12xlarge, 1 of 4x L40S at 48 GB | 44.4 GiB as CUDA reports it |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | ModelOpt FP8 | `QUANTIZATION` |
| **KV cache precision** | 16-bit | No `--kv-cache-dtype`; see Settings |
| **Attention backend** | `TRITON_ATTN` | `ATTENTION_BACKEND`; see Settings |
| **Context length** | 32,768, not pinned by the recipe | `config.json`, `max_position_embeddings` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 2 KiB here, or 56 KiB per token across all 28 layers. The 28.5 GiB pool is 44.4 GiB at 0.95 utilization, less 8.1 GiB of weights, less roughly 5.5 GiB for activations and CUDA graphs. Read `KV cache size` in the result file's provenance block for the allocation in force.

`legacy_corpus` sends 38 wire input tokens and asks for up to 150 output tokens, so 512 concurrent sequences hold under 100k tokens, about 5.1 GiB of KV. `--max-num-seqs 512`, not KV memory, caps the batch for this workload.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create an EKS cluster with Karpenter, Karpenter discovery tags and node role](../../../scripts/create-cluster.sh)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export AWS_ACCOUNT_ID=<your-account-id>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen25-7b-fp8-g6e-12xlarge
   export INSTANCE_TYPE=g6e.12xlarge
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

5. Build the quantizer tool image in-cluster with kaniko, create the ECR repository, and push it. This step takes several minutes to complete. Alternatively, to build locally on x86 Docker, run `./scripts/push-image.sh`.
   ```bash
   ./scripts/push-image-kaniko.sh
   ```

6. Deploy the inference server, whose init container quantizes Qwen2.5-7B to FP8 on NVMe on first start, and confirm it serves the model.

   ```bash
   ./vllm-recipes/qwen2.5-7b/fp8-g6e.12xlarge-deploy.sh
   ```

7. Generate the accuracy gate (.JSON) that the benchmark checks against.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-12xl-fp8 \
       --mode fingerprint \
       --model-tag qwen2.5-7b --config fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --out-dir docs/results
   ```

8. Dry-run the exact in-cluster benchmark.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-12xl-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.12xlarge --model-tag qwen2.5-7b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --out-dir docs/results \
       --reps 3 \
       --dry-run
   ```

9. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g6e-12xl-fp8 \
       --protocol corpus --profile legacy_corpus \
       --instance g6e.12xlarge --model-tag qwen2.5-7b \
       --config fp8 --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --out-dir docs/results \
       --reps 3
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/qwen2.5-7b/fp8-g6e.12xlarge.yaml --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-g6e-12xlarge ec2nodeclass/bench-g6e-12xlarge --ignore-not-found
```
