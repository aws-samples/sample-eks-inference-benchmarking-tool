# Qwen2.5-7B-Instruct FP8, kvscales on g7e.12xlarge

Serves [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on g7e.12xlarge (2x RTX PRO 6000 Blackwell, sm_120) with 1 GPU per pod.

## Capacity sizing

Qwen2.5-7B-Instruct at FP8 occupies 7.4 GiB of the 89.4 GiB on 1 RTX PRO 6000 Blackwell card, leaving 70.1 GiB for KV cache. At the model maximum 32,768 token context that pool holds 80 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 7B | Model card |
| **Layers** | 28 | `config.json`, `num_hidden_layers` |
| **KV heads, head dimension** | 4, 128 | `config.json` |
| **Checkpoint size, FP8** | 7.4 GiB | 1 byte per parameter, derived from the BF16 card |
| **Instance** | g7e.12xlarge, 1 of 2x RTX PRO 6000 Blackwell at 96 GB | 96 GB for this pod, 89.4 GiB |
| **Tensor parallel size** | 1 | `TP_SIZE` |
| **Weight precision** | FP8 | `QUANTIZATION` |
| **KV cache precision** | FP8 (`fp8_e4m3`) | `--kv-cache-dtype fp8_e4m3` |
| **Context length** | 32,768, not pinned by the recipe | `config.json`, `max_position_embeddings` |
| **Memory fraction** | 0.95 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 1 KiB here. Every layer attends to the full context, so per-sequence cost scales linearly with context length across all 28 layers. The 70.1 GiB pool is 89.4 GiB at 0.95 utilization, less 7.4 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.12xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)
- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen25-7b-fp8-g7e-12xlarge-kvscales
   export INSTANCE_TYPE=g7e.12xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the persistent PVC, then stage the FP8 checkpoint used for live KV-scale calculation. A completed earlier download Job makes staging a no-op.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-downloads/qwen2.5-7b-fp8.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait --for=condition=complete job/download-qwen25-7b-fp8 --timeout=6h
   ```

4. Deploy and validate the model server mounted to `fsx-pvc`.

   ```bash
   ./vllm-recipes/qwen2.5-7b/fp8-g7e.12xlarge-kvscales-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g7e-kvscales \
       --mode fingerprint \
       --model-tag qwen2.5-7b --config kvscales \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --out-dir docs/results
   ```

6. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g7e-kvscales \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen2.5-7b \
       --config kvscales --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

7. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen7b-vllm-g7e-kvscales \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen2.5-7b \
       --config kvscales --quant fp8 \
       --model-id /models/quantized/Qwen2.5-7B-Instruct-fp8 \
       --tokenizer Qwen/Qwen2.5-7B-Instruct \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/qwen2.5-7b/fp8-g7e.12xlarge-kvscales.yaml
```

---


## Settings


- **`- "/models/quantized/Qwen2.5-7B-Instruct-fp8"`** Serve the FP8-quantized model from the PVC

- **`- "--quantization=fp8"`** FP8 quantization: halves memory traffic vs BF16, unlocks Blackwell 2x FP8 compute

- **`- "--calculate-kv-scales"`** Compute KV cache scaling factors dynamically from live activations. Eliminates the "no proper scaling factor" warning and prevents accuracy drift on FP8 KV cache without requiring an offline calibration pass.

- **`- "--gpu-memory-utilization=0.95"`** Blackwell-optimized concurrency: extra VRAM headroom from FP8 allows higher batching

- **`- "--port=8000"`** and improves scheduling fairness under mixed workloads
- **`nvidia.com/gpu.present: "true"`** Targets any GPU node rather than one instance type, so the pod schedules wherever capacity exists. Pin `node.kubernetes.io/instance-type` when the result has to be attributable to a specific instance.
- **`nvidia.com/gpu: 1`** g7e.12xlarge = 2 GPUs available (96GB each), using 1 for 7B model
- **`- "--enable-chunked-prefill"`** Breaks long prefills into chunks; reduces TTFT for long prompts
- **`VLLM_LOGGING_LEVEL=WARNING`** Suppresses per-request INFO logs; errors and warnings still surface
- **`claimName: fsx-pvc`** A ReadWriteMany FSx for Lustre claim.
