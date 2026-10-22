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

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `p6-b300.48xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-glm-52-nvfp4-p6-b300-48xlarge
   export INSTANCE_TYPE=p6-b300.48xlarge
   export GPU_NODE=<your-p6-b300-node-name>
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

4. Deploy the vLLM inference server on `GPU_NODE` and confirm it serves the model.

   ```bash
   ./vllm-recipes/glm-5.2/nvfp4-p6-b300.48xlarge-deploy.sh
   ```

5. Generate the accuracy gate (.JSON) that the benchmark checks against.

   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service glm-52-b300-nvfp4 \
       --mode fingerprint \
       --model-tag glm-5.2 --config nvfp4 \
       --model-id glm-5.2-nvfp4 \
       --thinking off \
       --out-dir docs/results
   ```

6. Dry-run the exact in-cluster benchmark.

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

7. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).

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
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/glm-5.2/nvfp4-p6-b300.48xlarge.yaml
```

---


## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **`kubernetes.io/hostname`** Pins the pod to one node; the deploy script fills in `GPU_NODE` as it applies the manifest, so the file in git stays untouched. The pin does not preserve model data.
- **`fetch-weights` init container** Downloads `nvidia/GLM-5.2-NVFP4` from Hugging Face into `/models/glm-5.2-nvfp4`. It verifies at least 500 GiB is free, installs `huggingface_hub==1.28.0`, uses `/models/.hf-cache`, enables high-performance Xet with `HF_XET_HIGH_PERFORMANCE=1`, and verifies that every shard is complete before vLLM starts.

- **`weights: emptyDir: { sizeLimit: 800Gi }`** Stores both the checkpoint and Hugging Face cache in pod-local storage shared by the init container and vLLM. Kubernetes deletes this volume with the pod, so replacement or scale-to-zero discards all downloaded data and the new pod downloads it again. This recipe does not use a PVC and does not preserve downloads across pods; two observed cold downloads took about 2h15m each.

- **No local quantization step** The published checkpoint already contains Model Optimizer NVFP4 weights, and vLLM auto-detects that metadata without a `--quantization` flag.

- **`ARGS=(`** Build argv as an ARRAY. A '#' comment cannot follow a '\' line continuation: bash ends the command there and silently discards every remaining flag, and the server still starts successfully.


- **`--max-cudagraph-capture-size 512`** MUST be >= --max-num-seqs. A batch wider than the largest captured CUDA graph falls back to eager execution, which costs most of the graph speedup. Setting it below --max-num-seqs is the single most expensive misconfiguration in this recipe.

- **`startupProbe:`** The Hugging Face download happens in the init container, so this probe covers weight load and graph capture. Keep generous headroom: a restart creates a replacement pod with a new `emptyDir`, which forces the checkpoint and cache to be downloaded again.

- **Node-local `emptyDir`, not PVC or hostPath** The p6-b300 instance store backs the kubelet and containerd roots, so the pod's `emptyDir` lands there. `/mnt` is on the much smaller root volume on this node shape; do not change this recipe to `/mnt` hostPath without first verifying the actual NVMe mount with `df -h`. The current storage is intentionally ephemeral and never survives pod replacement.
