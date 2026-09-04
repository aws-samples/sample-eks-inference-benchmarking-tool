# Qwen3-235B-A22B FP8, lws on g7e.12xlarge

Serves [`Qwen/Qwen3-235B-A22B-FP8`](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8) on g7e.12xlarge (2x RTX PRO 6000 Blackwell, sm_120) with 2 GPUs per pod.

## Capacity sizing

Qwen3-235B-A22B-FP8 at FP8 occupies 222.6 GiB of the 357.6 GiB on 4 RTX PRO 6000 Blackwell cards, leaving 91.8 GiB for KV cache. At the recipe's 8,192 token context that pool holds 62 concurrent sequences.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 235B total, 22B active | Model card |
| **Layers** | 94 | `config.json`, `num_hidden_layers` |
| **KV heads, head dimension** | 4, 128 | `config.json` |
| **Checkpoint size, FP8** | 222.6 GiB | Model card safetensors metadata |
| **Instance** | g7e.12xlarge x2, 4x RTX PRO 6000 Blackwell at 96 GB | 384 GB across 2 nodes, 357.6 GiB |
| **Tensor parallel size** | 2 | `TP_SIZE` |
| **Pipeline parallel size** | 2 | `PP_SIZE` |
| **Weight precision** | FP8 | declared by the checkpoint's `quantization_config` |
| **KV cache precision** | 16-bit default | No `--kv-cache-dtype`, see Settings |
| **Context length** | 8,192 | `MAX_MODEL_LEN` |
| **Memory fraction** | 0.90 | `GPU_MEM_UTIL` |
| **Max sequences** | 512 | `MAX_NUM_SEQS` |

KV cache per layer per token is `2 x kv_heads x head_dim x bytes_per_element`, which is 2 KiB here. Every layer attends to the full context, so per-sequence cost scales linearly with context length across all 94 layers. The 91.8 GiB pool is 357.6 GiB at 0.90 utilization, less 222.6 GiB of weights, less roughly 7.5 GiB for activations and CUDA graphs.

## Before you begin

Create the following in the namespace you deploy into. Secrets and ConfigMaps are namespaced.

- [Create a `g7e.12xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-efa.yaml)
- Label the nodes `role=g7e-efa`
- [Install the EFA device plugin (`vpc.amazonaws.com/efa: 1`)](../../../k8s/examples/eksctl-nodegroup-gpu-efa.yaml)

  ```bash
  helm repo add eks https://aws.github.io/eks-charts
  helm repo update
  helm install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin -n kube-system
  ```

- Install the LeaderWorkerSet controller

  ```bash
  kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/latest/download/manifests.yaml
  ```

- [Install the FSx CSI driver and create the `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) (nothing mounts without the driver)
- [Create the `vllm-sitecustomize` ConfigMap](../../../k8s/examples/configmap-vllm-sitecustomize.yaml)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen3-235b-fp8-g7e-12xlarge-lws
   export INSTANCE_TYPE=g7e.12xlarge
   ```

2. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

3. Confirm the EFA nodes, shared FSx storage, LeaderWorkerSet controller, and site customization exist.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" get nodes -l role=g7e-efa
   kubectl --context "$EKS_CLUSTER_NAME" get nodes -l role=g7e-efa -o custom-columns='NAME:.metadata.name,EFA:.status.allocatable.vpc\.amazonaws\.com/efa'
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc fsx-pvc
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get secret hf-token
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get configmap vllm-sitecustomize
   kubectl --context "$EKS_CLUSTER_NAME" get crd leaderworkersets.leaderworkerset.x-k8s.io
   ```

4. Stage `Qwen/Qwen3-235B-A22B-FP8` on the persistent `fsx-pvc`. A completed earlier Job and checkpoint make this a no-op on redeploy.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f model-downloads/qwen3-235b-fp8.yaml
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait --for=condition=complete job/download-qwen3-235b-fp8 --timeout=6h
   ```

5. Deploy and validate the model server. The helper discovers the controller-created leader pod by stable labels and waits for its `/health` readiness probe.

   ```bash
   ./vllm-recipes/qwen3-235b/fp8-g7e.12xlarge-lws-deploy.sh
   ```

6. Generate the accuracy gate (.JSON) required for the next step.

   ```bash
   python3 app/cli.py launch --capture-accuracy-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen3-235b-lws \
       --mode fingerprint \
       --model-tag qwen3-235b --config lws \
       --model-id qwen3-235b \
       --thinking off \
       --out-dir docs/results
   ```

7. Dry-run the in-cluster benchmark. This validates and renders the Job without creating cluster resources.

   ```bash
   python3 app/cli.py launch \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen3-235b-lws \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen3-235b \
       --config lws --quant fp8 \
       --model-id qwen3-235b \
       --tokenizer Qwen/Qwen3-235B-A22B \
       --thinking off \
       --reps 3 --out-dir docs/results \
       --dry-run
   ```

8. Benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`.

   ```bash
   python3 app/cli.py launch \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen3-235b-lws \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen3-235b \
       --config lws --quant fp8 \
       --model-id qwen3-235b \
       --tokenizer Qwen/Qwen3-235B-A22B \
       --thinking off \
       --reps 3 --out-dir docs/results
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/qwen3-235b/fp8-g7e.12xlarge-lws.yaml
```

## Settings

- **`/usr/local/bin/python -m pip install -q "$RAY_SPEC"`** The vLLM DLC image doesn't ship Ray (dropped as a hard dep), but multi-node `vllm serve` needs it. Install into the vLLM env (/usr/local); pin a floor so leader+worker resolve to the same latest (Ray requires matching versions).

- **`ray start --head --port=6379 --disable-usage-stats`** Ray head on the leader

- **`echo "waiting for 4 GPUs to register in Ray..."`** Wait until BOTH nodes' GPUs (4 total) have joined the Ray cluster

- **`/usr/local/bin/python -m pip install -q "$RAY_SPEC"`** Match the leader: install Ray into the vLLM env before joining.

- **`until ray start --address=${LWS_LEADER_ADDRESS}:6379 --block; do`** Join the leader's Ray cluster and block (LWS injects LWS_LEADER_ADDRESS)

- **`apiVersion: v1`** API service -> leader pod only. `python3 app/cli.py launch` reaches it over cluster DNS; do not port-forward it for a measurement.
- **`size: 2`** 1 leader + 1 worker = 2 pods = 2 nodes
- **`role: g7e-efa`** from the EFA node group labels
- **`add: ["IPC_LOCK"]`** required by EFA/libfabric
- **`VLLM_ATTENTION_BACKEND=FLASHINFER`** Blackwell sm_120
- **`NCCL_DEBUG=INFO`** confirm EFA transport is actually selected
