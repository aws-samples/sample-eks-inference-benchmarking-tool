# Qwen3-30B-A3B FP8 on g7e.12xlarge, TP2 with expert parallel, NCCL default transport

Serves [`Qwen/Qwen3-30B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8) across both GPUs of one g7e.12xlarge (2x RTX PRO 6000 Blackwell, sm_120, PCIe, no NVLink) with tensor and expert parallelism, using [`fp8-g7e.12xlarge-switch-default.yaml`](../../../vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-default.yaml). It sets no NCCL transport override.

## Capacity sizing

Qwen3-30B-A3B is a mixture-of-experts model. Each of its 48 layers holds 128 experts and routes every token to 8 of them, so about 3.3B of its 30.5B parameters run per token. The FP8 checkpoint is about 31 GiB and would fit one 96 GB card; this recipe spans both GPUs because the study measures the transport between them. The KV cache is BF16 at 96 KiB per token across 48 layers, 48 KiB per GPU under TP2, so a full 8,192-token sequence costs 0.75 GiB.

`legacy_corpus` sends a median of about 17 wire input tokens and asks for up to 150 output tokens, so 512 concurrent sequences hold roughly 85k tokens, under 8 GiB of KV. `--max-num-seqs 512`, not KV memory, caps the batch for this workload. Read `KV cache size` in the result file's provenance block for the allocation in force.

| Property | Value | Source |
|---|---|---|
| **Parameters** | 30.5B total, 3.3B active per token | Model card |
| **Layers** | 48 | `config.json`, `num_hidden_layers` |
| **Experts** | 128 per layer, 8 routed per token | `config.json`, `num_experts`, `num_experts_per_tok` |
| **KV heads, head dimension** | 4, 128 | `config.json`, `num_key_value_heads`, `head_dim` |
| **KV cache per token, BF16** | 96 KiB, 48 KiB per GPU | `2 x 4 x 128 x 2 bytes x 48 layers`, halved by TP2 |
| **Checkpoint size, FP8** | ~31 GiB, 7 safetensors shards | `download-qwen3-30b-a3b-fp8` shard check |
| **Instance** | g7e.12xlarge, 2x RTX PRO 6000 Blackwell at 96 GB, PCIe, no NVLink | Instance type; `nvidia-smi topo -m` in the server log |
| **GPUs per pod** | 2 | `nvidia.com/gpu: "2"` |
| **Parallelism** | TP2 with expert parallel, `allgather_reducescatter` | `--tensor-parallel-size`, `--enable-expert-parallel`, `--all2all-backend` |
| **Weight precision** | FP8 E4M3, 128x128 blocks | Checkpoint `quantization_config`; the recipe passes no `--quantization` |
| **KV cache precision** | BF16 | `--kv-cache-dtype` |
| **NCCL transport** | NCCL default selection, no override | No `NCCL_P2P_LEVEL` or `NCCL_P2P_DISABLE` in `env` |
| **Context length** | 8,192 | `--max-model-len` |
| **Max sequences** | 512 | `--max-num-seqs` |
| **Batched tokens per step** | 8,192 | `--max-num-batched-tokens` |
| **Memory fraction** | 0.90 | `--gpu-memory-utilization` |
| **Image** | `vllm/vllm-openai:v0.26.0` | `image` |

## Before you begin

Create the following before you start.

- [Create a `g7e.12xlarge` node group](../../../k8s/examples/eksctl-nodegroup-gpu-spot.yaml) (set the instance type; the node must be idle)
- [Create a benchmark-client node group](../../../k8s/examples/eksctl-nodegroup-bench-client.yaml) (scale it to one node; the client Job does not tolerate the GPU taint)
- [Install the Amazon EBS CSI driver and a `gp3` StorageClass](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html) (the claim requests `storageClassName: gp3`)
- [Rebuild the NVIDIA open kernel module if the node advertises no GPUs](../../../k8s/examples/job-node-driver-fix.yaml)

## Quickstart

1. Set environment variables.
   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-qwen3-30b-a3b-fp8-g7e-12xlarge-switch
   export INSTANCE_TYPE=g7e.12xlarge
   export GPU_NODE=<your-g7e-12xlarge-node-name>
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

4. Download the published checkpoint from Hugging Face to the `g7e12-switch-model-storage` claim.
   ```bash
   ./model-jobs/qwen3-30b-a3b/fp8-model-download.sh
   ```

## Throughput test

1. Deploy the vLLM inference server on `GPU_NODE` and confirm it serves the model.
   ```bash
   ./vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-default-deploy.sh
   ```

2. Confirm no NCCL transport override is set and NCCL chose P2P, not SHM.
   ```bash
   SERVER_LOG=$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs \
       deployment/qwen3-30b-a3b-g7e12-switch-default -c vllm)
   grep -E 'NCCL transport override|^NCCL_P2P_(LEVEL|DISABLE)=' <<<"$SERVER_LOG"
   grep -cE 'P2P/direct pointer|via P2P/' <<<"$SERVER_LOG"
   grep -cE 'via SHM/' <<<"$SERVER_LOG"
   ```

3. Generate the accuracy gate (.JSON) that the benchmark checks against.
   ```bash
   eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service qwen3-30b-a3b-g7e12-switch-default \
       --mode fingerprint \
       --model-tag qwen3-30b-a3b --config fp8-switch-default \
       --model-id qwen3-30b-a3b-fp8 \
       --thinking off \
       --out-dir docs/results
   ```

4. Dry-run the exact in-cluster benchmark.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service qwen3-30b-a3b-g7e12-switch-default \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen3-30b-a3b \
       --config fp8-switch-default --quant fp8 \
       --model-id qwen3-30b-a3b-fp8 \
       --tokenizer Qwen/Qwen3-30B-A3B-FP8 \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=qwen3-30b-a3b-g7e12-switch-default -o jsonpath='{.items[0].metadata.name}')" \
       --out-dir docs/results \
       --reps 5 \
       --dry-run
   ```

5. Warm up the server with one unrecorded benchmark run, saved outside `docs/results`.
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service qwen3-30b-a3b-g7e12-switch-default \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen3-30b-a3b \
       --config fp8-switch-default --quant fp8 \
       --model-id qwen3-30b-a3b-fp8 \
       --tokenizer Qwen/Qwen3-30B-A3B-FP8 \
       --skip-accuracy --no-latency --no-provenance \
       --out-dir /tmp/qwen3-30b-a3b-switch-warmup/fp8-switch-default \
       --reps 1
   ```

6. Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags).
   ```bash
   eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" \
       --service qwen3-30b-a3b-g7e12-switch-default \
       --protocol corpus --profile legacy_corpus \
       --instance g7e.12xlarge --model-tag qwen3-30b-a3b \
       --config fp8-switch-default --quant fp8 \
       --model-id qwen3-30b-a3b-fp8 \
       --tokenizer Qwen/Qwen3-30B-A3B-FP8 \
       --pod "$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods \
           -l app=qwen3-30b-a3b-g7e12-switch-default -o jsonpath='{.items[0].metadata.name}')" \
       --out-dir docs/results \
       --reps 5
   ```

## Tear down

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
    -f vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-default.yaml --ignore-not-found
```

Once both arms are measured, delete the shared claim and scale the node groups to zero. Deleting the claim deletes its EBS volume and every checkpoint staged on it.

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete job download-qwen3-30b-a3b-fp8 --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete pvc g7e12-switch-model-storage --ignore-not-found
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name gpu-spot-1node \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
aws eks update-nodegroup-config --cluster-name "$EKS_CLUSTER_NAME" --nodegroup-name bench-client-cpu \
    --region "$AWS_REGION" --scaling-config minSize=0,maxSize=1,desiredSize=0
```

---

## Settings

- **`source .venv/bin/activate`** Puts `eks-bench` on your PATH in the current terminal only. If you open a new terminal, run it again from the repository root before continuing.
- **No NCCL transport override.** The recipe sets neither `NCCL_P2P_LEVEL` nor `NCCL_P2P_DISABLE`, so NCCL picks the transport itself. This arm is the negative control for the `PIX` arm: on a switch-local pair it should select P2P without being told to. Throughput step 2 checks the server log for P2P channels and no SHM channels; do not measure an arm that fails it.
- **Both arms pin the same physical node.** Use the same `GPU_NODE` in both runbooks. A Spot reclaim or replacement node has a new name and a new GPU pair; export its name as `GPU_NODE` and treat it as a different host.
- **The hostname is a placeholder in git.** The staging and deploy scripts substitute `GPU_NODE` as they apply each manifest, so the files in git stay untouched.
- **`--config fp8-switch-default`** The study's config tag, so the result lands in `qwen3-30b-a3b-fp8-g7e.12xlarge-fp8-switch-default-corpus-legacy_corpus.txt` beside the `fp8-switch-pix` arm.
- **The forced-SHM arm is not a recipe.** [`g7e12-switch-qwen3-30b-a3b-shm.yaml`](../../../experiments/g7e12-switch-qwen3-30b-a3b-shm.yaml) sets `NCCL_P2P_DISABLE=1` as a host-copy proxy and lives in `experiments/`. For the full three-arm study with its NCCL matrix, see the [PIX versus PHB-proxy study runbook](../studies/g7e.12xlarge-pix-vs-phb-proxy-runbook.md).
- **Two GPUs on purpose.** The checkpoint fits one card. TP2 with `--enable-expert-parallel` splits attention heads and places whole experts across the switch-local pair, and `--all2all-backend allgather_reducescatter` exchanges tokens between them, so every layer crosses the PCIe switch. That traffic is what the transport pair compares.
- **`--no-enable-prefix-caching`** The recipe passes it. `legacy_corpus` repeats a fixed prompt set, so a warm cache would measure cache hits rather than prefill. Confirm `Prefix cache hit` reads 0.0% in the provenance block before publishing any figure.
- **`--kv-cache-dtype bfloat16`** A 16-bit KV cache keeps KV quantization out of the comparison.
- **`--async-scheduling` and `--enable-chunked-prefill`** On in both arms and held constant between them.
- **`NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,NET`** Makes NCCL log each channel's transport, which Throughput step 2 and the provenance `NCCL transport` line read. The container also prints `nvidia-smi topo -m` at start; the GPU0 to GPU1 cell should read `PIX`.
- **`CUDA_DEVICE_ORDER=PCI_BUS_ID`** Keeps CUDA device numbering in the same order as `nvidia-smi`.
- **Thinking is disabled for the accuracy gate.** Qwen3 is hybrid-thinking, and with thinking on the gate's maths problems spend their budget on `<think>` and truncate. Throughput step 3 passes `--thinking off` to `capture-reference`; `measure` reads the mode back from the stored reference and rejects a `--thinking` flag of its own.
- **Benchmark requests carry no thinking override.** `legacy_corpus` posts chat requests capped at 150 output tokens, so a reasoning trace is truncated by the cap, as the protocol lock notes. Both arms share this.
- **One unrecorded warmup, then five recorded repetitions.** This is the study's protocol. The warmup skips the accuracy gate, latency pass and provenance, and writes outside `docs/results` so it is never mistaken for a result.
- **`--pod` names the server pod.** `measure` reads the image, KV pool and NCCL log from it. The two arms share the namespace, so name the pod explicitly.
- **One arm at a time.** Each Deployment requests both GPUs, so a second arm stays Pending until the first is deleted. `strategy: Recreate` stops a reapply from overlapping GPU claims.
- **Fixed arm order couples treatment to time.** The study ran default, then `PIX`, then SHM. Alternate the arms across fresh deployments for a stronger comparison, and keep such a rerun out of the original result files.
- **Weights come from a gp3 EBS claim, not node-local NVMe.** The recipe mounts `g7e12-switch-model-storage`, a 300 GiB `ReadWriteOnce` gp3 claim created by `model-jobs/qwen3-30b-a3b/fp8-model-download.yaml`. This departs from the repo's rule of staging on local NVMe and is kept because the recipe records what the study ran; every server start re-reads about 31 GiB from EBS.
- **The claim name matches the study's.** The study's shared staging manifest in `experiments/` creates the same `g7e12-switch-model-storage` claim and adds Llama 3.3 70B to it, so either manifest leaves a claim these recipes can mount. The download Job fails unless `config.json` and all 7 shards are present and no partial file remains.
- **No Hugging Face token is used.** Both checkpoints are public. `install-prerequisites.sh` still requires `HUGGING_FACE_HUB_TOKEN` and creates an `hf-token` secret that nothing here mounts.
- **Closed-loop only.** This runbook measures closed-loop concurrency. It says nothing about SLA-safe open-loop arrival rates; do not compare its figures to one.
