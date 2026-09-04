# Multi-node Qwen3-235B FP8 across 2x g7e.12xlarge with EFA

Serves [`Qwen/Qwen3-235B-A22B-FP8`](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8) on g7e.12xlarge (2x RTX PRO 6000 Blackwell, sm_120) with 2 GPUs per pod, tensor-parallel across two nodes over EFA.

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

Run the commands from the repository root. This capacity/EFA study also requires:

- AWS CLI, `eksctl`, `kubectl`, Helm, and Python configured on the workstation
- An existing EKS cluster in `us-east-1` and permission to create a two-instance On-Demand Capacity Reservation
- [The FSx CSI driver and a bound `fsx-pvc` claim](../../../k8s/examples/README.md#creating-the-weights-filesystem) in the same Availability Zone as the reservation
- [A copy of the EFA node group manifest](../../../k8s/examples/eksctl-nodegroup-gpu-efa.yaml) whose `subnets:` points at the AZ reserved in Quickstart step 3
- The EFA device plugin and LeaderWorkerSet controller, installed cluster-wide in Quickstart steps 7 and 8

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=us-east-1
   export EKS_CLUSTER_NAME="<your-cluster-name>"
   export HUGGING_FACE_HUB_TOKEN="<your-hf-token>"
   export NAMESPACE=vb-qwen3-235b-fp8-g7e-12xlarge-efa-study
   export INSTANCE_TYPE=g7e.12xlarge
   ```

2. Confirm the current on-demand rate. Pricing lives in the us-east-1 endpoint.

   ```bash
   aws pricing get-products --region "$AWS_REGION" --service-code AmazonEC2 \
     --filters "Type=TERM_MATCH,Field=instanceType,Value=$INSTANCE_TYPE" \
               "Type=TERM_MATCH,Field=location,Value=US East (N. Virginia)" \
               "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
               "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
               "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
     --query "PriceList" --output text | head
   ```

3. Create an open reservation for 2 instances, colocated with FSx in us-east-1c. Open match criteria lets the eksctl-launched instances slot into it automatically. If us-east-1c returns `InsufficientInstanceCapacity`, retry with `--availability-zone us-east-1a`.

   ```bash
   aws ec2 create-capacity-reservation \
     --region "$AWS_REGION" \
     --instance-type "$INSTANCE_TYPE" \
     --instance-platform Linux/UNIX \
     --availability-zone us-east-1c \
     --instance-count 2 \
     --instance-match-criteria open \
     --query "CapacityReservation.[CapacityReservationId,State,AvailabilityZone]" --output text
   ```

4. Confirm the reservation reached `active`, meaning the capacity is physically held, before launching nodes.

   ```bash
   aws ec2 describe-capacity-reservations --region "$AWS_REGION" \
     --filters "Name=instance-type,Values=$INSTANCE_TYPE" "Name=state,Values=active" \
     --query "CapacityReservations[].[CapacityReservationId,AvailabilityZone,TotalInstanceCount,AvailableInstanceCount,State]" \
     --output text
   ```

5. Launch the EFA node group. Point `subnets:` at the subnet in the same AZ as the reservation, because EFA requires every node in one AZ.

   ```bash
   eksctl create nodegroup -f my-nodegroup.yaml   # your copy of k8s/examples/eksctl-nodegroup-gpu-efa.yaml
   ```

6. Configure the EKS context and namespace, install the NVIDIA Kubernetes device plugin, and create the `hf-token` secret.

   ```bash
   ./scripts/install-prerequisites.sh
   ```

7. Install the EFA device plugin cluster-wide.

   ```bash
   helm repo add eks https://aws.github.io/eks-charts
   helm repo update eks
   helm upgrade --install aws-efa-k8s-device-plugin eks/aws-efa-k8s-device-plugin \
     --namespace kube-system --kube-context "$EKS_CLUSTER_NAME"
   ```

8. Install the LeaderWorkerSet controller cluster-wide.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" apply --server-side \
     -f https://github.com/kubernetes-sigs/lws/releases/latest/download/manifests.yaml
   ```

9. Stage `Qwen/Qwen3-235B-A22B-FP8` on the persistent `fsx-pvc`. A completed earlier Job and checkpoint make this a no-op on redeploy.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply \
     -f model-downloads/qwen3-235b-fp8.yaml
   kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
     --for=condition=complete job/download-qwen3-235b-fp8 --timeout=6h
   ```

10. Verify each node advertises 2 GPUs and 1 EFA interface. If GPUs show `0`, the node booted the closed NVIDIA driver.

    ```bash
    kubectl --context "$EKS_CLUSTER_NAME" get nodes -l role=g7e-efa \
      -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,EFA:.status.allocatable.vpc\.amazonaws\.com/efa
    ```

11. Deploy and validate the LeaderWorkerSet. The executable recipe applies the workload, discovers and waits for the leader, checks `/v1/models`, prints the live KV-cache configuration, and validates the serving manifest. Record its cache output with the result.

    ```bash
    ./vllm-recipes/qwen3-235b/fp8-g7e.12xlarge-lws-deploy.sh
    ```

12. Confirm NCCL will use EFA rather than TCP. Discover the controller-created leader Pod by its stable labels and run the check in its `vllm-leader` container. If the plugin is absent, NCCL falls back to TCP and the sweep measures the wrong transport.

    ```bash
    leader_pod="$(kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" get pods \
      -l 'app=qwen3-235b-lws,role=leader' \
      -o jsonpath='{.items[0].metadata.name}')"

    kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
      "pod/$leader_pod" --container vllm-leader -- \
      sh -c 'ls /opt/amazon/ofi-nccl/lib* 2>/dev/null; fi_info -p efa'
    ```

13. Run the benchmark.

    ```bash
    python3 app/bench/src/main.py launch \
        --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service qwen3-235b-lws \
        --protocol corpus --profile legacy_corpus \
        --instance g7e.12xlarge --model-tag qwen3-235b \
        --config 2node --quant fp8 \
        --model-id qwen3-235b \
        --tokenizer Qwen/Qwen3-235B-A22B \
        --reps 3 --out-dir docs/results
    ```

14. Sweep `NCCL_P2P_LEVEL` to isolate the PCIe-switch contribution from the host-memory-copy contribution. Set it in the `env:` block of the recipe, re-run Quickstart step 11 to apply and validate the LeaderWorkerSet and capture its cache output, then repeat Quickstart step 13 for each arm. Run 3 to 5 repeats per arm and compare means against spread, because the deltas fall within run-to-run noise.

    | Arm | `NCCL_P2P_LEVEL` | What it forces |
    |---|---|---|
    | default | (unset) | NCCL auto-selects transport |
    | PIX | `PIX` | P2P only across a single PCIe switch (SHM path) |
    | PHB | `PHB` | P2P across a PCIe host bridge |

## Tear down

Run these immediately after the benchmark, in order. Step 3 is what stops the hourly charge.

1. Delete the deployment.

   ```bash
   kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f vllm-recipes/qwen3-235b/fp8-g7e.12xlarge-lws.yaml
   ```

2. Delete the node group, or scale it to 0.

   ```bash
   eksctl delete nodegroup --cluster "$EKS_CLUSTER_NAME" --region us-east-1 --name g7e12-efa-ng
   ```

3. Cancel the capacity reservation.

   ```bash
   aws ec2 cancel-capacity-reservation --region us-east-1 --capacity-reservation-id "<cr-id>"
   ```

4. Confirm no active reservation remains.

   ```bash
   aws ec2 describe-capacity-reservations --region us-east-1 \
     --filters "Name=instance-type,Values=g7e.12xlarge" "Name=state,Values=active" \
     --query "CapacityReservations[].[CapacityReservationId,State]" --output text
   ```

---

## Settings

This is a separate flow from the single-node pipelines. It needs reserved capacity before a node group can launch, and it exists to measure an interconnect penalty rather than to serve well.

`g7e.12xlarge` has 1 EFA card at 400 Gbps and no NVLink between its two Blackwell GPUs. Splitting a 235B model's tensor-parallel all-reduce across two nodes over a single EFA card each is expected to be interconnect-bound, so the cross-node link sets the ceiling rather than GPU compute. The data point is intentional. It will not scale like a multi-rail NVLink box.

`g7e.12xlarge` on-demand capacity in us-east-1 is intermittent and has failed to launch in both offered AZs, us-east-1a and us-east-1c. The On-Demand Capacity Reservation in Quickstart step 3 guarantees the two instances exist before the node group tries to launch into them.

If Quickstart step 10 reports 0 GPUs, rebuild the open kernel module with [`k8s/examples/job-node-driver-fix.yaml`](../../../k8s/examples/job-node-driver-fix.yaml), or recreate the node group with `--ami-type AL2023_x86_64_NVIDIA`, which is the cleaner fix.

`launch` runs the harness inside the cluster over cluster DNS. Do not substitute a `kubectl port-forward` and a bare `main.py throughput`. That combination measures the tunnel rather than the server, and writes a file with no protocol lock, no accuracy gate and no medians. See [`docs/results/README.md`](../../results/README.md).
