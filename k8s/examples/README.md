# Kubernetes examples

These are examples rather than runnable manifests. Every identifier in them is a placeholder. Copy one, substitute your own values, and apply it.

These manifests exist because the benchmark harness assumes a cluster with GPU nodes, storage for model weights, and a few supporting resources, none of which ships with the repo. The examples cover the most commonly needed Kubernetes manifests.

## What to substitute

Placeholders are structurally valid, so a manifest parses and AWS validation returns a useful error. They are also clearly synthetic, so an accidentally applied manifest cannot reference a real account.

The namespace is not a placeholder. No example sets `metadata.namespace`, matching the recipes in [`vllm-recipes/`](../../vllm-recipes/). Pass it at apply time:

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f k8s/examples/<file>.yaml
```

A manifest that pinned its own namespace would either apply to an unintended namespace or be rejected when the `-n` value disagreed with it.

| Placeholder | Meaning |
|---|---|
| `012345678901` | AWS account. Twelve digits, since a shorter value fails validation with an error that does not identify the cause. |
| `us-east-1` | Region. A valid region so manifests parse. Change it to where your GPU capacity is. |
| `my-cluster` | EKS cluster name. |
| `fsx-pvc` | The PVC holding model weights. |
| `subnet-0123456789abcdef0` | Subnet. Suffix is 17 hex characters. |
| `sg-0123456789abcdef0` | Security group. |
| `fs-0123456789abcdef0` | FSx for Lustre filesystem. |
| `my-model-weights` | S3 bucket for staged weights. |
| `my-repo:my-tag` | ECR repository and tag. |

Model ids like `Qwen/Qwen2.5-7B-Instruct` are real values and are left as-is, so an example works as written once the infrastructure values are substituted.

## The examples

| File | Kind | Purpose |
|---|---|---|
| `eksctl-cluster.yaml` | ClusterConfig | EKS cluster with the add-ons a GPU benchmark needs, and a CPU node group only |
| `eksctl-nodegroup-gpu-spot.yaml` | ClusterConfig | Single-node Spot GPU group with node-local weights, several AZs |
| `persistentvolume-fsx-lustre.yaml` | PersistentVolume + PersistentVolumeClaim | Bind an FSx for Lustre filesystem and claim it as `fsx-pvc` |
| `configmap-vllm-sitecustomize.yaml` | ConfigMap | The runtime patch every `vllm-recipes/` manifest mounts |
| `job-build-image-kaniko.yaml` | ConfigMap + Job | Build a derived serving image in-cluster and push to ECR |
| `job-node-driver-fix.yaml` | Job | Rebuild the NVIDIA open kernel module when GPUs do not register |
| `pod-diagnostic-probe.yaml` | Pod | The throwaway-pod pattern for answering one question about a cluster |
| `pod-profile-nsys.yaml` | Pod | Run vLLM under Nsight Systems to see where step time goes |

Serving manifests are not in this directory. They live in [`vllm-recipes/`](../../vllm-recipes/), one per published result, with the flags that produced it.

## Notes per example

Each manifest carries a one-line header stating what it is and when to use it, plus its `SUBSTITUTE:` list. The remaining explanation is here, which keeps the manifests readable and the documentation in one place.

### `eksctl-cluster.yaml`

This cluster installs Karpenter at creation, so GPU capacity comes from a Karpenter node pool later, not from a node group here. OIDC is required for any add-on that uses IRSA, including the FSx and EBS CSI drivers. The CPU node group serves the controllers, the add-ons, and any weight-download or image-build Job, so size its disk for that work rather than to the minimum.

The file uses `CLUSTER_NAME` and `AWS_REGION` placeholders. `scripts/create-cluster.sh` fills them from `EKS_CLUSTER_NAME` and `AWS_REGION` and runs eksctl, creating the cluster only when it does not already exist, so create it with the script rather than a hand-edited copy:

```bash
export AWS_REGION=us-east-2
export EKS_CLUSTER_NAME=my-cluster
./scripts/create-cluster.sh
```

eksctl selects the availability zones. If your GPU capacity is confined to one zone, add an `availabilityZones` list that includes it to the file before running, since the control plane spans only the zones it is given and a zone missing here cannot be used later.

### Discover GPU capacity before creating node groups

Instance availability is specific to both type and Availability Zone. Check offerings before choosing subnets or creating same-AZ storage:

```bash
export AWS_REGION="us-east-1"
export INSTANCE_TYPE="g7e.12xlarge"

aws ec2 describe-instance-type-offerings \
  --region "$AWS_REGION" \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
  --query 'InstanceTypeOfferings[].Location' --output text
```

For Spot node groups, placement scores provide an additional capacity signal. They are not a reservation or guarantee:

```bash
aws ec2 get-spot-placement-scores \
  --region "$AWS_REGION" \
  --instance-types "$INSTANCE_TYPE" \
  --target-capacity 1 \
  --region-names "$AWS_REGION"
```

Use only zones that the cluster has subnets in. Multi-node EFA and FSx-backed workloads normally narrow this further to one zone.

### `eksctl-nodegroup-gpu-spot.yaml`

List every zone where the instance type is offered and the cluster has a subnet, so Spot has somewhere to choose from.

Size the volume for the checkpoint plus the image plus the Hugging Face cache, and set maximum throughput and IOPS for gp3, because model load re-reads everything on every start. Spot costs roughly a third of on-demand, with the tradeoff that a two-minute reclaim terminates the repetition in flight and requires re-downloading the weights: workable for one run, less so for a long sweep. Scaling to zero discards the checkpoint, so use the FSx PV where that matters. Confirm instance availability in each cluster subnet before creating the node group, then create it from a substituted copy:

```bash
eksctl create nodegroup -f my-spot-nodegroup.yaml   # copy of k8s/examples/eksctl-nodegroup-gpu-spot.yaml
```

### Creating the weights filesystem

Covers [`persistentvolume-fsx-lustre.yaml`](persistentvolume-fsx-lustre.yaml).

Every recipe that mounts `fsx-pvc` needs three things: an FSx for Lustre filesystem, the FSx CSI driver, and a security group that permits Lustre traffic. You create the filesystem with the AWS CLI, then bind it with a PersistentVolume.

No StorageClass is involved. The claim sets `storageClassName: ""` and `volumeName`, which binds it to the named PV and nothing else. Setting the field empty is not the same as omitting it: an absent field means the cluster's default StorageClass applies, and on EKS that default is usually EBS.

Read the subnet and security group from the cluster rather than from the console. Use the subnet in the zone that step 2 reports.

```bash
aws eks describe-cluster --name "$EKS_CLUSTER_NAME" \
  --query 'cluster.resourcesVpcConfig.[clusterSecurityGroupId,subnetIds]'
```

Using the cluster security group for the filesystem makes step 3 a self-referencing rule rather than a second security group to create and manage.

#### 1. Install the FSx CSI driver

Without the driver, nothing mounts: the PVC stays `Pending` and pods stay `ContainerCreating` with no clear cause. It is not an EKS managed addon, so install the upstream driver.

```bash
helm repo add aws-fsx-csi-driver https://kubernetes-sigs.github.io/aws-fsx-csi-driver
helm repo update
helm upgrade --install aws-fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver -n kube-system
```

Confirm the driver registered:

```bash
kubectl get csidrivers fsx.csi.aws.com
kubectl -n kube-system get deploy fsx-csi-controller
```

#### 2. Choose the zone before creating anything

FSx for Lustre is single-AZ and a pod cannot mount a filesystem in another zone. GPU capacity determines the zone, so provision the GPU nodes first, then read the zone from them.

```bash
kubectl get nodes -l node.kubernetes.io/instance-type=g6e.12xlarge \
  -o jsonpath='{range .items[*]}{.metadata.name}{"  "}{.metadata.labels.topology\.kubernetes\.io/zone}{"\n"}{end}'
```

Use the subnet in that zone for the next step.

#### 3. Open the Lustre ports

The filesystem's security group must accept inbound TCP 988 and TCP 1018 through 1023 from the nodes' security group. Without those rules the mount hangs and the pod sits in `ContainerCreating` until it times out, which presents as a driver fault rather than a network one.

```bash
aws ec2 authorize-security-group-ingress --group-id sg-FSX --protocol tcp --port 988       --source-group sg-NODES
aws ec2 authorize-security-group-ingress --group-id sg-FSX --protocol tcp --port 1018-1023 --source-group sg-NODES
```

#### 4. Create the filesystem and bind it

Create the filesystem, then read back the values the PV needs. `PERSISTENT_2` accepts 1200 GiB or multiples of 2400 GiB. Creation takes roughly 10 minutes.

```bash
aws fsx create-file-system \
  --file-system-type LUSTRE \
  --file-system-type-version 2.15 \
  --storage-capacity 1200 \
  --subnet-ids subnet-0123456789abcdef0 \
  --security-group-ids sg-0123456789abcdef0 \
  --lustre-configuration DeploymentType=PERSISTENT_2,PerUnitStorageThroughput=250 \
  --tags Key=Name,Value=vllm-model-weights \
  --query "FileSystem.[FileSystemId,Lifecycle]" --output text

aws fsx describe-file-systems --file-system-id fs-0123456789abcdef0 \
  --query "FileSystems[0].[DNSName,LustreConfiguration.MountName,StorageCapacity,Lifecycle,SubnetIds[0]]" \
  --output text
```

The second command returns, in order, the `dnsname`, the `mountname`, the size for both `storage:` fields, the lifecycle state, and the subnet whose zone belongs in `nodeAffinity`. A mount name is eight characters, for example `4ufilamv`. Wait for `AVAILABLE` before applying.

```bash
kubectl -n "$NAMESPACE" apply -f persistentvolume-fsx-lustre.yaml
```

Some driver versions want `id::mountname` in `volumeHandle` rather than the filesystem ID alone. If the PVC will not bind, try that form.

`persistentVolumeReclaimPolicy: Retain` keeps the filesystem when the PVC is deleted, so a multi-terabyte checkpoint survives a teardown. Billing also continues, so delete the filesystem explicitly when the work is finished.

#### 5. Verify the claim binds and a pod can mount it

Binding alone does not confirm the mount works, because the network path is only exercised when a pod attaches.

```bash
kubectl -n "$NAMESPACE" get pvc fsx-pvc
kubectl -n "$NAMESPACE" apply -f pod-diagnostic-probe.yaml
kubectl -n "$NAMESPACE" logs diagnostic-probe
kubectl -n "$NAMESPACE" delete pod diagnostic-probe
```

Expect `STATUS Bound` from the claim and a `df -h /models` line from the probe. A probe stuck in `ContainerCreating` points at step 1 or step 3.

Both manifests use `ReadWriteMany`, which multi-node serving requires and which lets a download Job and a serving pod share the weights.

### `configmap-vllm-sitecustomize.yaml`

huggingface_hub's `validate_repo_id` rejects an absolute path before checking whether it is a local directory, which breaks serving a checkpoint from a PVC. The patch makes the validator a no-op for paths beginning `/`. The GLM and Kimi images do not need it.

It is mounted with `subPath` over the interpreter's own site-packages path, so the Python version in that path must match the image; otherwise it mounts a file that nothing imports.

Create it from the tracked copy so the two stay in sync:

```bash
kubectl create configmap vllm-sitecustomize --from-file=sitecustomize.py=k8s/examples/sitecustomize.py
```

### `job-build-image-kaniko.yaml`

Building in-cluster avoids pulling a large base image to a local machine, which is often a different architecture; a vLLM DLC is roughly 25 GB. kaniko needs no Docker daemon, so no privileged mode and no GPU.

`backoffLimit` is 0, so the build log can be read before repeating a long build. Layer caching in the registry makes a second build of a large base considerably faster. A large base needs substantial disk and memory to unpack and push.

The node's IAM role must be able to push, either through the imageBuilder add-on policy that the eksctl examples set or through a repository policy.

### `job-node-driver-fix.yaml` and `daemonset-nvidia-open-driver.yaml`

Blackwell parts support only the open NVIDIA kernel module flavor, and the EKS-optimized AL2023 NVIDIA AMI ships the closed one, so the node boots, joins, reports `Ready` and advertises `nvidia.com/gpu: 0` indefinitely. Present on AMI release 1.33.13-20260801 (AL2023 2023.12.20260727, kernel 6.12.94, kmod-nvidia-latest-dkms-580.159.03). Re-check each AMI release rather than assuming this is still needed.

Confirm the cause first with `pod-diagnostic-probe.yaml`. Three unrelated problems leave a node with zero allocatable GPUs, and Kubernetes state does not distinguish between them. These manifests address only the second:

1. `nvidia-smi -L` lists the GPUs, so the driver is working and the device plugin is the problem. These manifests do not apply.
2. `nvidia-smi` reports "No devices found" and dmesg carries `requires use of the NVIDIA open kernel modules` and `RmInitAdapter failed!`. This is the case they address.
3. `nvidia-smi` is absent entirely, so the node group was built on a non-GPU AMI. Recreate it with `--ami-type AL2023_x86_64_NVIDIA` rather than patching individual nodes.

Both run privileged, build against the host kernel, and reboot the node, so apply them only to nodes holding no workloads. Choose between them by lifetime: the Job repairs one node and leaves no standing resource, while the DaemonSet keeps every node that joins repaired, which matters because the built module lives on the instance's root volume and is lost on a Spot reclaim or a scale-from-zero. Where the launch template is under your control, user data is preferable to either, since it runs before workloads exist and needs no reboot.

The Job does not select a node on its own. A `nodeSelector` on an instance type is not a node name, so set `nodeName` to the intended node, and cordon it first on a shared cluster:

```yaml
nodeName: ip-10-0-0-1.us-east-1.compute.internal
```

The script is kept as a ConfigMap file rather than inline shell in the container args, because it runs inside `chroot` and nesting it in quotes expanded variables in the wrong shell. The DaemonSet re-checks on a timer rather than running once and sleeping, and `heal.sh` is idempotent, records one success per kernel, and stops after a bounded number of failures. The DaemonSet is scheduled ahead of ordinary workloads and tolerates the taints a GPU node group carries, and it narrows by instance type rather than by `nvidia.com/gpu.present`, so it does not land on hardware whose driver variant is already correct. Both size their container for the build rather than for idle, because parallel compilers need substantial memory and a low ceiling causes an OOM kill reported as `gcc: fatal error: Killed signal terminated program cc1`, which resembles a source error but is not one.

`NVIDIA_OPEN_VERSION` defaults to the driver version the AMI ships, which is the recommended value, because a newer open package removes the `/usr/src` tree that carries the kernel-open sources. Set it only after confirming that version's sources are present:

```yaml
env:
  - name: NVIDIA_OPEN_VERSION
    value: "580.167.08"
```

Two further notes apply before editing them. The build cannot be replaced with `dkms build`: on kernel 6.12 it runs its hardcoded fallback target `make -j2 module` instead of the `modules` target `dkms.conf` declares, and fails. State lives in `/var/lib/nvidia-open-fix`, holding a per-kernel success marker, a failure counter, and a copy of the driver sources, so deleting the marker allows a node to retry. The two scripts are intentional copies so each manifest applies independently; changes should be made to both.

### `pod-diagnostic-probe.yaml`

A GPU probe has to be scheduled onto a GPU node and occupies it while it runs, so drop the `nodeSelector` and the toleration for a storage or credentials probe. Match the image to the question, using the serving image itself when inspecting what it ships.

Commands worth running in the probe:

```bash
mount works:      echo ok > /models/.probe && cat /models/.probe && rm /models/.probe
space and layout: df -h /models && ls -la /models
GPUs visible:     nvidia-smi --query-gpu=index,name,memory.total --format=csv
GPU topology:     nvidia-smi topo -m
NCCL version:     python3 -c "import torch; print(torch.cuda.nccl.version())"
EFA present:      fi_info -p efa && ls /opt/amazon/ofi-nccl/lib*
credentials:      aws sts get-caller-identity && aws s3 ls s3://my-model-weights
flag surface:     vllm serve --help=all | grep -- --some-flag
```

### `pod-profile-nsys.yaml`

nsys has to be the parent of `vllm serve`, because tracing cannot attach to a running process, so this is a separate pod rather than an edit to a recipe. It needs an image that already carries nsys, which the vLLM DLC does not; build one with `job-build-image-kaniko.yaml`.

Use a tensor-parallel size of 2 or more, otherwise there is no collective to measure. The capture window has to land on steady-state decode with load arriving during it, or the trace is empty. For driving that load, see the Profiling with Nsight Systems section of the [root README](../../README.md).

NVTX makes the collective visible, because NCCL emits ranges. The command avoids `exec` deliberately: nsys stops the target when the window closes, and under `exec` the shell would be replaced, leaving nothing to print the tables afterwards. Those tables are the durable output, since the `.nsys-rep` is large and only needed for the GUI timeline. The loop avoids pipes because SIGPIPE plus `pipefail` would terminate it before it finishes.

The startup probe must exceed model load plus warmup, so raise it substantially for a large checkpoint. Per-request INFO logging becomes a bottleneck under load and affects what is measured, so it stays off. Without `SYS_ADMIN` the capture still works but drops OS-runtime sampling. Pinned memory is for the NCCL transport. Size the scratch volume for several traces: a 90-second window on a busy server produced a 132 MB trace.
