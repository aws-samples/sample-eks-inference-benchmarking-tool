# G7e PIX versus PHB-proxy study on g7e.12xlarge

This runbook reproduces the two-model study that compared switch-local direct P2P (`PIX`) with a no-direct-peer proxy (`NCCL_P2P_DISABLE=1`, NCCL `SHM`) on the same two GPUs. It runs Llama 3.3 70B FP8 with TP2 and Qwen3-30B-A3B FP8 with TP2+EP2.

> **Scope:** the forced-SHM arm is a conservative software proxy for a switchless/PHB product question, not a physical PHB topology. A PHB instance type still requires measurement on its actual motherboard, root-complex, NUMA, and PCIe configuration.

The retained manifests are cleaned, functionally equivalent versions of the executed configurations. The post-run cleanup removed only a non-functional `TREATMENT` log label and converted YAML flow mappings to block style; model, image, resources, vLLM arguments, NCCL controls, storage, placement, and Kubernetes resource names are unchanged.

## Capacity sizing

| Resource | Requirement |
|---|---|
| GPU server | One idle `g7e.12xlarge`, 2x 96 GB RTX PRO 6000 Blackwell |
| Llama checkpoint | `nvidia/Llama-3.3-70B-Instruct-FP8`, 15 shards, about 68 GiB |
| Qwen checkpoint | `Qwen/Qwen3-30B-A3B-FP8`, 7 shards, about 31 GiB |
| Study storage | New 300 GiB gp3 RWO PVC created by this study |
| Benchmark client | Dedicated `m5.4xlarge` or larger, at least 8 allocatable CPUs |
| Serving | TP2, BF16 KV cache, 8,192-token context, 512 max sequences |
| Workload | `corpus` / `legacy_corpus`, 150 output tokens, concurrency 1–512 |

## Before you begin

1. Run from the repository root with AWS, `kubectl`, Helm, Python 3, and `sha256sum` available. Setup step 1 runs `scripts/install-prerequisites.sh`, which creates `$NAMESPACE`, installs the pinned NVIDIA device plugin and creates an `hf-token` secret.
2. Use an idle GPU node. Do not repurpose a node that is running another workload.
3. The eight executed manifests pin `kubernetes.io/hostname` to the placeholder `ip-10-0-0-1.us-east-1.compute.internal`. Every arm must land on the **same** physical node, or the comparison measures two different hosts rather than two transports. Substitute your own node name identically across all eight files before applying anything, and treat them as one versioned set. The namespace is not pinned in the manifests; it comes from `-n "$NAMESPACE"`.
4. The storage manifest creates `g7e12-switch-model-storage`. 
5. The client node group incurs EC2 cost. It must inherit the cluster's working system-node launch template/security groups; an ad-hoc launch template can produce a node with broken pod DNS.
6. Preserve every artifact and checksum before deleting the PVC. The PVC contains both staged checkpoints and raw NCCL logs.

## Setup

### 1. Configure the environment and record the manifest revision

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -e '.[tokenizer]'

export AWS_REGION=<your-cluster-region>
export EKS_CLUSTER_NAME=<your-cluster-name>
export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
export NAMESPACE=vb-g7e-12xlarge-switch-study
export INSTANCE_TYPE=g7e.12xlarge
export GPU_NODE=<your-gpu-node-name>
export OTHER_PROTECTED_GPU_NODE=<your-second-gpu-node-name>
export CLIENT_NODEGROUP=g7e12-switch-client-ondemand
export CLIENT_ROLE=g7e12-switch-client
export ARTIFACT_ROOT=artifacts/g7e12-switch
mkdir -p "$ARTIFACT_ROOT"

./scripts/install-prerequisites.sh

kubectl --context "$EKS_CLUSTER_NAME" wait node "$GPU_NODE" \
  --for=condition=Ready \
  --timeout=5m
kubectl --context "$EKS_CLUSTER_NAME" get node "$GPU_NODE" \
  -o 'custom-columns=NODE:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPU:.status.allocatable.nvidia\.com/gpu'

git rev-parse HEAD > "$ARTIFACT_ROOT/git-revision.txt"
MANIFESTS=(
  experiments/g7e12-switch-study-storage-and-model-stage.yaml
  experiments/g7e12-nccl-p2p-shm-matrix.yaml
  experiments/g7e12-switch-llama3.3-70b-default.yaml
  experiments/g7e12-switch-llama3.3-70b-pix.yaml
  experiments/g7e12-switch-llama3.3-70b-shm.yaml
  vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-default.yaml
  vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-pix.yaml
  experiments/g7e12-switch-qwen3-30b-a3b-shm.yaml
)
sha256sum "${MANIFESTS[@]}" > "$ARTIFACT_ROOT/manifest-sha256.txt"
```

Confirm the target node has two idle GPUs and record all pre-existing workloads. The second command must show no workload that you are unwilling to interrupt.

```bash
kubectl --context "$EKS_CLUSTER_NAME" get node "$GPU_NODE" \
  -o custom-columns=NODE:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPU:.status.allocatable.nvidia\.com/gpu
kubectl --context "$EKS_CLUSTER_NAME" get pods -A \
  --field-selector "spec.nodeName=$GPU_NODE" -o wide \
  | tee "$ARTIFACT_ROOT/preexisting-target-node-pods.txt"
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get deploy,service,job,pod,pvc \
  -o wide | tee "$ARTIFACT_ROOT/preexisting-namespace-resources.txt"
```

### 2. Create the dedicated benchmark-client node group

Use the same launch template as a working system node group so networking, security groups, bootstrap, and DNS match the cluster. Set `CLIENT_SUBNET_ID` to a subnet in the GPU node's Availability Zone.

```bash
export SYSTEM_NODEGROUP=system
export CLIENT_SUBNET_ID=<same-AZ-subnet-id>
export NODE_ROLE=$(aws eks describe-nodegroup \
  --region "$AWS_REGION" --cluster-name "$EKS_CLUSTER_NAME" \
  --nodegroup-name "$SYSTEM_NODEGROUP" --query 'nodegroup.nodeRole' --output text)
export LT_ID=$(aws eks describe-nodegroup \
  --region "$AWS_REGION" --cluster-name "$EKS_CLUSTER_NAME" \
  --nodegroup-name "$SYSTEM_NODEGROUP" --query 'nodegroup.launchTemplate.id' --output text)
export LT_VERSION=$(aws eks describe-nodegroup \
  --region "$AWS_REGION" --cluster-name "$EKS_CLUSTER_NAME" \
  --nodegroup-name "$SYSTEM_NODEGROUP" --query 'nodegroup.launchTemplate.version' --output text)

aws eks create-nodegroup \
  --region "$AWS_REGION" \
  --cluster-name "$EKS_CLUSTER_NAME" \
  --nodegroup-name "$CLIENT_NODEGROUP" \
  --subnets "$CLIENT_SUBNET_ID" \
  --node-role "$NODE_ROLE" \
  --instance-types m5.4xlarge \
  --capacity-type ON_DEMAND \
  --scaling-config minSize=0,maxSize=1,desiredSize=1 \
  --labels role="$CLIENT_ROLE",topology-study=switch-client \
  --tags Project=vllm-switch-study,Temporary=true \
  --launch-template id="$LT_ID",version="$LT_VERSION"
aws eks wait nodegroup-active \
  --region "$AWS_REGION" --cluster-name "$EKS_CLUSTER_NAME" \
  --nodegroup-name "$CLIENT_NODEGROUP"

kubectl --context "$EKS_CLUSTER_NAME" get nodes -l "role=$CLIENT_ROLE" -o wide
```

The isolation helper temporarily cordons both GPU nodes until each `bench-suite-*` runner binds, verifies the runner node's `role` label, and always uncordons in `finally`.

```bash
isolated_launch() {
  .venv/bin/python scripts/run_isolated_launch.py \
    --context "$EKS_CLUSTER_NAME" \
    --namespace "$NAMESPACE" \
    --gpu-node "$GPU_NODE" \
    --gpu-node "$OTHER_PROTECTED_GPU_NODE" \
    --expected-role "$CLIENT_ROLE" \
    -- "$@"
}
```

### 3. Generate and retain the tokenizer records

These two JSON files are required by the six result protocol locks and must remain under `docs/prompt_tokens/`.

```bash
.venv/bin/eks-bench verify-prompts \
  --model nvidia/Llama-3.3-70B-Instruct-FP8 \
  --profile legacy_corpus
.venv/bin/eks-bench verify-prompts \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --profile legacy_corpus

test -s docs/prompt_tokens/nvidia__Llama-3.3-70B-Instruct-FP8.json
test -s docs/prompt_tokens/Qwen__Qwen3-30B-A3B-FP8.json
cp docs/prompt_tokens/nvidia__Llama-3.3-70B-Instruct-FP8.json "$ARTIFACT_ROOT/"
cp docs/prompt_tokens/Qwen__Qwen3-30B-A3B-FP8.json "$ARTIFACT_ROOT/"
```

### 4. Create only the study PVC and stage both checkpoints

Refuse to continue if the claim already exists. This prevents a rerun from adopting a pre-existing shared PVC.

```bash
if kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" \
  get pvc g7e12-switch-model-storage >/dev/null 2>&1; then
  echo "REFUSING: g7e12-switch-model-storage already exists" >&2
  return 1 2>/dev/null || exit 1
fi

kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply \
  -f experiments/g7e12-switch-study-storage-and-model-stage.yaml
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
  --for=condition=complete job/g7e12-switch-model-stage --timeout=7200s
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs \
  job/g7e12-switch-model-stage | tee "$ARTIFACT_ROOT/model-stage.log"

kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc \
  g7e12-switch-model-storage -o yaml > "$ARTIFACT_ROOT/pvc-created.yaml"
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc \
  g7e12-switch-model-storage -o jsonpath='{.metadata.uid}' \
  > "$ARTIFACT_ROOT/pvc-created.uid"
```

### 5. Run and preserve the balanced NCCL matrix as text

The Job runs default, explicit PIX, and forced SHM in a balanced order over five repeats, four collectives, and 60 raw logs. It asserts the physical pair is PIX and verifies P2P or SHM transport in every log.

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply \
  -f experiments/g7e12-nccl-p2p-shm-matrix.yaml
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
  --for=condition=complete job/g7e12-nccl-p2p-shm-matrix --timeout=7200s
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs \
  job/g7e12-nccl-p2p-shm-matrix | tee "$ARTIFACT_ROOT/nccl-job.log"

NCCL_POD=$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pod \
  -l job-name=g7e12-nccl-p2p-shm-matrix \
  -o jsonpath='{.items[0].metadata.name}')
NCCL_RAW_DIR="$ARTIFACT_ROOT/g7e12-switch-nccl-raw"
mkdir -p "$NCCL_RAW_DIR"
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" cp \
  "$NCCL_POD:/models/results/g7e12-switch/nccl/." "$NCCL_RAW_DIR"

test "$(find "$NCCL_RAW_DIR" -maxdepth 1 -name '*.txt' | wc -l | tr -d ' ')" = 60
while read -r digest source; do
  printf '%s  %s\n' "$digest" "$NCCL_RAW_DIR/$(basename "$source")"
done < "$NCCL_RAW_DIR/SHA256SUMS" | sha256sum -c -

python3 scripts/flatten_nccl_logs.py \
  "$NCCL_RAW_DIR" \
  docs/results/g7e.12xlarge-nccl-p2p-shm-raw.txt
(
  cd docs/results
  sha256sum g7e.12xlarge-nccl-p2p-shm-raw.txt \
    > g7e.12xlarge-nccl-p2p-shm-raw-checksum.txt
  sha256sum -c g7e.12xlarge-nccl-p2p-shm-raw-checksum.txt
)
```

## Quickstart

Run one deployment at a time in this recorded order. The default arm is a negative control: it must agree with explicit PIX and select P2P without an override.

1. Confirm kubectl points at the study cluster. These arms continue in the same shell as `Setup`, which set the variables and the `isolated_launch` function.

   ```bash
   aws eks update-kubeconfig \
     --name "$EKS_CLUSTER_NAME" \
     --region "$AWS_REGION" \
     --alias "$EKS_CLUSTER_NAME"

   kubectl config current-context
   ```

2. Deploy and measure the six arms in this order.

| Order | Manifest | Deployment / Service | Model tag | Config | Model ID | Tokenizer | Extra |
|---:|---|---|---|---|---|---|---|
| 1 | `experiments/g7e12-switch-llama3.3-70b-default.yaml` | `llama33-70b-g7e12-switch-default` | `llama3.3-70b` | `fp8-switch-default` | `llama3.3-70b-fp8` | `nvidia/Llama-3.3-70B-Instruct-FP8` | none |
| 2 | `experiments/g7e12-switch-llama3.3-70b-pix.yaml` | `llama33-70b-g7e12-switch-pix` | `llama3.3-70b` | `fp8-switch-pix` | `llama3.3-70b-fp8` | same | none |
| 3 | `experiments/g7e12-switch-llama3.3-70b-shm.yaml` | `llama33-70b-g7e12-switch-shm` | `llama3.3-70b` | `fp8-switch-shm` | `llama3.3-70b-fp8` | same | none |
| 4 | `vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-default.yaml` | `qwen3-30b-a3b-g7e12-switch-default` | `qwen3-30b-a3b` | `fp8-switch-default` | `qwen3-30b-a3b-fp8` | `Qwen/Qwen3-30B-A3B-FP8` | `--thinking off` |
| 5 | `vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-pix.yaml` | `qwen3-30b-a3b-g7e12-switch-pix` | `qwen3-30b-a3b` | `fp8-switch-pix` | `qwen3-30b-a3b-fp8` | same | `--thinking off` |
| 6 | `experiments/g7e12-switch-qwen3-30b-a3b-shm.yaml` | `qwen3-30b-a3b-g7e12-switch-shm` | `qwen3-30b-a3b` | `fp8-switch-shm` | `qwen3-30b-a3b-fp8` | same | `--thinking off` |

For each row, set the variables and run the function below. `RECIPE` is the manifest path from the repository root; only the arms under `vllm-recipes/` go through `validate-manifests`. Use `THINKING_ARGS=()` for Llama and `THINKING_ARGS=(--thinking off)` for Qwen. Only `capture-reference` takes it; `measure` reads the mode from the stored reference.

```bash
run_arm() {
  local recipe_path="$RECIPE"
  local arm_artifacts="$ARTIFACT_ROOT/$MODEL_TAG/$CONFIG"
  local warmup_dir="/tmp/g7e12-switch-warmup/$MODEL_TAG/$CONFIG"
  mkdir -p "$arm_artifacts" "$warmup_dir"

  case "$RECIPE" in
    vllm-recipes/*) .venv/bin/eks-bench validate-manifests "${RECIPE#vllm-recipes/}" ;;
  esac
  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" apply -f "$recipe_path"
  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" rollout status \
    "deployment/$DEPLOYMENT" --timeout=3600s

  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec \
    "deployment/$DEPLOYMENT" -c vllm -- curl -s localhost:8000/v1/models \
    | tee "$arm_artifacts/models.json"
  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec \
    "deployment/$DEPLOYMENT" -c vllm -- \
    sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'" \
    | tee "$arm_artifacts/cache-config.txt"

  isolated_launch .venv/bin/eks-bench capture-reference \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service "$SERVICE" \
    --mode fingerprint --model-tag "$MODEL_TAG" --config "$CONFIG" \
    --model-id "$MODEL_ID" --out-dir docs/results "${THINKING_ARGS[@]}" \
    | tee "$arm_artifacts/accuracy-runner.log"

  isolated_launch .venv/bin/eks-bench measure \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service "$SERVICE" \
    --protocol corpus --profile legacy_corpus --instance g7e.12xlarge \
    --model-tag "$MODEL_TAG" --config "$CONFIG" --quant fp8 \
    --model-id "$MODEL_ID" --tokenizer "$TOKENIZER" \
    --skip-accuracy --no-latency --no-provenance \
    --out-dir "$warmup_dir" \
    --reps 1 \
    | tee "$arm_artifacts/warmup-runner.log"

  .venv/bin/eks-bench measure \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service "$SERVICE" \
    --protocol corpus --profile legacy_corpus --instance g7e.12xlarge \
    --model-tag "$MODEL_TAG" --config "$CONFIG" --quant fp8 \
    --model-id "$MODEL_ID" --tokenizer "$TOKENIZER" \
    --out-dir docs/results \
    --reps 5 \
    --dry-run \
    > "$arm_artifacts/measured-dry-run.txt"

  isolated_launch .venv/bin/eks-bench measure \
    --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service "$SERVICE" \
    --protocol corpus --profile legacy_corpus --instance g7e.12xlarge \
    --model-tag "$MODEL_TAG" --config "$CONFIG" --quant fp8 \
    --model-id "$MODEL_ID" --tokenizer "$TOKENIZER" \
    --out-dir docs/results \
    --reps 5 \
    | tee "$arm_artifacts/measured-runner.log"

  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get deployment \
    "$DEPLOYMENT" -o yaml > "$arm_artifacts/deployment-snapshot.yaml"
  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pod \
    -l "app=$DEPLOYMENT" -o wide > "$arm_artifacts/serving-placement.txt"
  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" logs \
    "deployment/$DEPLOYMENT" -c vllm > "$arm_artifacts/server.log"
  cp "docs/accuracy/$MODEL_TAG-$CONFIG.json" "$arm_artifacts/"

  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f "$recipe_path"
  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
    --for=delete "deployment/$DEPLOYMENT" --timeout=600s
}
```

Example for the first row:

```bash
export RECIPE=experiments/g7e12-switch-llama3.3-70b-default.yaml
export DEPLOYMENT=llama33-70b-g7e12-switch-default
export SERVICE=llama33-70b-g7e12-switch-default
export MODEL_TAG=llama3.3-70b
export CONFIG=fp8-switch-default
export MODEL_ID=llama3.3-70b-fp8
export TOKENIZER=nvidia/Llama-3.3-70B-Instruct-FP8
THINKING_ARGS=()
run_arm
```

Repeat with the remaining five table rows. Do not overlap arms: each Deployment requests both GPUs.

The original report used fixed default → PIX → SHM blocks with five repetitions inside each arm. That reproduces the published data but couples treatment to time and deployment lifecycle. A stronger follow-up should rotate treatments across fresh deployments and retain the same placement/utilization evidence; do not merge such a follow-up into the original result files.

## Tear down

### 1. Verify all durable evidence before deleting storage

```bash
(
  cd docs/results
  sha256sum -c g7e.12xlarge-nccl-p2p-shm-raw-checksum.txt
)
test -s docs/prompt_tokens/nvidia__Llama-3.3-70B-Instruct-FP8.json
test -s docs/prompt_tokens/Qwen__Qwen3-30B-A3B-FP8.json
test "$(find docs/results -maxdepth 1 -name '*g7e.12xlarge*switch*-corpus-legacy_corpus.txt' | wc -l | tr -d ' ')" = 6
```

Delete any remaining study Deployments/Services and Jobs by their explicit names. Do not use broad namespace selectors.

```bash
for recipe in \
  experiments/g7e12-switch-llama3.3-70b-default.yaml \
  experiments/g7e12-switch-llama3.3-70b-pix.yaml \
  experiments/g7e12-switch-llama3.3-70b-shm.yaml \
  vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-default.yaml \
  vllm-recipes/qwen3-30b-a3b/fp8-g7e.12xlarge-switch-pix.yaml \
  experiments/g7e12-switch-qwen3-30b-a3b-shm.yaml; do
  kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete -f "$recipe" --ignore-not-found=true
done
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete job \
  g7e12-nccl-p2p-shm-matrix g7e12-switch-model-stage --ignore-not-found=true
```

### 2. Delete only the PVC created by this run

Compare the live UID with the UID captured immediately after creation. Refuse deletion on any mismatch.

```bash
RECORDED_PVC_UID=$(cat "$ARTIFACT_ROOT/pvc-created.uid")
CURRENT_PVC_UID=$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc \
  g7e12-switch-model-storage -o jsonpath='{.metadata.uid}')
test -n "$RECORDED_PVC_UID"
test "$CURRENT_PVC_UID" = "$RECORDED_PVC_UID" || {
  echo "REFUSING: PVC UID does not match this run" >&2
  return 1 2>/dev/null || exit 1
}

PV_NAME=$(kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pvc \
  g7e12-switch-model-storage -o jsonpath='{.spec.volumeName}')
VOLUME_ID=$(kubectl --context "$EKS_CLUSTER_NAME" get pv "$PV_NAME" \
  -o jsonpath='{.spec.csi.volumeHandle}')
RECLAIM_POLICY=$(kubectl --context "$EKS_CLUSTER_NAME" get pv "$PV_NAME" \
  -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')
test "$RECLAIM_POLICY" = Delete

kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete pvc \
  g7e12-switch-model-storage --wait=true --timeout=180s
aws ec2 wait volume-deleted --region "$AWS_REGION" --volume-ids "$VOLUME_ID"
```

### 3. Delete only the tagged temporary client node group

```bash
TEMP_TAG=$(aws eks describe-nodegroup \
  --region "$AWS_REGION" --cluster-name "$EKS_CLUSTER_NAME" \
  --nodegroup-name "$CLIENT_NODEGROUP" \
  --query 'nodegroup.tags.Temporary' --output text)
PROJECT_TAG=$(aws eks describe-nodegroup \
  --region "$AWS_REGION" --cluster-name "$EKS_CLUSTER_NAME" \
  --nodegroup-name "$CLIENT_NODEGROUP" \
  --query 'nodegroup.tags.Project' --output text)
test "$TEMP_TAG" = true
test "$PROJECT_TAG" = vllm-switch-study

aws eks delete-nodegroup \
  --region "$AWS_REGION" --cluster-name "$EKS_CLUSTER_NAME" \
  --nodegroup-name "$CLIENT_NODEGROUP"
aws eks wait nodegroup-deleted \
  --region "$AWS_REGION" --cluster-name "$EKS_CLUSTER_NAME" \
  --nodegroup-name "$CLIENT_NODEGROUP"
```

Finally, verify the protected GPU nodes are Ready and schedulable and that no named study resource remains:

```bash
kubectl --context "$EKS_CLUSTER_NAME" uncordon "$GPU_NODE" "$OTHER_PROTECTED_GPU_NODE"
kubectl --context "$EKS_CLUSTER_NAME" get nodes \
  "$GPU_NODE" "$OTHER_PROTECTED_GPU_NODE"
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" \
  get deploy,service,job,pod,pvc -o wide
```

## Settings

- **Default control:** neither `NCCL_P2P_LEVEL` nor `NCCL_P2P_DISABLE` is set; NCCL must select P2P.
- **PIX:** `NCCL_P2P_LEVEL=PIX`; logs must show `P2P/direct pointer` or `P2P/CUMEM`.
- **PHB proxy:** `NCCL_P2P_DISABLE=1`; logs must show `SHM/direct/direct`. This is not physical PHB.
- **Llama:** FP8 weights, BF16 KV, TP2.
- **Qwen:** FP8 weights, BF16 KV, TP2+EP2, `allgather_reducescatter`, thinking disabled.
- **Prompt shape:** `legacy_corpus`, 150-token output cap; wire input medians are 44 tokens for Llama and 17 for Qwen.
- **Cache:** prefix caching disabled; retained provenance must report zero cache hits.
- **Measured run:** one separate unrecorded one-repetition throughput warmup, then five recorded repetitions with the companion streaming-latency pass.
