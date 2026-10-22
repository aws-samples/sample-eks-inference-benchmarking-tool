# Project rules

## Always

* Commit every checkpoint-staging Job to `model-jobs/<model-name>/`, named `<quant>-model-download.yaml` for published weights or `<quant>-model-quantize.yaml` for weights this repo produces. Use the same model directory name as `vllm-recipes/` and `docs/runbooks/`.
* Commit any manifest you apply to a cluster. A cluster is not source control, and an ad-hoc `kubectl apply` leaves nothing in git to recover.
* Disable prefix caching in every benchmark (`--no-enable-prefix-caching`), no matter what production runs. It roughly doubled throughput on `chat_terse` and only the uncached number is honest.
* Use the placeholder account (`012345678901`), a region placeholder, and a `SUBSTITUTE:` header line in every committed manifest.
* Make a tool fail on its own bad output. A quantize Job that exits 0 having written an invalid checkpoint is worse than one that crashes, because the next reader inherits it silently.
* Put a verification check outside the artifact it verifies when the artifact could itself be stale. A verifier that only runs inside the producing image cannot catch a producing image that is wrong.
* Report end-to-end latency alongside TTFT and TPOT. A p99 sum is not a summed p99, so the two cannot reconstruct it, and end-to-end is the SLA for any caller that does not render a partial answer.
* Measure open-loop arrival rates when the question is capacity. Closed-loop concurrency is a load-generator dial, and it understated SLA-safe capacity by 26% here.
* Prove a feature is active before attributing a result to it. The n-gram deploy scripts fail if no `vllm:spec_decode` metrics appear, because a silently ignored config reads as "the feature didn't help".
* Run speculative decoding as its own labelled config, never folded into a baseline.
* Capture speculation acceptance as a delta between two counter snapshots. `vllm:spec_decode_*` are cumulative and never reset.
* Verify version-specific claims against the running image rather than inheriting them from another model's runbook. Async scheduling and speculation combine fine on 0.29, even though an earlier runbook said they conflict.
* Prefer a flag over an environment variable for anything that must reach an in-cluster Job. Exported variables do not survive the trip.
* Re-read the KV pool size after changing engine features. Enabling speculation cut it from ~135k to ~72k tokens.
* Pass `--pod` explicitly whenever more than one vLLM pod shares the namespace.
* Keep runbook claims in sync with the recipes they describe, and cite the field each one comes from.
* Write every instruction a runbook needs into that runbook. Link to a file as a reference, never as a substitute for the steps. Cross-references also hide divergence: the FP8 runbook inherited a RAID0 launch template from the BF16 one that its recipes do not need.
* Keep a runbook to its established sections and nothing else: `Capacity sizing`, `Before you begin`, `Quickstart`, `Throughput test`, `Latency test`, `Tear down`, `Settings`. Compare against a sibling runbook before adding a heading.
* Make each Quickstart step one sentence followed by its command, around 10 to 15 words, saying what the step does and what it acts on, such as the claim, bucket or server. Do not cut it so far that the reader cannot tell what will happen. Skip conditions, GPU counts and caveats belong in `Settings` as bullets, never in the step or as prose between the steps.
* Write each `Before you begin` bullet as one line: a bulleted markdown link whose text is an imperative phrase and whose target is the relative path, with at most a short parenthetical caveat. The linked file carries the explanation; the bullet does not.
* Split a function once it needs a second `try`/`except` or a second conditional variable, and compose small named helpers instead. Each check in `verify_checkpoint.py` is its own function returning `(problems, notes)`, and `verify()` only concatenates them, so a new check is a new function rather than another branch.
* Write each deploy-script step as a comment describing it, an `echo`, then the command. Use no `if`, loops, or guard blocks: `set -euo pipefail` already stops the script when a step fails, so a check is a plain command such as `kubectl logs ... | grep "Using AttentionBackendEnum.FLASHINFER backend"`. A long-running command may run in the background, but keep it to one tidy line.
* Open every runbook's Quickstart with the same three steps: export the variables the scripts read, install the `eks-bench` CLI into `.venv` with `pip install -e '.[tokenizer]'`, then run `./scripts/install-prerequisites.sh`.
* Make Quickstart step 4 `./scripts/setup-pod-identity.sh <model>-bench` in every runbook whose recipe reads S3 through Pod Identity. Never list the Pod Identity Agent or the association as `Before you begin` bullets.
* Create an AWS resource in a script with one `describe || create` line, so a rerun keeps what exists and still exits 0. This is the only guard a script may use.
* Test a script that creates AWS resources with throwaway names before a runbook relies on it. Run it twice, prove the access from a pod, prove a different resource is refused, then delete everything it created.
* Stage every checkpoint with the script beside its Job, `model-jobs/<model>/<job>.sh`, written like a deploy script. It applies the Job, waits for it, and fails on its own check, so the runbook step is one command with no separate "confirm" step.
* Mount model storage at `/models` in every Job and recipe, whether a claim or an `emptyDir` backs it. Download into `MODEL_DIR=/models/<prefix>`, where `<prefix>` is the S3 prefix or checkpoint name, and set `HF_HOME=/models/.hf-cache`. Name the volume `models`.
* Write every checkpoint this repo quantizes to `/models/quantized/<checkpoint-name>`, and serve it from that same path. The quantize, verify and claim-to-S3 Jobs and the recipes all read it there.
* Share one FSx claim, `fsx-pvc`, across models, each in its own `MODEL_DIR`. A claim binds to one PersistentVolume and one filesystem, so a claim per model would mean a filesystem per model. Size the filesystem for every checkpoint on it plus headroom, because a nearly full Lustre filesystem can hang reads and writes for every model. Run one large download or load against it at a time, because all models split its throughput. Keep GPU nodes in the filesystem's Availability Zone; nodes in other zones cannot mount it.
* Share `/models/.hf-cache` freely. On an `emptyDir` it is private to the pod; on a claim every model shares it, which is safe because each model downloads into its own `MODEL_DIR` and the cache only holds locks and download chunks. Keep the `flock` mount option on the FSx PersistentVolume so those locks hold across nodes.
* Give every model its own S3 bucket, `<model>-model-weights-<account-id>-<region>`, in the cluster's `AWS_REGION`. `<model>` is the `vllm-recipes/` directory name with any `.` replaced by `-`, because a dot in a bucket name breaks HTTPS addressing. The staging script creates the bucket with one `head-bucket || mb` line, and every script works the name out from `aws sts get-caller-identity` and `AWS_REGION`, so runbooks never ask for a bucket or a bucket region. Write the bucket in a committed manifest as `<model>-model-weights-012345678901-us-east-1`, which the script's `sed` replaces. Keep one prefix per checkpoint inside the bucket, such as `kimi-k3-mxfp4/` and `kimi-k3-dspark/`.
* Give every container that downloads from Hugging Face the token the same way: `HUGGING_FACE_HUB_TOKEN` from Secret `hf-token`, key `token`, never `optional`. Every runbook's Quickstart step 3 runs `./scripts/install-prerequisites.sh`, which creates that Secret, so the Secret always exists. A runbook mentions the token only when the repository is gated and the token's account must accept a license.
* Fill a manifest's placeholders as the script applies it, `sed ... | kubectl apply -f -`, from the variables Quickstart step 1 exports. Never edit a tracked manifest in place with `sed -i`; the files in git stay untouched.
* Serve every ModelOpt checkpoint this repo produces with `--quantization modelopt`. The plain `fp8` loader misreads a ModelOpt export, and on a BF16 checkpoint it silently quantizes at load time instead of failing.
* The quantizer image is a tool, not the model: it carries ModelOpt, torch and `quantize.py`, and building or pushing it quantizes nothing. The model is quantized only when that image runs — a standalone quantize Job, or a serving pod's init container that writes the FP8 checkpoint to NVMe before vLLM starts. Word the build step so no one reads image-build as model-quantization, and name the model in the step that actually quantizes it. For a small single-node model prefer the init-container form, which needs no `fsx-pvc` claim or S3 copy to persist.
* Tear down GPU deployments and scale the nodegroup to zero as soon as measurement stops.
* Provision GPU capacity through Karpenter, not eksctl node groups. A runbook's cluster prerequisite runs `scripts/create-cluster.sh`, which fills the `CLUSTER_NAME` and `AWS_REGION` placeholders in `k8s/examples/eksctl-cluster.yaml` (cluster plus Karpenter) from `EKS_CLUSTER_NAME` and `AWS_REGION` and runs eksctl, creating the cluster only when `eksctl get cluster` does not already find it. Quickstart then runs `scripts/provision-capacity.sh <instance-type>` for on-demand, or `scripts/provision-capacity-block.sh <instance-type> <reservation-id>` when the instance type is Capacity Block only (p5, p5en, p6-b200). Both apply a NodeClass with `instanceStorePolicy: RAID0` so `/models` lands on NVMe, and both default the node role and discovery tag to the cluster name, overridable with `KARPENTER_NODE_ROLE` and `KARPENTER_DISCOVERY`. Tear down deletes the recipe, then the node pool and node class; Karpenter scales the node to zero.
* Stage model weights for serving on the node's local NVMe instance store: the recipe's `fetch-weights` init container syncs them from S3 into an `emptyDir` at `/models`. Never stage them on EBS, neither the root volume nor an EBS claim, and never create a per-model claim. The only claim is the shared `fsx-pvc`, used to download, quantize and verify checkpoints before they go to S3. Four Muse-Glimmer copies writing 132 GiB to a 125 MB/s gp3 root disk took ~18 minutes to land and stalled `kubectl exec` for the whole window. Serve from FSx or S3 only for checkpoints the size of GLM 5.2 or Kimi K3. On AL2023 managed nodegroups the instance store stays unmounted unless the launch template sets nodeadm `localStorage.strategy: RAID0`; Karpenter's `instanceStorePolicy: RAID0` does the same. Confirm `df /models` shows the NVMe array before trusting a staging step.

## Never

* Never commit measured results or accuracy gates. Both are gitignored (`.gitignore:43` and `:65`) and belong nowhere in git, including inside markdown.
* Never commit a real account ID, cluster name, namespace, or personal alias.
* Never treat an empty git-history search as proof something never existed. Untracked files leave no trace; check the live cluster and the filesystem too.
* Never rely on a human reading logs as the only guard against a silent failure. If the correct outcome and the broken one both exit 0, that is the bug.
* Never attribute output divergence to a change while the baseline is not bit-reproducible. The FP8 baseline diverges from itself, which made the speculation losslessness check unanswerable.
* Never trust the two-pass reproducibility probe. It runs both passes under identical load and has already mis-certified this model.
* Never promote a workaround into the canonical template. The `/app/src` overlay exists because ECR push was unavailable; rebuilding the image is the real fix.
* Never compare closed-loop concurrency figures against open-loop arrival rates, or ITL across configs with different `--stream-interval`.
* Never carry a null result from a short-prompt profile to a long-prompt one. `--max-num-batched-tokens` was immaterial at 96 tokens and will not be at 8k.
* Never claim a benchmark answers a question it did not measure. N-gram on generic chat says nothing about n-gram on autocomplete, where the output echoes the prompt.
* Never start a long or costly run on your own after asking for a go/no-go. Wait for the answer.
* Never report a diff before ruling out an extraction artifact. A jsonpath key containing dots silently produced empty files and three bogus mismatches.
* Never write comments that record why something changed, compare a file with another, or restate obvious logic. "Identical to X except", "Treat results as their own labelled config" and "calibration is not bit-reproducible, so..." belong in a commit message or nowhere. A comment says what a non-obvious setting does.

## Runbook step wording

Use these sentences for runbook steps, word for word, in the order the steps occur. Fill in the claim name or node where the template shows one. A step with no template here follows the same pattern: say what the step does and what it acts on, in about 10 to 15 words.

| Step | Sentence |
|---|---|
| Variables | Set environment variables. |
| CLI | Install the `eks-bench` CLI into a Python virtual environment. |
| Prerequisites | Configure the EKS context and namespace, install the NVIDIA device plugin, create the `hf-token` secret, and grant ECR push permissions. |
| Provision capacity | Provision a GPU node pool with NVMe RAID0 for the instance type. |
| Provision capacity, Capacity Block | Provision a Capacity Block GPU node pool with NVMe RAID0 for the instance type. |
| Pod Identity | Install the Pod Identity Agent and grant the `<model>-bench` ServiceAccount S3 access. |
| Download to a claim | Download the published checkpoint from Hugging Face to the `<claim>` claim. |
| Download to S3 | Copy the published checkpoint from Hugging Face to your S3 bucket. |
| Download a draft to S3 | Copy the published DSpark draft model from Hugging Face to your S3 bucket. |
| Quantizer image | Build the quantizer tool image (the toolbox, not the model) in-cluster with kaniko, which creates the ECR repository and pushes it. Takes several minutes; to build locally on x86 Docker instead, run `./scripts/push-image.sh`. |
| Quantize | Quantize the model to ModelOpt FP8 and save the checkpoint to the `fsx-pvc` claim. |
| Quantize and verify | Quantize the model to ModelOpt FP8 on the `fsx-pvc` claim, then verify the checkpoint. |
| Claim to S3 | Copy the quantized checkpoint from the claim to your S3 bucket. |
| Deploy | Deploy the inference server and confirm it serves the model. |
| Deploy, pinned | Deploy the inference server on `GPU_NODE` and confirm it serves the model. |
| Deploy, replicas | Deploy eight vLLM inference server replicas and confirm they serve the model. |
| NVMe check | Confirm `/models` is on the NVMe RAID0 array, not the EBS root volume. |
| KV cache check | Confirm the KV cache is FP8 and record the KV pool size. |
| NCCL check | Confirm no NCCL transport override is set and NCCL chose P2P, not SHM. |
| NCCL check, pinned level | Confirm `NCCL_P2P_LEVEL=PIX` is set and NCCL chose P2P, not SHM. |
| Accuracy gate | Generate the accuracy gate (.JSON) that the benchmark checks against. |
| Dry run | Dry-run the exact in-cluster benchmark. |
| Benchmark | Run the benchmark in-cluster. If needed, repeat per protocol, changing `--protocol` and `--profile`. For CLI configurations, see `` [`docs/guides/REFERENCE.md`](../../guides/REFERENCE.md#measure-flags) ``. |
| Optional pair | (Optional) Dry-run the NVIDIA AIPerf cross-client calibration. / (Optional) Run the NVIDIA AIPerf cross-client calibration. |
| Optional pair | (Optional) Dry-run the open-loop latency test. / (Optional) Run the open-loop latency test. |

## Script templates

Write every deploy and staging script from these templates, step for step. Add a step only when the recipe needs it and a sibling script already has it, such as the EFA check or the `vllm-sitecustomize` ConfigMap check. Drop a step only when the recipe lacks the thing it acts on: no ServiceAccount step when the recipe sets no `serviceAccountName`, no backend step when the recipe pins no `--attention-backend`. Leave every ServiceAccount plain; EKS Pod Identity binds its role, granted by `setup-pod-identity.sh`. Pin a node only when the runbook needs one physical host. Write the pin as `nodeSelector` `kubernetes.io/hostname: SUBSTITUTE-gpu-node`, and fill it in the deploy script's `sed` with `-e "s|SUBSTITUTE-gpu-node|${GPU_NODE}|"`.

Deploy script, `vllm-recipes/<model>/<recipe>-deploy.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="<model>-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Apply the <model>-bench ServiceAccount the server runs as.
echo "==> Applying ServiceAccount <model>-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount <model>-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Deploy the <Model> server with the weights bucket and its region filled in.
echo "==> Deploying <Model> <QUANT> on <instance>"
sed -e "s|value: <model>-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: \"${AWS_REGION}\"|;}" \
  "$SCRIPT_DIR/<recipe>.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the weights to stage and the server to start.
echo "==> Waiting for the <Model> server"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" rollout status \
  deployment/<deployment> \
  --timeout=<seconds>s
# Confirm the expected model is served.
echo "==> Checking the served model"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/<deployment> \
  --container vllm \
  -- curl -s -w '\n' localhost:8000/v1/models
# Record the live KV-cache allocation.
echo "==> Reading the KV-cache configuration"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" exec \
  deployment/<deployment> \
  --container vllm \
  -- sh -c "curl -s localhost:8000/metrics | grep 'vllm:cache_config_info'"
# Print the attention backend the server selected.
echo "==> Confirming the attention backend is <BACKEND>"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" logs \
  deployment/<deployment> \
  --container vllm \
  | grep "Using AttentionBackendEnum.<BACKEND> backend"
# Validate the serving manifest.
echo "==> Validating the <Model> serving manifest"
eks-bench validate-manifests \
  <model>/<recipe>.yaml
```

Staging script, `model-jobs/<model>/<quant>-model-download.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${NAMESPACE:?Set NAMESPACE before running this script}"
: "${AWS_REGION:?Set AWS_REGION before running this script}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_BUCKET="<model>-model-weights-$(aws sts get-caller-identity --query Account --output text)-${AWS_REGION}"
# Create the model's weights bucket if it does not exist.
echo "==> Creating bucket $WEIGHTS_BUCKET"
aws s3api head-bucket --bucket "$WEIGHTS_BUCKET" >/dev/null 2>&1 \
  || aws s3 mb "s3://$WEIGHTS_BUCKET" --region "$AWS_REGION"
# Apply the <model>-bench ServiceAccount the staging Job runs as.
echo "==> Applying ServiceAccount <model>-bench"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" create serviceaccount <model>-bench \
  --dry-run=client -o yaml \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Start the Job that stages the published checkpoint from Hugging Face to S3.
echo "==> Staging the <Model> <QUANT> checkpoint to s3://$WEIGHTS_BUCKET/<prefix>/"
sed -e "s|value: <model>-model-weights-012345678901-us-east-1|value: ${WEIGHTS_BUCKET}|" \
  -e "/name: AWS_REGION/{n;s|value: .*|value: ${AWS_REGION}|;}" \
  "$SCRIPT_DIR/<quant>-model-download.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" apply -f -
# Wait for the Job, which fails on a shard-count or size mismatch.
echo "==> Waiting for the staging Job"
kubectl --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" wait \
  --for=condition=complete job/<job> \
  --timeout=24h
# Confirm S3 holds all <N> shards.
echo "==> Checking the shard count in S3"
aws s3 ls "s3://${WEIGHTS_BUCKET}/<prefix>/" --region "$AWS_REGION" \
  | grep -c '\.safetensors$' | grep -x <N>
```

Setup script, `scripts/<verb>-<thing>.sh`, for anything that creates AWS resources. `scripts/setup-pod-identity.sh` is the reference:

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
: "${<VARIABLE>:?Set <VARIABLE> before running this script}"
<NAME>=${1:?Pass the <thing> name, for example <example>}
# Create the <resource> if it does not exist.
echo "==> Creating <resource> <name>"
aws <service> describe-<resource> <identifying flags> >/dev/null 2>&1 \
  || aws <service> create-<resource> <identifying flags> <settings> >/dev/null
# Wait for the <resource> to become active.
echo "==> Waiting for <resource> <name>"
aws <service> wait <resource>-active <identifying flags>
# Set the <resource>'s <setting>, which is idempotent.
echo "==> Setting <setting> on <name>"
aws <service> put-<setting> <identifying flags> <settings>
# Show the <resource> in force.
echo "==> Showing <resource> <name>"
aws <service> describe-<resource> <identifying flags> --query <field> --output text | grep "<expected>"
```

When a `list-` call stands in for `describe-`, it returns `None` instead of failing, so test it with `| grep -v None` before the `||`.

Capacity script, `scripts/provision-capacity.sh` (and `provision-capacity-block.sh`). Fills the matching Karpenter manifest in `k8s/examples/` and applies it; it purchases no AWS resource:

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?Set AWS_REGION before running this script}"
: "${EKS_CLUSTER_NAME:?Set EKS_CLUSTER_NAME before running this script}"
INSTANCE_TYPE=${1:?Pass the instance type, for example p6-b200.48xlarge}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NODE_NAME="bench-${INSTANCE_TYPE//./-}"
NODE_ROLE="${KARPENTER_NODE_ROLE:-eksctl-KarpenterNodeRole-${EKS_CLUSTER_NAME}}"
DISCOVERY="${KARPENTER_DISCOVERY:-${EKS_CLUSTER_NAME}}"
# One node's GPU count caps the pool at a single benchmark node.
GPU_LIMIT=$(aws ec2 describe-instance-types --instance-types "$INSTANCE_TYPE" --region "$AWS_REGION" \
  --query 'InstanceTypes[0].GpuInfo.Gpus[0].Count' --output text)
# Apply the EC2NodeClass and NodePool with the cluster and instance type filled in.
echo "==> Applying NodePool $NODE_NAME"
sed -e "s|NODE_NAME|${NODE_NAME}|g" -e "s|NODE_ROLE|${NODE_ROLE}|g" \
  -e "s|CLUSTER_NAME|${DISCOVERY}|g" -e "s|INSTANCE_TYPE|${INSTANCE_TYPE}|g" \
  -e "s|GPU_LIMIT|${GPU_LIMIT}|g" "$SCRIPT_DIR/../k8s/examples/<manifest>.yaml" \
  | kubectl --context "$EKS_CLUSTER_NAME" apply -f -
# Show the node pool. A node launches only when a benchmark pod needs one, and scales to zero when idle.
echo "==> Showing the NodePool"
kubectl --context "$EKS_CLUSTER_NAME" get nodepool "$NODE_NAME"
```

The Capacity Block variant adds `RESERVATION_ID=${2:?...}` and seds `RESERVATION_ID` into the manifest. A capacity script never purchases a reservation; the Capacity Block one's missing-argument message tells the user how to find an offering with `describe-capacity-block-offerings`.
