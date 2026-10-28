# Kimi K3 (2.8T MoE) MXFP4, DSpark draft-token sweep on p6-b300.48xlarge

Serves [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) on p6-b300.48xlarge (8x B300-SXM6-AC, sm_100) with 8 GPUs per pod, and sweeps DSpark speculative decoding across 3, 5, and 7 draft tokens to find the latency sweet spot. It reuses the [DSpark recipe](./mxfp4-p6-b300.48xlarge-dspark-runbook.md); only `NUM_SPEC_TOKENS` changes between arms.

## Capacity sizing

Same as the DSpark recipe: the MXFP4 checkpoint is about 1,454 GiB plus the published DSpark draft [`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark), on 8 B300-SXM6-AC GPUs, TP8, FP8 KV, 32,768-token context. See the [DSpark runbook](./mxfp4-p6-b300.48xlarge-dspark-runbook.md) for the full table. More draft tokens raise the proposed count per step but cost verify slots, so the question this sweep answers is where acceptance stops paying for that cost.

Each arm is its own labelled config (`dspark-spec3`, `dspark-spec5`, `dspark-spec7`), so results sort separately and never merge. Speculation acceptance is read as a delta between two `vllm:spec_decode_*` counter snapshots, because those counters are cumulative and never reset.

## Before you begin

- [Create an EKS cluster with Karpenter, Karpenter discovery tags and node role](../../../scripts/create-cluster.sh)
- [Purchase a p6-b300.48xlarge Capacity Block for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)

## Quickstart

1. Set environment variables.

   ```bash
   export AWS_REGION=<your-cluster-region>
   export EKS_CLUSTER_NAME=<your-cluster-name>
   export HUGGING_FACE_HUB_TOKEN=<your-hf-token>
   export NAMESPACE=vb-kimi-k3-mxfp4-p6-b300-dspark-sweep
   export INSTANCE_TYPE=p6-b300.48xlarge
   export RESERVATION_ID=<your-capacity-block-id>
   export ARTIFACT_ROOT=artifacts/kimi-k3-dspark-sweep
   mkdir -p "$ARTIFACT_ROOT"
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

5. Install the Pod Identity Agent and grant the `kimi-k3-bench` ServiceAccount S3 access.
   ```bash
   ./scripts/setup-pod-identity.sh kimi-k3-bench
   ```

6. Copy the published checkpoint from Hugging Face to your S3 bucket.
   ```bash
   ./model-jobs/kimi-k3/mxfp4-model-download.sh
   ```

7. Copy the published DSpark draft model from Hugging Face to your S3 bucket.
   ```bash
   ./model-jobs/kimi-k3/dspark-draft-download.sh
   ```

8. Run the sweep. For each draft-token count, deploy, snapshot the speculation counters, capture the accuracy gate, measure, snapshot the counters again, then tear the arm down before the next one. Run the arms one at a time; each requests all eight GPUs.

   ```bash
   for SPEC in 3 5 7; do
     CONFIG="dspark-spec${SPEC}"
     ARM="$ARTIFACT_ROOT/$CONFIG"
     mkdir -p "$ARM"

     ./vllm-recipes/kimi-k3/mxfp4-p6-b300.48xlarge-dspark-sweep-deploy.sh "$SPEC"

     # Snapshot the cumulative speculation counters before the measured run.
     kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec deployment/kimi-k3-b300-dspark \
       --container vllm -- sh -c "curl -s localhost:8000/metrics | grep '^vllm:spec_decode'" \
       | tee "$ARM/spec-before.txt"

     eks-bench capture-reference \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300-dspark \
       --mode fingerprint --model-tag kimi-k3 --config "$CONFIG" \
       --model-id kimi-k3 --thinking off --out-dir docs/results

     eks-bench measure \
       --context "$EKS_CLUSTER_NAME" --namespace "$NAMESPACE" --service kimi-k3-b300-dspark \
       --protocol corpus --profile legacy_corpus \
       --instance p6-b300.48xlarge --model-tag kimi-k3 \
       --config "$CONFIG" --quant mxfp4 \
       --model-id kimi-k3 --tokenizer moonshotai/Kimi-K3 --thinking off \
       --out-dir docs/results --reps 3

     # Snapshot again; the delta of accepted vs proposed tokens is this arm's acceptance.
     kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec deployment/kimi-k3-b300-dspark \
       --container vllm -- sh -c "curl -s localhost:8000/metrics | grep '^vllm:spec_decode'" \
       | tee "$ARM/spec-after.txt"

     kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete \
       deployment/kimi-k3-b300-dspark service/kimi-k3-b300-dspark --ignore-not-found
     kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" wait \
       --for=delete deployment/kimi-k3-b300-dspark --timeout=600s
   done
   ```

   For each arm, the acceptance is the increase in accepted tokens divided by the increase in proposed tokens between `spec-before.txt` and `spec-after.txt`; mean accepted length is accepted tokens per verify step. Confirm the exact `vllm:spec_decode_*` counter names on the first run, since they are version-specific. Compare `tokens_per_sec`, end-to-end latency, and TTFT/TPOT across the three result files, read side by side with the per-arm acceptance.

## Tear down

The loop deletes each arm's server as it finishes. Remove the node pool once the sweep is done.

```bash
kubectl --context "$EKS_CLUSTER_NAME" delete \
    nodepool/bench-cb-p6-b300-48xlarge ec2nodeclass/bench-cb-p6-b300-48xlarge --ignore-not-found
```

Deleting the servers leaves the node empty, and Karpenter scales it to zero.
