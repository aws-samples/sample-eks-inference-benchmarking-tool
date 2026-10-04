# Studies

The runbooks in this section are not scoped to a single model. A study answers one question about the platform, usually by holding the model constant and varying something underneath it; a technique applies to whatever recipe you point it at. Either way there is no single model directory these belong in, which is what distinguishes them from the recipes indexed in the [runbooks overview](../README.md).

NCCL interconnect studies that serve a model are recorded in [`experiments/`](../../../experiments/README.md#interconnect-studies) rather than here.

## Runbooks

| Runbook | Question it answers | Scope |
| ------- | ------------------- | ----- |
| [`g7e.12xlarge-pix-vs-phb-proxy-runbook.md`](./g7e.12xlarge-pix-vs-phb-proxy-runbook.md) | What does inference lose when two GPUs lose direct peer access and NCCL stages through host memory? | Llama 3.3 70B FP8 and Qwen3-30B-A3B FP8, TP2, three arms, five repetitions |
| [`g7e.12xlarge-profile-gpus-nsight-systems-runbook.md`](./g7e.12xlarge-profile-gpus-nsight-systems-runbook.md) | Where is a serving pod actually spending GPU time? | Any recipe. The walkthrough uses Qwen2.5-7B at TP=2; the ad-hoc variant wraps any reachable `vllm serve` |

## Naming

Recipe runbooks lead with precision, as in `fp8-g7e.12xlarge-runbook.md`, because precision is what varies across a model's recipes. Study runbooks lead with whatever the study varies instead: the instance and the axis under test for the switch study, the tool for the profiling one. The `-runbook.md` suffix is common to both.

## Substituting your own environment

Both runbooks assume an existing cluster and use the same placeholders as the rest of the repository, documented in the [`k8s/examples` placeholder table](../../../k8s/examples/README.md). Node names appear as `ip-10-0-0-1.us-east-1.compute.internal`; the cluster, region, and namespace come from environment exports the runbook defines.

One constraint is specific to the switch study and easy to miss: every arm must pin the **same** physical node. The comparison is between two transports on one host, so substituting a different node name into one manifest silently turns it into a comparison between two hosts. The eight manifests it applies are a single versioned set.

## Results

Studies write their results into `docs/results/` like any other run, using the recipe stem so a result sorts next to the manifest that produced it. Study manifests carry a `<config>` segment naming the arm, as in `llama3.3-70b-fp8-g7e.12xlarge-fp8-switch-pix-corpus-legacy_corpus.txt`, so the arms stay distinguishable. Results and written reports are gitignored; see [`docs/results/README.md`](../../results/README.md) and [`docs/reports/README.md`](../../reports/README.md) for the formats.
