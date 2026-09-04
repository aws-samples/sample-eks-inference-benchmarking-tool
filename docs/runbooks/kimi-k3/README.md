# Kimi K3

The runbooks listed in this section are for the [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3): a 2.8-trillion parameter mixture-of-experts model from Moonshot AI that activates roughly 104 billion parameters per token. Kimi K3 is built on Kimi Delta Attention and Attention Residuals, accepts native vision input, and supports a 1-million-token context window.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

Several recipes share an instance type and differ only in how they split the model, so the `--config` column is what distinguishes them and what you pass to `suite`.

| Instance | Precision | `--config` | Node | GPUs | TP | PP | Runbook |
| -------- | --------- | ---------- | ---- | ---- | -- | -- | ------- |
| `p6-b200.48xlarge` | MXFP4 | `lws` | Multi node inference | 16 | 16 | none | [Runbook](./mxfp4-p6-b200.48xlarge-lws-runbook.md) |
| `p6-b200.48xlarge` | MXFP4 | `lws-dspark` | Multi node inference | 16 | 16 | none | [Runbook](./mxfp4-p6-b200.48xlarge-lws-dspark-runbook.md) |
| `p6-b200.48xlarge` | MXFP4 | `sts` | Multi node inference | 16 | 8 | 2 | [Runbook](./mxfp4-p6-b200.48xlarge-sts-runbook.md) |
| `p6-b200.48xlarge` | MXFP4 | `sts-latency` | Multi node inference | 16 | 8 | 2 | [Runbook](./mxfp4-p6-b200.48xlarge-sts-latency-runbook.md) |
| `p6-b300.48xlarge` | MXFP4 | `baseline` | Single node inference | 8 | 8 | none | [Runbook](./mxfp4-p6-b300.48xlarge-baseline-runbook.md) |
| `p6-b300.48xlarge` | MXFP4 | `dspark` | Single node inference | 8 | 8 | none | [Runbook](./mxfp4-p6-b300.48xlarge-dspark-runbook.md) |
| `p6-b300.48xlarge` | MXFP4 | `ep` | Single node inference | 8 | 8 | none | [Runbook](./mxfp4-p6-b300.48xlarge-ep-runbook.md) |
| `u-p6e-gb200x72` | MXFP4 | none | Single node inference | 16 | 16 | none | [Runbook](./mxfp4-u-p6e-gb200x72-runbook.md) |

The multi node recipes all use 16 GPUs across two `p6-b200.48xlarge` nodes, and they split the model differently. The `lws` recipes are a LeaderWorkerSet with tensor parallelism spanning all 16 GPUs. The `sts` recipes are a StatefulSet that runs tensor parallelism inside each node and pipeline parallelism between the two, with the rank 0 pod serving the API and rank 1 running headless.

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `kimi-k3-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, so every recipe here carries its topology tag except `u-p6e-gb200x72`. Open-loop runs add an `openloop` suffix.

## Known issues

These recipes need infrastructure that most others in this repo do not. Most run a custom image built with matched NCCL and EFA versions rather than the AWS Deep Learning Container, and each runbook's `Setup` section covers building it. The exceptions are the `lws-dspark` and `u-p6e-gb200x72` recipes, which run the upstream image., and the multi-node recipes need the LeaderWorkerSet controller and the EFA device plugin installed before they will schedule. Each runbook states which of these its recipe requires and where its weights come from, which is either S3 to node-local NVMe or an FSx PVC that must be staged in advance.

For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
