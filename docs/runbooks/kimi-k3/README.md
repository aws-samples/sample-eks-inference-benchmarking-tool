# Kimi K3

The runbooks listed in this section are for the [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3): a 2.8-trillion parameter mixture-of-experts model from Moonshot AI that activates roughly 104 billion parameters per token. Kimi K3 is built on Kimi Delta Attention and Attention Residuals, accepts native vision input, and supports a 1-million-token context window.

## Recipes

Use any one of the following runbook recipes to get started. For additional flags and configurations, see [Which benchmark to run](../../guides/REFERENCE.md#which-benchmark-to-run).

Recipes that share an instance type differ in their serving configuration, so the `--config` column is what distinguishes them and what you pass to `measure`.

| Instance | Precision | `--config` | Node | GPUs | TP | PP | Runbook |
| -------- | --------- | ---------- | ---- | ---- | -- | -- | ------- |
| `p6-b200.48xlarge` | MXFP4 | `sts` | Multi node inference | 16 | 8 | 2 | [Runbook](./mxfp4-p6-b200.48xlarge-sts-runbook.md) |
| `p6-b300.48xlarge` | MXFP4 | `baseline` | Single node inference | 8 | 8 | none | [Runbook](./mxfp4-p6-b300.48xlarge-baseline-runbook.md) |
| `p6-b300.48xlarge` | MXFP4 | `dspark` | Single node inference | 8 | 8 | none | [Runbook](./mxfp4-p6-b300.48xlarge-dspark-runbook.md) |

The multi node recipe uses 16 GPUs across two `p6-b200.48xlarge` nodes. It is a StatefulSet that runs tensor parallelism inside each node and pipeline parallelism between the two, with the rank 0 pod serving the API and rank 1 running headless.

## Results

Each benchmark run saves a gitignored results file to `docs/results/`. The filename follows the recipe name of `kimi-k3-<quant>-<instance>[-<config>]-<protocol>[-<profile>]` to ensure results sort next to the recipe that produced them. The `<config>` segment is only included when it differs from `<quant>`, so every recipe here carries its topology tag. Open-loop runs add an `openloop` suffix.

## Known issues

These recipes need infrastructure that most others in this repo do not. They run a custom image built with matched NCCL and EFA versions rather than the AWS Deep Learning Container, and each runbook's `Setup` section covers building it. The multi node recipe also needs the EFA device plugin installed before it will schedule. Each recipe syncs its weights from S3 to node-local NVMe. [`mxfp4-model-download.yaml`](../../../model-jobs/kimi-k3/mxfp4-model-download.yaml) stages the published checkpoint into S3 once, and [`dspark-draft-download.yaml`](../../../model-jobs/kimi-k3/dspark-draft-download.yaml) stages the draft the dspark recipe adds.

For reference, see [`TROUBLESHOOTING.md`](../../guides/TROUBLESHOOTING.md) for background on what was encountered and why.
