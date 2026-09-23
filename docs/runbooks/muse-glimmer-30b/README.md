# Muse-Glimmer-30B runbooks

Runbooks for benchmarking [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B), a ~29.6B dense multimodal model with a 1.8B perception encoder, on Amazon EKS.

| Instance | Precision | Node | GPUs | TP | Runbook |
| -------- | --------- | ---- | ---- | -- | ------- |
| `g7e.2xlarge` | FP8, FP8 KV, 32k context | Single node inference | 1 | unset | [Runbook](./fp8-g7e.2xlarge-fp8kv-32k-runbook.md) |


