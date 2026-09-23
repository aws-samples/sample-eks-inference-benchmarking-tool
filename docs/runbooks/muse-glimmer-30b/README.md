# Muse-Glimmer-30B runbooks

Runbooks for benchmarking [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B), a ~29.6B dense multimodal model with a 1.8B perception encoder, on Amazon EKS.

| Instance | Precision | Node | GPUs | TP | Runbook |
| -------- | --------- | ---- | ---- | -- | ------- |
| `g7e.2xlarge` | FP8, FP8 KV, 32k context | Single node inference | 1 | unset | [Runbook](./fp8-g7e.2xlarge-fp8kv-32k-runbook.md) |

The recipe is the direct counterpart of the [Gemma 4 31B 32k recipe](../gemma4-31b/fp8-g7e.2xlarge-fp8kv-32k-runbook.md): same GPU, context, profile, KV precision and backend, so the two result files compare the models and nothing else. The model ships a speculative-decoding drafter; it is deliberately not enabled here, because speculation is always its own labelled config.
