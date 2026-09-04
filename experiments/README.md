# Experiments

**These are records of specific studies, not templates.** Their cluster names, namespaces and node selectors are real on purpose: each file is the manifest a measurement was actually taken with, and replacing those values with placeholders would make the record describe a run that never happened. Do not "fix" them.

## What belongs here

A manifest belongs in `experiments/` when all three are true.

- **It answers one question, once.** The artifact is a written finding, not a reusable file. If someone would apply it to serve a model, it is a recipe.
- **It is a controlled comparison.** Several files that differ in exactly one variable are the normal shape here, because the comparison is the point. `k8s/examples/` deliberately collapses near-duplicates into one file per purpose; that would destroy a study.
- **It does not run `vllm serve`.** Anything that does is a recipe by definition and lives in [`vllm-recipes/`](../vllm-recipes/), where `validate-manifests` can prove it passes the flags it claims. The serving arm of the NCCL study moved out for exactly this reason: it is now `vllm-recipes/gemma4-31b/fp8-g7.24xlarge-p2p.yaml`.

Nothing here is checked by `validate-manifests`, whose scope is `vllm-recipes/*.yaml`. That is correct for files with no `vllm serve` invocation to verify, and it is the reason the third rule matters.

## The three directories

| Directory | Contract | Identifiers |
|---|---|---|
| [`vllm-recipes/`](../vllm-recipes/) | One per published result, with the real flags that produced it | Real, and validated |
| [`k8s/examples/`](../k8s/examples/) | One per purpose, copy and substitute before applying | Placeholders, structurally valid |
| `experiments/` | One per study arm, kept as evidence for a report | Real, and left alone |

## What is here

The PCIe-switch vs host-copy study: does peer-to-peer DMA between GPUs beat staging through host memory, and does having a real PCIe switch change the answer?

| File | Arm |
|---|---|
| `nccl-tests-p2p-job.yaml` | `all_reduce_perf` across 4 GPUs on one g7.24xlarge, which has no shared switch |
| `nccl-tests-p2p-g7e.yaml` | `all_reduce_perf` across 2 GPUs on one g7e.12xlarge, whose GPUs are switch-local (`PIX`) |

Both are the clean interconnect signal, with no model and no vLLM stack. Sweep by changing `NCCL_P2P_LEVEL`, or setting `NCCL_P2P_DISABLE=1` for the host-staging lower bound, and compare `busbw`. Keep everything else identical between runs.

Measurements land in `docs/results/`, which is gitignored, so a fresh clone gives you the manifests and you run them yourself. Compare `busbw` between arms; the absolute number is less interesting than which transport wins.

## Interconnect studies

NCCL is NVIDIA's collective communications library, and it moves tensors between GPUs on every step of a tensor-parallel forward pass. The microbenchmarks above isolate the interconnect with no model. These two run vLLM like a recipe, and by the third rule under [What belongs here](#what-belongs-here) they live in `vllm-recipes/` rather than here, but their purpose is to compare `NCCL_P2P_LEVEL` settings, not to measure serving performance, so they are studies first and their results are not serving benchmarks. Change only `NCCL_P2P_LEVEL` between runs of the same recipe; `--config` keeps each arm in its own results file.

| Model | Instance | Precision | Parallelism | What's swept | Recipe | Runbook |
|---|---|---|---|---|---|---|
| Gemma 4 31B | `g7.24xlarge` | FP8 | TP4 | `NCCL_P2P_LEVEL` across four PCIe GPUs with no NVLink | [`fp8-g7.24xlarge-p2p.yaml`](../vllm-recipes/gemma4-31b/fp8-g7.24xlarge-p2p.yaml) | [Runbook](../docs/runbooks/gemma4-31b/fp8-g7.24xlarge-p2p-runbook.md) |
| Qwen3-235B-A22B | `g7e.12xlarge` x2 | FP8 | TP2, PP2 | `NCCL_P2P_LEVEL` across two nodes, one 400 Gbps EFA card each, no NVLink | [`fp8-g7e.12xlarge-lws.yaml`](../vllm-recipes/qwen3-235b/fp8-g7e.12xlarge-lws.yaml) (`--config 2node`) | [Runbook](../docs/runbooks/qwen3-235b/fp8-g7e.12xlarge-multi-node-efa-runbook.md) |

The Gemma arm forces a per-layer all-reduce across four PCIe GPUs with no NVLink, which is the thing being measured, not a tuned configuration. The Qwen3 arm splits a 235B model's all-reduce across two nodes with one EFA card each and no NVLink; the cross-node link, not GPU compute, sets the ceiling, and `--config 2node` keeps its results out of the single-node set for the same recipe.

## Adding or removing a study

All three reports cite these paths, and three of those citations are clickable links, so a rename or a move breaks them. Repoint the citing report in the same change. A study that is finished and whose report is written still belongs here: the report is only as good as the manifest a reader can check it against.
