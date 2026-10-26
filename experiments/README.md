# Experiments

**These are records of specific studies, not templates.** Their cluster names, namespaces and node selectors are real on purpose: each file is the manifest a measurement was actually taken with, and replacing those values with placeholders would make the record describe a run that never happened. Do not "fix" them.

## What belongs here

A manifest belongs in `experiments/` when all three are true.

- **It answers one question, once.** The artifact is a written finding, not a reusable file. If someone would apply it to serve a model, it is a recipe.
- **It is a controlled comparison.** Several files that differ in exactly one variable are the normal shape here, because the comparison is the point. `k8s/examples/` deliberately collapses near-duplicates into one file per purpose; that would destroy a study.
- **It does not run `vllm serve`, unless vLLM is only the load for an interconnect study.** A manifest that serves a model to measure serving performance is a recipe and lives in [`vllm-recipes/`](../vllm-recipes/), where `validate-manifests` can prove it passes the flags it claims. The Gemma 4 31B, Qwen3-235B and Llama 3.3 70B arms and all three Qwen3-30B-A3B switch arms (default, `PIX`, forced SHM) live here.

Nothing here is checked by `validate-manifests`, whose scope is `vllm-recipes/*/*.yaml`. That is correct for the microbenchmarks, which have no `vllm serve` invocation to verify. The Gemma 4 31B, Qwen3-235B, Llama 3.3 70B and Qwen3-30B-A3B arms here do run `vllm serve`, so their deploy helpers and the switch study runbook check the served model and the live KV-cache configuration instead. The Gemma 4 31B arm has neither, so check its server log by hand.

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
| `g724-nccl-tests-p2p-job.yaml` | `all_reduce_perf` across 4 GPUs on one g7.24xlarge, which has no shared switch |
| `g7e12-nccl-tests-p2p.yaml` | `all_reduce_perf` across 2 GPUs on one g7e.12xlarge, whose GPUs are switch-local (`PIX`) |
| `g7e12-nccl-p2p-shm-matrix.yaml` | The g7e.12xlarge pair in three arms: NCCL default selection, explicit `PIX` P2P, and forced SHM |
| `g7e12-switch-study-storage-and-model-stage.yaml` | Shared storage and checkpoint staging for the g7e.12xlarge switch study; not an arm itself |

The three NCCL files are the clean interconnect signal, with no model and no vLLM stack. Sweep by changing `NCCL_P2P_LEVEL`, or setting `NCCL_P2P_DISABLE=1` for the host-staging lower bound, and compare `busbw`. Keep everything else identical between runs.

Measurements land in `docs/results/`, which is gitignored, so a fresh clone gives you the manifests and you run them yourself. Compare `busbw` between arms; the absolute number is less interesting than which transport wins.

## Interconnect studies

NCCL is NVIDIA's collective communications library, and it moves tensors between GPUs on every step of a tensor-parallel forward pass. The microbenchmarks above isolate the interconnect with no model. These run vLLM, but their purpose is to compare `NCCL_P2P_LEVEL` settings, not to measure serving performance, so they are studies first and their results are not serving benchmarks. All of these arms live here, so none are checked by `validate-manifests`. Change only `NCCL_P2P_LEVEL` between runs of the same manifest; `--config` keeps each arm in its own results file.

| Model | Instance | Precision | Parallelism | What's swept | Manifest | Runbook |
|---|---|---|---|---|---|---|
| Gemma 4 31B | `g7.24xlarge` | FP8 | TP4 | `NCCL_P2P_LEVEL` across four PCIe GPUs with no NVLink | [`g724-p2p-gemma4-31b.yaml`](./g724-p2p-gemma4-31b.yaml) | None |
| Qwen3-235B-A22B | `g7e.12xlarge` x2 | FP8 | TP2, PP2 | `NCCL_P2P_LEVEL` across two nodes, one 400 Gbps EFA card each, no NVLink | [`g7e12-2node-qwen3-235b-lws.yaml`](./g7e12-2node-qwen3-235b-lws.yaml), deployed with [`g7e12-2node-qwen3-235b-lws-deploy.sh`](./g7e12-2node-qwen3-235b-lws-deploy.sh) (`--config 2node`) | None |
| Llama 3.3 70B Instruct | `g7e.12xlarge` | FP8 | TP2 | NCCL default selection, explicit `PIX` P2P, and forced SHM on one switch-local GPU pair | [`g7e12-switch-llama3.3-70b-default.yaml`](./g7e12-switch-llama3.3-70b-default.yaml), [`-pix`](./g7e12-switch-llama3.3-70b-pix.yaml), [`-shm`](./g7e12-switch-llama3.3-70b-shm.yaml) | [Runbook](../docs/runbooks/studies/g7e.12xlarge-pix-vs-phb-proxy-runbook.md) |
| Llama 3.3 70B Instruct | `g7e.12xlarge` | FP8 | TP2 | Nothing: single-arm baseline, NCCL defaults, checkpoint downloaded into node-local storage | [`g7e12-llama3.3-70b-tp2.yaml`](./g7e12-llama3.3-70b-tp2.yaml), deployed with [`g7e12-llama3.3-70b-tp2-deploy.sh`](./g7e12-llama3.3-70b-tp2-deploy.sh) | None |
| Qwen3-30B-A3B | `g7e.12xlarge` | FP8 | TP2, expert parallel | NCCL default selection, explicit `PIX` P2P, and forced SHM on one switch-local GPU pair | [`g7e12-switch-qwen3-30b-a3b-default.yaml`](./g7e12-switch-qwen3-30b-a3b-default.yaml), [`-pix`](./g7e12-switch-qwen3-30b-a3b-pix.yaml), [`-shm`](./g7e12-switch-qwen3-30b-a3b-shm.yaml) | [Runbook](../docs/runbooks/studies/g7e.12xlarge-pix-vs-phb-proxy-runbook.md) |

The Qwen3-235B weights are staged by [`model-jobs/qwen3-235b/fp8-model-download.yaml`](../model-jobs/qwen3-235b/fp8-model-download.yaml), which stays with the other checkpoint-staging Jobs.

The Gemma arm forces a per-layer all-reduce across four PCIe GPUs with no NVLink, which is the thing being measured, not a tuned configuration. The Qwen3 arm splits a 235B model's all-reduce across two nodes with one EFA card each and no NVLink; the cross-node link, not GPU compute, sets the ceiling, and `--config 2node` labels its results so they are never mistaken for a serving benchmark. The Llama switch arms hold the model, node and GPU pair constant and change only the NCCL transport; their checkpoint is staged by `g7e12-switch-study-storage-and-model-stage.yaml`. The Qwen3-30B-A3B default and `PIX` arms also ship `-deploy.sh` helpers for serving a single arm on a pinned node outside the full study, with the checkpoint staged by [`model-jobs/qwen3-30b-a3b/fp8-model-download.yaml`](../model-jobs/qwen3-30b-a3b/fp8-model-download.yaml).

## Adding or removing a study

All three reports cite these paths, and three of those citations are clickable links, so a rename or a move breaks them. Repoint the citing report in the same change. A study that is finished and whose report is written still belongs here: the report is only as good as the manifest a reader can check it against.
