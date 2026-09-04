# eks-inference-benchmarking-tool

A benchmarking tool for sizing self-managed vLLM inference on Amazon EKS. Concurrency sweeps with a correctness gate, a protocol lock on every result, and a serving recipe with a runbook for every model, precision and instance type.

**[Quickstart](#quickstart)** | **[Recipes](./vllm-recipes/)** | **[Runbooks](./docs/runbooks/README.md)** | **[Guides](./docs/guides/README.md)** | **[Architecture](./docs/guides/ARCHITECTURE.md)** | **[Reference](./docs/guides/REFERENCE.md)** | **[Troubleshooting](./docs/guides/TROUBLESHOOTING.md)** | **[FAQs](./docs/guides/FAQS.md)**

## Core capabilities

Each capability names the tool it is built on. This repo does not reimplement what already exists; it supplies the methodology around it.

| Capability | What it does | Why it matters |
|---|---|---|
| ✅ [**Correctness gate**](./docs/accuracy/README.md) | Compares the live server's output against a pinned reference before any measurement runs, scoring a pinned [GSM8K](https://github.com/openai/grade-school-math) subset committed to this repo | A misconfigured server aborts the run instead of producing a fast, wrong result. The gate fails closed, so a missing reference stops the run rather than passing it |
| 🔒 [**Protocol lock and provenance**](./docs/results/README.md) | Opens every result file with the conditions the run was taken under, and closes it with the serving pod's argv, NCCL transport, KV sizing and GPU topology | A number cannot drift away from what produced it, so a result stays comparable against one taken a year later |
| 📝 [**Pinned prompt profiles**](./app/bench/data/prompt_profiles/README.md) | Ships the literal prompt text as committed JSON, generated deterministically, with per-model wire token counts measured offline through Hugging Face Transformers | Every model receives byte-identical input, so a difference between two models belongs to the models. Input size is recorded in every result rather than inferred from a generator |
| 🧪 [**Manifest validation**](./docs/guides/TROUBLESHOOTING.md#a-recipe-applies-cleanly-but-serves-the-wrong-configuration) | Re-runs each recipe's script with [vLLM](https://github.com/vllm-project/vllm) stubbed out and compares the argv that arrives against the flags the recipe is written to pass | Catches a recipe that applies cleanly but serves the wrong configuration, before GPU time is spent trusting it |
| ☸️ [**In-cluster measurement**](./docs/guides/TROUBLESHOOTING.md#a-latency-result-looks-off-or-ttft-is-much-higher-than-expected) | Runs the harness as a Job beside the server, reaching it over cluster DNS | A port-forward adds tunnel setup to every TTFT sample, so the harness refuses to report first-token latency through one |
| 📈 [**Closed-loop concurrency sweeps**](./docs/guides/REFERENCE.md#protocols) | Sweeps 1 to 512 concurrent requests on this repo's own aiohttp client, reporting tokens/sec, req/sec and TTFT, TPOT and ITL across three reps with medians | Converts a latency SLA into a node count, which is the sizing decision |
| 🎯 [**Open-loop SLA verification**](./docs/guides/REFERENCE.md#trusting-the-open-loop-verdict) | Issues Poisson arrivals at a target QPS through [MLPerf LoadGen](https://github.com/mlcommons/inference/tree/master/loadgen), with direct issue-rate accounting | Answers whether the SLA holds at a given arrival rate, a different question from the sweeps, so the two never share a table |
| 🔀 [**Cross-client calibration**](./docs/guides/REFERENCE.md#what-may-be-compared-with-what) | Reruns the same workload through [NVIDIA AIPerf](https://github.com/ai-dynamo/aiperf) behind the same protocol seam, and replays committed agentic traces with real think-time gaps | Two independent clients on one server show whether a figure belongs to the server or to the client that measured it |
| 📦 [**Weight staging**](./model-downloads/README.md) | One Job per model and precision, applied once per filesystem, that downloads the checkpoint to the shared claim | Recipes wait for weights rather than fetching them, so a serving run is not also a download, and a partial checkpoint fails before the GPU is booked |
| 🗜️ [**Quantization pipeline**](#quantization) | Produces FP8, INT8 and INT4-AWQ checkpoints with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), as a GPU Job in your own cluster | Public quantized checkpoints do not exist for every model, and converting one needs more memory than serving the result afterwards |
| ⚙️ [**Serving-stack coverage**](#vllm-recipes) | Covers prefix caching, KV-cache scale configs, [FlashInfer](https://github.com/flashinfer-ai/flashinfer) attention, and multi-node tensor and pipeline parallelism over EFA, across Deployment, StatefulSet and [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) | The serving configuration is the object of study, so the knobs that move a number are already wired up |
| 🔬 [**Profiling and interconnect studies**](#profile-gpus-with-nsight-systems) | Captures [Nsight Systems](https://developer.nvidia.com/nsight-systems) traces, and isolates the interconnect with [nccl-tests](https://github.com/NVIDIA/nccl-tests) | Explains a ceiling that a sweep can only locate. A trace has already overturned a conclusion this repo had published |
| 📚 [**Runbooks, not snippets**](#vllm-recipes) | Pairs every serving recipe with capacity sizing, prerequisites, verified steps and tear-down, from single-GPU `g6e` to `u-p6e-gb200x72` | The starting point is a working deployment rather than a blank manifest |

## Prerequisites

- [Create an EKS cluster with GPU nodes](https://docs.aws.amazon.com/eks/latest/userguide/getting-started.html) (g6e, g7e, p5, p6 or p6e family)
- [Install the NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin#quick-start)
- [Get a Hugging Face token](https://huggingface.co/settings/tokens) (read-only access)
- [Create the weights filesystem and the `fsx-pvc` claim](./k8s/examples/README.md#creating-the-weights-filesystem) (recipes mount this claim and do not create it)
- [Create the `vllm-sitecustomize` ConfigMap](./k8s/examples/configmap-vllm-sitecustomize.yaml) (without it a server exits at startup with an error about repository ids that does not name the cause)

## Quickstart

### 1. Install the local benchmark client

The installer creates `.venv` and installs the benchmark, tokenizer, AIPerf, and CLI dependencies. Run it once to prepare or refresh the environment. After that, `python3 app/cli.py` reuses `.venv`; if `.venv` is missing, the CLI bootstraps it with the same installer and re-executes automatically, so no activation command is required.

```bash
./scripts/install-benchmark-client.sh
python3 app/cli.py --help
```

### 2. Follow the selected runbook

Choose a serving runbook from [vLLM recipes](#vllm-recipes). The runbook sets its recipe-specific environment, installs the shared cluster prerequisites, preserves any recipe-specific checks and weight staging, then calls the recipe-local deploy helper. Keep the visible `python3 app/cli.py` accuracy, dry-run, and benchmark commands in order; each real benchmark is paired with an otherwise identical `--dry-run` command.

---

## Quantization

Quantization reduces the size of the model weights, so a model can fit on fewer or smaller GPUs, allowing more KV cache headroom to run your workloads.

Some runbooks stage published quantized checkpoints with Jobs from [`model-downloads/`](./model-downloads/README.md#fp8-weights-downloaded-or-quantized). Recipes without a usable published checkpoint require local FP8, INT8, or INT4-AWQ quantization with NVIDIA Model Optimizer. Follow the canonical [`job-quantize-model.yaml` workflow](./k8s/examples/README.md#job-quantize-modelyaml) for image preparation, checkpoint conversion, logs, and idempotent reruns.

---

## vLLM recipes

Each runbook below contains the prerequisites to run each benchmark, the steps to run the benchmark, and tear down steps. Choose a model below to find available recipes, with links to each runbook.

The recipes in this section tune the server for throughput and real-time latency. Throughput and latency move in opposite directions across a concurrency sweep. Read the [batch vs real-time inference methodology](./docs/guides/BATCH-VS-REALTIME-INFERENCE.md) before benchmarking.

> **Cost gate:** unless the GPUs are part of a prepaid capacity block, tear down on-demand or Spot nodes as soon as the run finishes.

| Model | Precisions | Instance types | Model guide |
| ----- | ---------- | -------------- | ----------- |
| GLM-5.2 | FP8, NVFP4 | `p5en.48xlarge`, `p6-b200.48xlarge`, `p6-b300.48xlarge` | [README](./docs/runbooks/glm-5.2/README.md) |
| Kimi K3 | MXFP4 | `p6-b200.48xlarge`, `p6-b300.48xlarge`, `u-p6e-gb200x72` | [README](./docs/runbooks/kimi-k3/README.md) |
| Llama 3.3 70B Instruct | FP8 | `g7e.12xlarge` | [README](./docs/runbooks/llama3.3-70b/README.md) |
| Qwen3-235B-A22B | FP8 | `g7e.12xlarge` | [README](./docs/runbooks/qwen3-235b/README.md) |
| Qwen2.5-32B-Instruct | FP8 | `g7e.4xlarge` | [README](./docs/runbooks/qwen2.5-32b/README.md) |
| Qwen2.5-7B-Instruct | BF16, FP8 | `g6e.4xlarge`, `g7e.12xlarge`, `p5.4xlarge`, `p5en.48xlarge` | [README](./docs/runbooks/qwen2.5-7b/README.md) |
| Gemma 4 31B | BF16, FP8 | `g6e.12xlarge`, `g7.24xlarge`, `g7e.4xlarge` | [README](./docs/runbooks/gemma4-31b/README.md) |

Each model guide lists every recipe for that model and links its runbook. [`vllm-recipes/README.md`](./vllm-recipes/README.md) indexes all of them in one table, with instance, GPU count, parallelism, workload kind and serving image per recipe. Recipes whose purpose is to measure the interconnect rather than serving performance are documented under [Interconnect studies](./experiments/README.md#interconnect-studies) in [`experiments/`](./experiments/) instead, and are absent from the model guides for that reason.

---

## NCCL experiments

NCCL is NVIDIA's collective communications library, and it moves tensors between GPUs on every step of a tensor-parallel forward pass. These studies measure the interconnect rather than the model: microbenchmarks with no model or vLLM stack at all, and serving arms that run vLLM like a recipe but exist to compare `NCCL_P2P_LEVEL` settings rather than to measure serving performance.

Follow the [experiments README](./experiments/README.md) for what belongs there, the PCIe-switch-vs-host-copy study, and the recipes that measure the interconnect with their runbooks.

---

## Profile GPUs with Nsight Systems

The benchmark harnesses in this project tell you *how slow*. An Nsight Systems trace tells you *why*: how step time splits across GEMM, the NCCL collective, attention, softmax, sampling and scheduling gaps. Reach for it when a sweep shows a ceiling and nothing explains it. A trace has already overturned one published conclusion: it showed the NCCL collective consuming a substantial share of GPU kernel time after the P2P study above had called it a small, well-overlapped fraction of step time.

Follow the [Profile GPUs with Nsight Systems runbook](./docs/runbooks/studies/g7e.12xlarge-profile-gpus-nsight-systems-runbook.md) for the prerequisites and the capture itself, an eight-step in-cluster walkthrough, an ad-hoc variant for a container you are already inside, and the gotchas that each cost a capture run.

---

## Reference

Use the [Reference](./docs/guides/REFERENCE.md) guide to learn about the measurement CLI: the `launch` and `suite` commands, their flags, the environment variables a run sets, the benchmark protocols, and the rules for which results may be compared with which. It includes [Which benchmark to run](./docs/guides/REFERENCE.md#which-benchmark-to-run), which covers what each protocol measures, what tables it returns and what it does not report.

Run `./scripts/install-benchmark-client.sh` once to prepare or refresh `.venv`, then use `python3 app/cli.py <command> [flags]`. The CLI reuses that environment and bootstraps it with the installer if it is missing, then re-executes there before importing third-party dependencies; Click and Rich do not remain on the measurement path after dispatch.

## Troubleshooting

Use the [Troubleshooting](./docs/guides/TROUBLESHOOTING.md) guide to find errors and solutions, organized by error string. It covers GPU nodes that report `Ready` while advertising no GPUs, two `sm_120` failures on g7e, one from an attention backend with no Blackwell kernels and one during CUDA-graph capture, a checkpoint missing its `quantization_config` block, a context length set above the batched-token budget, and served-model-name mismatches that fail preflight or return 404.

## FAQs

Use the [FAQs](./docs/guides/FAQS.md) guide for scope questions (does it cover training, Inferentia, non-EKS deployments), what data leaves your environment, and how this repo compares to MLPerf, `vllm bench serve`, AIPerf, FMBench and general-purpose load testers.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.