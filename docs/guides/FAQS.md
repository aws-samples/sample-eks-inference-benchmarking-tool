# FAQs

Common questions about scope, methodology, and how this repo relates to other benchmarking tools. See the [README](../../README.md) for the recipes and runbooks. Install the CLI with `pip install -e '.[tokenizer]'`, then use `eks-bench <command> [flags]`. The [CLI commands table](./REFERENCE.md#cli-commands) lists which commands need the other two extras.

### Does it benchmark training or fine-tuning?

No. Inference serving only.

### What models does it support?

Text models servable by vLLM. Speech, vision, and multimodal are out of scope, because they need a different request shape than a text completion and a different serving stack.

### Does it support Inferentia or Trainium?

No. Both use a different stack.

### Does this only work on Amazon EKS?

The harness is a CPU-only HTTP client, so it measures anything serving an OpenAI-compatible API, including vLLM on a bare EC2 instance. The recipes, the in-cluster launcher, and the provenance capture are EKS-specific. On plain EC2 you keep the measurement protocol and supply your own deployment.

### What data leaves my environment?

None. Result files record your endpoint, cluster context, namespace, served model path, and pod names, which is why `docs/results/` and `docs/reports/` are gitignored. The file format is public; the measurements stay in your environment.

### How does this relate to MLPerf?

They answer different questions. MLPerf answers how one vendor's system compares to another's on an agreed workload. This repo answers whether a specific model runs on a specific instance type, and which configuration settings affect the result.

That places this work outside MLPerf's Closed division, because its model list is fixed. At the time of this comparison (MLPerf Inference v6.0), the list covers GPT-OSS-120B, DeepSeek-R1, Qwen3-VL-235B-A22B, Wan 2.2, and a handful of others, rather than Kimi K3, GLM-5.2, Gemma 4 31B, or Qwen2.5-7B. The models benchmarked here are outside that point-in-time list. The Open division permits custom models, but without the cross-vendor comparability that is the Closed division's main contribution.

MLPerf also requires accuracy within a tolerance of a reference implementation. Its accuracy harnesses are bound to specific model-dataset pairs, so there is no MLPerf check for the models here. This repo applies the same idea locally: a pinned GSM8K subset and a tolerance against a reference captured from your own server. See [`docs/accuracy/README.md`](../accuracy/README.md).

Two further differences in output shape. MLPerf's Server scenario produces one number, the maximum queries/sec under a latency bound. This repo produces a concurrency-sweep curve at 1, 8, 32, 128, 256 and 512, reporting aggregate tokens/sec and per-request latency at each level, which is what converts a latency SLA into a node count. MLCommons also recommends benchmarking through the vendor implementation, which is more setup than running this harness.

### How does this relate to `vllm bench serve`?

`vllm bench serve` is a vLLM-native load generator. This repo adds the surrounding machinery: one methodology held constant across models and instance types, a correctness gate that aborts before measuring, a protocol lock and provenance in every result file, and manifest validation before a deploy. Together those make a result comparable with one taken on different hardware at a different time.

### Is there anything from MLPerf adopted here?

Yes, LoadGen. `eks-bench harness-arrival-rate` runs MLPerf's load generator directly. LoadGen is not in the core install, so running it directly with `harness-arrival-rate`, or exercising `selftest-openloop`'s wire checks, needs `pip install -e '.[loadgen]'`. An in-cluster `measure --protocol server_qps` needs nothing extra locally, because the Job installs LoadGen on every run.

LoadGen is an addition rather than a replacement, because the two load models answer different questions and their figures are not comparable:

| Load model | Question answered | Used for |
| ---------- | ----------------- | -------- |
| Closed loop, fixed concurrency (the sweeps) | with N concurrent users, what does each one see? | sizing |
| Open loop, Poisson at target QPS (LoadGen) | at X requests/sec arriving, is the SLA held? | capacity and SLA verification |

LoadGen also provides direct issue-rate accounting: it knows the schedule it intended, so it can report when it fell behind. A closed-loop harness has no intended schedule, so client saturation is inferred from client CPU utilization, which `measure` flags above 0.85. One limitation remains on the closed-loop side: percentiles are nearest-rank with no interpolation, which is imprecise at low sample counts, and the per-level request floor mitigates rather than resolves this.

Two things are kept regardless: the concurrency sweep, which is the sizing tool and has no LoadGen equivalent; and the protocol lock, because LoadGen records its own scenario settings but not the server config, instance type, or harness revision that produced a result.

### How does this relate to AIPerf?

This repo uses AIPerf, as a second independent load generator behind the same protocol seam and as the driver for the committed agentic traces. AIPerf generates the load; the accuracy gate, the record of what the server was running, and manifest validation come from this repo. Three protocols use it: `corpus_aiperf` for cross-client calibration, and `agentx` and `trace_replay` for trace replay. An in-cluster `measure` run of any of them needs nothing extra locally, because the Job installs AIPerf on every run.

### How does this relate to FMBench?

They answer different questions. [FMBench](https://github.com/aws-samples/foundation-model-benchmarking-tool) compares platforms across Bedrock, SageMaker, EC2 and EKS, and treats the endpoint as a black box. This repo assumes self-managed serving is already chosen, and measures the effect of the serving configuration itself.

### Why not a general-purpose load tester like Locust or k6?

They have no concept of a token, so TTFT, time per output token, and throughput accounting cannot be measured.

### How do I know a result was correctness-checked?

Read the `accuracy gate` line in the protocol lock at the top of the result file. It reads either `PASSED (docs/accuracy/<model-tag>-<config>.json)`, naming the reference it matched, or it records that the gate was skipped.

The gate runs before every benchmark and aborts the run if the server disagrees with its reference, so a result cannot be produced with a silently failing gate. Capturing the reference is a one-time step per model and server config, and every runbook includes it. See [`docs/accuracy/README.md`](../accuracy/README.md).

### Why don't the throughput baselines benefit from prefix caching?

A cache hit skips prefill work, so any throughput number it affects is inflated by construction, and the size of the inflation depends on how much prefix real traffic would share. The canonical `corpus` protocol is retained for historical comparability and cycles the five fixed prompts in `legacy_corpus`; with server-side prefix caching enabled, it can hit the cache and is then not an uncached throughput baseline. The default `corpus` workload therefore requires prefix caching disabled for a cache-free claim. More generally, a cache-free claim requires either server-side caching disabled or demonstrably unique prompts, plus provenance showing a zero observed hit rate. If provenance is absent or reports hits, treat the figure as cache-inflated. Generated `PROMPT_MODE=unique` prompts defeat prefix matching, but only when no pinned profile is selected, and they are not what the default `corpus` protocol sends.

The cache-hit case is measured separately, as its own workload, by the `latency_sharedprefix` protocol. It sends a byte-identical 4,000-token shared prefix and requires prefix caching enabled server side. The effect can go either way: a shared prefix is usually longer than the prompt it replaces, so at moderate concurrency the extra prefill can cost more than the cache hit saves, and TTFT goes up. Compare `latency_sharedprefix` only with runs of the same protocol, never against a unique-prompt or cache-free baseline.

### Why does a benchmark report show `ok%` per concurrency level?

A median can rest on far fewer completed requests than the protocol specifies, and a tok/s column alone does not show it. In one sweep the three reps at concurrency 1 disagreed widely, and two had lost nearly all their requests to client-side connection failures, so the 3-rep median mostly reflected a broken client. That cell now reports the one clean rep, marked `†`.

`measure` enforces this. It emits a `REQUEST SUCCESS per level, per rep` table and excludes any rep with a level below `MIN_LEVEL_SUCCESS` (95%) from the medians. Failures are split into transport (the client's connection broke, so the sample is invalid) and server (the server rejected the request, which is a result). With no usable rep it does not aggregate. The `ok%` column, the `*` degraded-success marker, and the `†` single-rep marker make this visible in the results file itself; results predating the enforcement carry hand-applied markers.
