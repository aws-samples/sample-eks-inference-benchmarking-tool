# FAQs

Common questions about scope, methodology, and how this repo relates to other benchmarking tools. See the [README](../../README.md) for the recipes and runbooks. The standard local setup is `./scripts/install-benchmark-client.sh`; installing individual requirements files directly is a minimal-environment path.

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

Two further differences in output shape. MLPerf's Server scenario produces one number, the maximum queries/sec under a latency bound, rather than the concurrency-sweep curve (per-user tok/s at 1, 8, 32, 128, 256, 512) used here to convert a latency SLA into a node count. And MLCommons recommends benchmarking a system through the vendor implementation, which is a heavier lift than running this harness.

### How does this relate to `vllm bench serve`?

`vllm bench serve` is a vLLM-native load generator. This repo adds the surrounding machinery: one methodology held constant across models and instance types, a correctness gate that aborts before measuring, a protocol lock and provenance in every result file, and manifest validation before a deploy. The goal is a number that can be compared against another number taken elsewhere.

### Is there anything from MLPerf adopted here?

Yes, LoadGen. `eks-bench harness-arrival-rate` runs MLPerf's load generator directly (Apache 2.0, no reference implementations needed). The standard client setup is `./scripts/install-benchmark-client.sh`; because LoadGen is not in that default environment, a local direct `arrival_rate` or `verify-openloop` invocation additionally needs `app/bench/requirements-loadgen.txt`. Installing requirements files individually is the minimal setup path.

LoadGen is an addition rather than a replacement, because the two load models answer different questions and their figures are not comparable:

| Load model | Question answered | Used for |
| ---------- | ----------------- | -------- |
| Closed loop, fixed concurrency (the sweeps) | with N concurrent users, what does each one see? | sizing |
| Open loop, Poisson at target QPS (LoadGen) | at X requests/sec arriving, is the SLA held? | capacity and SLA verification |

LoadGen also provides direct issue-rate accounting: it knows the schedule it intended, so it can report when it fell behind. A closed-loop harness has no intended schedule, so client saturation is inferred from client CPU utilization, which `measure` flags above 0.85. One limitation remains on the closed-loop side: percentiles are nearest-rank with no interpolation, which is imprecise at low sample counts, and the per-level request floor mitigates rather than resolves this.

Two things are kept regardless: the concurrency sweep, which is the sizing tool and has no LoadGen equivalent; and the protocol lock, because LoadGen records its own scenario settings but not the server config, instance type, or harness revision that produced a result.

### How does this relate to AIPerf?

This repo uses AIPerf, as a second independent load generator behind the same protocol seam and as the driver for the committed agentic traces. AIPerf provides load generation; the accuracy gate, the record of what the server was running, and manifest validation come from this repo. Its results are scoped to the deployment they sit beside.

### How does this relate to FMBench?

They serve different readers. [FMBench](https://github.com/aws-samples/foundation-model-benchmarking-tool) helps someone choose a platform across Bedrock, SageMaker, EC2, and EKS, and treats the endpoint as a black box. This repo is for the operator who has already chosen self-managed serving and needs the serving configuration itself to be the object of study.

### Why not a general-purpose load tester like Locust or k6?

They have no concept of a token, so TTFT, time per output token, and throughput accounting cannot be measured.

### What is the biggest known gap in this repo's methodology?

Accuracy validation was the largest gap, and that gate now exists: see [the accuracy references README](../accuracy/README.md). Every run records its verdict in the protocol lock, either `accuracy gate : PASSED` against a named reference or a statement that it was skipped.

The remaining gap is uneven coverage rather than absence, and coverage is environment-specific: a model with no captured reference is a model whose runs carry no correctness verification. Check the lock at the top of a result before quoting it.

MLPerf mandates accuracy within a tolerance of a reference implementation, which contributes to the trust its numbers carry, but its accuracy harnesses are bound to specific model-dataset pairs, so there is no MLPerf check for the models here. The gate in this repo is local and follows the same approach: a pinned problem set and a tolerance against a reference. See [`docs/accuracy/README.md`](../accuracy/README.md).

### Why don't the throughput baselines benefit from prefix caching?

A cache hit skips prefill work, so any throughput number it affects is inflated by construction, and the size of the inflation depends on how much prefix real traffic would share. The canonical `corpus` protocol is retained for historical comparability and cycles the five fixed prompts in `legacy_corpus`; with server-side prefix caching enabled, it can hit the cache and is then not an uncached throughput baseline. The default `corpus` workload therefore requires prefix caching disabled for a cache-free claim. More generally, a cache-free claim requires either server-side caching disabled or demonstrably unique prompts, plus provenance showing a zero observed hit rate. If provenance is absent or reports hits, treat the figure as cache-inflated. Generated `PROMPT_MODE=unique` prompts defeat prefix matching, but only when no pinned profile is selected, and they are not what the default `corpus` protocol sends.

The cache-hit case is measured separately, as its own workload, by the `latency_sharedprefix` protocol. It sends a byte-identical 4,000-token shared prefix and requires prefix caching enabled server side. The effect is not uniformly positive: a shared prefix is usually longer than the prompt it replaces, so at moderate concurrency the extra prefill can cost more than the hit saves and TTFT increases rather than decreases. Compare `latency_sharedprefix` only with runs of the same protocol, and not against a unique-prompt or cache-free baseline.

### Why does a benchmark report show `ok%` per concurrency level?

Because a median can rest on far fewer completed requests than the protocol specifies, and a bare tok/s column does not show it. In one sweep, the three reps at concurrency 1 disagreed widely, and two of them had lost nearly all their requests to client-side connection failures, so the 3-rep median largely reflected a broken client. That cell now reports the one clean rep, marked `†` as a single-rep figure.

The runner now enforces this. `measure` emits a `REQUEST SUCCESS per level, per rep` table and excludes any rep with a level below `MIN_LEVEL_SUCCESS` (95%) from the medians. Failures are split into transport (the client's connection broke, so the sample is invalid) and server (the server rejected the request, which is a result). With no usable rep it does not aggregate. The `ok%` column, the `*` degraded-success marker, and the `†` single-rep marker make this visible in the results file itself; results predating the enforcement carry hand-applied markers.
