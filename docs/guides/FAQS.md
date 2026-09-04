# FAQs

Common questions about scope, methodology and how this repo compares to other benchmarking tools. See the [README](../../README.md) for the recipes and runbooks. The standard local setup is `./scripts/install-benchmark-client.sh`; installing individual requirements files directly is an expert/minimal-environment path.

### Does it benchmark training or fine-tuning?

No. Inference serving only.

### What models does it support?

Text models servable by vLLM. Speech, vision and multimodal are out of scope, because both need a different request shape than a text completion and a different serving stack.

### Does it support Inferentia or Trainium?

No. Both are a different stack.

### Does this only work on Amazon EKS?

The harness is a CPU-only HTTP client, so it measures anything serving an OpenAI-compatible API, including vLLM on a bare EC2 instance. The recipes, the in-cluster launcher and the provenance capture are EKS-specific. On plain EC2 you keep the measurement protocol and bring your own deployment.

### What data leaves my environment?

None. Result files record your endpoint, cluster context, namespace, served model path and pod names, which is why `docs/results/` and `docs/reports/` are gitignored. The file format is public; the measurements stay yours.

### Why not use MLPerf instead of this harness?

**They answer different questions.** MLPerf answers *"how does vendor A's system compare to vendor B's on an agreed workload."* This repo answers *"will this specific model run on this specific instance, and which configuration knobs actually matter."*

That rules out MLPerf's Closed division here, because **its model list is fixed**. At the time of this comparison (MLPerf Inference v6.0), the list covers GPT-OSS-120B, DeepSeek-R1, Qwen3-VL-235B-A22B, Wan 2.2 and a handful of others, not Kimi K3, GLM-5.2, Gemma 4 31B, or Qwen2.5-7B. The models benchmarked here are outside that point-in-time list. The Open division permits custom models, but at the cost of comparability, which was MLPerf's whole advantage over a homegrown harness.

Two further mismatches: MLPerf's Server scenario produces one number (maximum queries/sec under a latency bound), not the concurrency-sweep curve (per-user tok/s at 1, 8, 32, 128, 256, 512) that converts a latency SLA into a node count. And MLCommons recommends benchmarking a system through the vendor implementation, a substantially heavier lift than this harness.

### Why not use vllm bench serve?

It is a good vLLM-native load generator, and that is all it is. This repo needs what surrounds the load: one methodology held constant across models and instance types, a correctness gate that aborts before measuring, a protocol lock and provenance in every result file, and manifest validation before a deploy. `vllm bench serve` produces a number; the machinery here exists so the number can be trusted and compared.

### So is there anything from MLPerf worth adopting?

**LoadGen, and it is already in.** `python3 app/cli.py arrival_rate` runs MLPerf's load generator directly (Apache 2.0, no reference implementations needed). The standard client setup is `./scripts/install-benchmark-client.sh`; because LoadGen is not in that default environment, a local direct `arrival_rate` or `verify-openloop` invocation additionally needs `app/bench/requirements-loadgen.txt`. Installing requirements files individually is the expert/minimal setup path.

It is an **addition, not a replacement**, because the two load models answer different questions and their figures must never share a table:

| Load model | Question answered | Used for |
| ---------- | ----------------- | -------- |
| Closed loop, fixed concurrency (the sweeps) | with N concurrent users, what does each one see? | sizing |
| Open loop, Poisson at target QPS (LoadGen) | at X requests/sec arriving, is the SLA held? | capacity and SLA verification |

LoadGen also brings direct issue-rate accounting: it knows the schedule it intended, so it can say when it fell behind. A closed-loop harness has no intended schedule, so client saturation is inferred from client CPU utilization, which `suite` flags above 0.85. One rough edge stays on the closed-loop side: percentiles are nearest-rank with no interpolation, which is crude at low sample counts; the per-level request floor makes it tolerable rather than correct.

What stays either way: the concurrency sweep, because it is the sizing tool and LoadGen has no equivalent; and the protocol lock, because LoadGen records its own scenario settings but nothing about which server config, instance type, or harness revision produced a result.

### Why not AIPerf, since NVIDIA already ships it?

This repo does use it, as a second independent load generator behind the same protocol seam and as the driver for the committed agentic traces. On its own it is load generation: no accuracy gate, no record of what the server was actually running, and no manifest validation. Its results are scoped to the deployment they sit beside.

### Why not FMBench, since AWS already has a benchmarking tool?

Different reader. [FMBench](https://github.com/aws-samples/foundation-model-benchmarking-tool) helps someone choose a platform across Bedrock, SageMaker, EC2 and EKS, and treats the endpoint as a black box. This is for the operator who has already chosen self-managed and needs the serving configuration itself to be the object of study.

### Why not a general-purpose load tester like Locust or k6?

They have no concept of a token, so TTFT, time per output token and throughput accounting are all unmeasurable.

### What is the biggest known gap in this repo's methodology?

**It was accuracy validation, and that gate now exists:** see [the accuracy references README](../accuracy/README.md). Every run records its verdict in the protocol lock: `accuracy gate : PASSED` against a named reference, or a plain statement that it was skipped.

The gap is now uneven coverage rather than absence, and coverage is yours to build: a model with no captured reference is a model whose runs carry no correctness verification. Check the lock at the top of a result before quoting it.

MLPerf mandates accuracy within a tolerance of a reference implementation, which is a large part of why its numbers are trusted, but its accuracy harnesses are bound to specific model-dataset pairs, so there is no MLPerf check for the models here. The gate is local and modelled on the same idea: a pinned problem set and a tolerance against a reference. See [`docs/accuracy/README.md`](../accuracy/README.md).

### Why don't the throughput baselines benefit from prefix caching?

Because a cache hit skips prefill work, any throughput number it touches is inflated by construction, and the size of the inflation depends entirely on how much prefix real traffic would share. The canonical `corpus` protocol is retained for historical comparability and cycles the five fixed prompts in `legacy_corpus`; with server-side prefix caching enabled, it can hit the cache and is not an uncached throughput baseline. The default `corpus` workload therefore requires prefix caching disabled for a cache-free claim. More generally, a cache-free claim requires either server-side caching disabled or demonstrably unique prompts, plus provenance showing a zero observed hit rate. If provenance is absent or reports hits, treat the figure as cache-inflated. Generated `PROMPT_MODE=unique` prompts defeat prefix matching, but only when no pinned profile is selected; they are not what the default `corpus` protocol sends.

The cache-hit case is measured separately, as its own deliberate workload, by the `latency_sharedprefix` protocol. It sends a byte-identical 4,000-token shared prefix and requires prefix caching enabled server side. The effect is not free: a shared prefix is usually longer than the prompt it replaces, so at moderate concurrency the extra prefill can cost more than the hit saves and TTFT gets worse rather than better. Compare `latency_sharedprefix` only with runs of the same protocol; never present it as a unique-prompt or cache-free baseline.

### Why does a benchmark report show `ok%` per concurrency level?

Because a median can rest on far fewer completed requests than the protocol specifies, and a bare tok/s column will not show it. In one sweep the three reps at concurrency 1 disagreed widely, and two of them had lost nearly all their requests to client-side connection failures. The naive 3-rep median was mostly an artifact of a broken client. That cell now reports the one clean rep, marked `†` as a single-rep figure.

**Now enforced by the runner.** `suite` emits a `REQUEST SUCCESS per level, per rep` table and **excludes** any rep with a level below `MIN_LEVEL_SUCCESS` (95%) from the medians. Failures are split into transport (the client's connection broke, so the sample is invalid) and server (the server rejected the request, which *is* a result). With no usable rep it refuses to aggregate at all. The `ok%` column, `*` degraded-success marker and `†` single-rep marker make all of this visible in the results file itself; results predating the enforcement carry hand-applied markers.
