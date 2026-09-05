# Reference

Complete specification of the measurement tool: the commands, their flags, the environment variables a run sets, and the benchmark protocols. See the [README](../../README.md) for the recipes and runbooks. The standard local setup is `./scripts/install-benchmark-client.sh`; then use `python3 app/cli.py <command> [flags]`. `app/utils.py` bootstraps the virtual environment, and the operator CLI execs `app/bench/src/main.py`, which dispatches the implementation commands. Individual requirements filenames below identify implementation dependencies; installing them directly is a minimal-environment path.

- [CLI commands](#cli-commands)
- [`launch` flags](#launch-flags)
- [`suite` flags](#suite-flags)
- [Environment knobs](#environment-knobs)
- [Protocols](#protocols)
- [What may be compared with what](#what-may-be-compared-with-what)
- [Trusting the open-loop verdict](#trusting-the-open-loop-verdict)
- [Which benchmark to run](#which-benchmark-to-run)

---

## CLI commands

| Command | Purpose | Implementation dependency |
| ------- | ------- | ---------- |
| `throughput` | Closed-loop throughput sweep, non-streaming. tokens/sec, req/sec, p50/p99. | n/a |
| `latency` | Closed-loop latency sweep, streaming. TTFT / TPOT / ITL. | n/a |
| `arrival_rate` | Open-loop capacity, Poisson arrivals via MLPerf LoadGen. Not comparable to the sweeps. | `requirements-loadgen.txt` |
| `suite` | Protocol-locked runner: reps, medians, accuracy gate, provenance. The normal entry point. | n/a |
| `accuracy` | Run the correctness gate directly: check a live server against a captured reference, or capture one with `--write-reference`. `--mode fingerprint` compares outputs, `--mode score` compares GSM8K accuracy within `--tolerance`. `suite` calls this before every measurement, so run it by hand only to capture a reference or to diagnose a gate failure. | n/a |
| `fetch-eval-data` | Download and pin the GSM8K subset the accuracy gate scores against. | n/a |
| `build-profiles` | Regenerate `app/bench/data/prompt_profiles/`. Deterministic; `--check` verifies reproducibility. | n/a |
| `verify-profiles` | Measure the pinned prompts against a real tokenizer, raw and wire. | `requirements-tokenizer.txt` |
| `verify-aiperf` | Regression checks for the AIPerf adapter: profile conversion, output-cap propagation, request scaling, stdout contract, success floor on a deliberately broken run. Runs against a local mock, no GPU. | `requirements-aiperf.txt` (wire checks skipped without it) |
| `verify-openloop` | Regression checks for the LoadGen harness: percentile-certification floor, warmup isolation, failure attribution, circuit breaker, dead-server exclusion. Runs against a local mock, no GPU. | `requirements-loadgen.txt` (wire checks skipped without it) |
| `verify-accuracy-gate` | Regression checks for fingerprint aggregation and truncation tolerance using synthetic item lists and local references. Runs locally, no server or GPU. | n/a |
| `validate-manifests [recipe…]` | Prove a recipe in [`vllm-recipes/`](../../vllm-recipes/) passes the flags it is written to pass, that its Service points at the port vLLM binds, and that no variable its script reads is left undefined. Checks every recipe when given no argument, which is the pre-deploy gate; name one to check only what you are about to apply. Nothing under `k8s/examples/` is in scope. | `app/requirements-cli.txt` |
| `launch` | Run `suite`, or capture an accuracy reference, **in-cluster** over cluster DNS. Required for any TTFT figure. | n/a |

**Why throughput and latency are separate clients.** Streaming is the only way to observe TTFT and TPOT, but per-token SSE parsing on one asyncio loop can itself become the bottleneck at high concurrency, which would affect the throughput figure at peak. Keeping the throughput harness non-streaming avoids that, and the latency pass is scoped to the concurrency range where first-token latency is meaningful. `arrival_rate` answers a third question, whether an SLA holds at a given arrival rate, and its figures are not comparable with the closed-loop sweeps.

## `launch` flags

Everything `launch` does not recognise is forwarded verbatim to `suite`, so protocol, profile and gate flags need no duplicate definition.

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--context` | required | kubectl context. |
| `--namespace` | none (required) | Namespace holding the vLLM Service and the runner Job. No default, because it forms part of the cluster-DNS endpoint and a default could resolve to an unrelated Service. |
| `--service` | required | vLLM Service name. The endpoint is derived as `http://<service>.<namespace>.svc.cluster.local:<port>/v1`. |
| `--port` | `8000` | Service port. |
| `--out-dir` | required | Local directory for results extracted from the Job's stdout, normally `docs/results`. Same convention as `measure-local`; still required in reference-capture mode even though the reference is restored under `docs/accuracy/`. |
| `--capture-accuracy-reference` | off | Run `accuracy --write-reference` in the in-cluster Job instead of `suite`, forwarding accuracy options such as `--mode`, `--model-tag`, `--config`, and `--model-id`. The launcher extracts the JSON into `docs/accuracy/`; capture remains a separate run from benchmark checking. |
| `--keep-job` | off | Leave the Job and code ConfigMap in place for debugging. |
| `--dry-run` | off | Pack and render, create nothing. Needs no cluster access. |

## `suite` flags

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--protocol` | required | Which measurement protocol to run, see [Which benchmark to run](#which-benchmark-to-run) to choose one and [Protocols](#protocols) for the comparability rules. Pair with `--profile`: `corpus` + `legacy_corpus`, `reasoning` + `legacy_reasoning`, `latency` + `coding_completion`, `server_qps` + `coding_completion`. The trace-replay protocols take no profile. One invocation writes one result file. |
| `--instance` | required | Full instance type recorded in the result header and filename, e.g. `p6-b300.48xlarge`. Not a family tag: `g7e` alone covers 2xlarge, 12xlarge, and 24xlarge, so three different measurements would appear identical. |
| `--config` | required | Server-config tag, e.g. `nvfp4`. Selects the accuracy reference. |
| `--model-tag` | `kimik3` | Model tag for the reference path `docs/accuracy/<model-tag>-<config>.json`. |
| `--quant` | `mxfp4` | Precision tag recorded in the result header. |
| `--model-id` | `kimi-k3` | **Served**-model-name as `/v1/models` reports it, which is the recipe's `--served-model-name` when it sets one and the path passed to `vllm serve` otherwise. Every runbook has it filled in. Preflight fails if it is not served. |
| `--tokenizer` | none | HF repo id of the model's tokenizer, e.g. `nvidia/GLM-5.2-NVFP4`. **Not** the same as `--model-id`. Resolves the recorded wire input size for the protocol lock only, the run never tokenizes. Unset, or pointed at a model with no recorded counts, and the lock says so and prints the command that would record it. |
| `--endpoint` | `http://127.0.0.1:8000/v1` | Base URL under test. |
| `--reps` | `3` | Repetitions. The aggregate reports the median per cell and flags spread above 10%. |
| `--profile` | protocol default | Pinned prompt profile. Overrides the protocol's prompt settings **and** its output cap, because a profile pins both ends of the shape. |
| `--no-profile` | off | Serve generated prompts instead. Input size becomes implicit, so this is not recommended for published figures. |
| `--trace` | protocol default | Repo-relative path to a timestamped trace for the trace-replay protocols, e.g. `app/bench/data/traces/agentx_32k_seed42.jsonl`. Required by `trace_replay`, which pins none. Keep traces under `app/bench/data/traces/` so the in-cluster payload carries them. The lock records the path **and** a SHA-256 of the file, so two results claiming the same trace can be checked rather than assumed. |
| `--no-latency` | off | Skip the companion streaming-latency pass after a throughput sweep. Roughly halves GPU time, and reports no TTFT/TPOT/ITL. |
| `--allow-loopback-latency` | off | Permit a streaming pass against a loopback endpoint. Refused by default, because a tunnel adds 150-200 ms to every TTFT sample. Recorded in the lock, so the result is identifiable as unsuitable for a latency claim. |
| `--skip-accuracy` | off | Skip the correctness gate. Recorded in the lock, so the run stays identifiable as unverified. |
| `--ttft-target` | protocol default | Override the TTFT target in milliseconds for `server_qps`. Forwarded through `measure`; changes the capacity question and result filename when it differs from the shipped value. |
| `--tpot-target` | protocol default | Override the TPOT target in milliseconds for `server_qps`. Forwarded through `measure`; changes the capacity question and result filename when it differs from the shipped value. |
| `--out-dir` | required | Results directory, normally `docs/results`, which is flat: the filename carries model, quantisation and instance, so there are no subdirectories. Required rather than defaulted, so results cannot be written to an unintended directory. |
| `--pod` | auto-discovered | Pod to capture provenance from. Found automatically: the Running pod with a `vllm` container in `--namespace`. Pass this only to disambiguate when several match. |
| `--no-provenance` | off | Skip provenance capture. The result then carries no evidence of NCCL transport, KV sizing or GPU topology. |
| `--k8s-context` | current kubectl context | Context used for pod discovery and provenance capture only. `measure` does not need it: it captures provenance itself, reusing `--context`. |
| `--namespace` | none (required) | Namespace used for pod discovery and provenance capture only. Required unless `--no-provenance` is given, and checked before the sweep starts rather than after it. |
| `--dry-run` | off | Print the protocol lock and output paths, run nothing. Needs no server. Useful as a first step to review the conditions before spending GPU time. |

## Environment knobs

`measure` sets these from the protocol so a run cannot be half-configured; set them by hand only for a bare harness invocation. Full list with defaults: `app/bench/src/config.py`.

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `VLLM_API_URL` | `http://vllm-server:8000/v1` | Endpoint under test. |
| `MODEL_ID` | Qwen2.5-7B FP8 path | Served-model-name. |
| `INSTANCE_TYPE` / `QUANTIZATION` | `unknown` / `fp8` | Tags recorded in the result header. |
| `CONCURRENCY_SWEEP` | `1,8,32,128,256,512` | Levels to sweep. Protocols must use the canonical list. |
| `CONCURRENCY` | `0` | Run a single level instead of the sweep; `0` means sweep. |
| `MAX_TOKENS` | `150` throughput / `256` latency | Output cap per request. A profile overrides this. |
| `NUM_REQUESTS` | `100` throughput / `50` latency | Request floor per level, never a cap. |
| `MIN_DURATION_S` | `60` throughput / `120` arrival-rate | Bare-harness duration floor per level. Protocol locks override it: `corpus=0`, `reasoning=90`, and `server_qps=120`. Latency protocols use their own sample floors rather than this duration setting. |
| `WARMUP_REQUESTS` | `0` | Requests discarded per level before measuring. The open-loop harness issues them before LoadGen starts and resets its counters afterwards, so warmup can never enter a reported figure. |
| `MIN_QUERY_COUNT` | `1500` | Open-loop query floor per level. This covers roughly five over-target samples under LoadGen's early-stopping arithmetic, but no fixed floor guarantees certification; the observed tail determines the requirement. An uncertified level is `UNDERSAMPLED` and `INVALID`, not an SLA pass. |
| `ABORT_AFTER_FAILURES` | `50` | Consecutive open-loop failures after which the client stops sending. A level whose server has become unreachable cannot produce a measurement, so the run stops rather than continuing to issue requests. |
| `THROUGHPUT_FILL_LEVELS` | on | Size each level to whole waves so it reaches its labelled in-flight depth. |
| `THROUGHPUT_MIN_WAVES` | `2` | Minimum whole waves per level. |
| `PROMPT_PROFILE` | unset | Pinned profile name. Takes precedence over the generated modes. |
| `PROMPT_MODE` | `unique` | Generated-prompt fallback: `unique`, `legacy`, `shared_prefix`. |
| `PROMPT_TOKENS` | `0` | Target input length for generated prompts; `0` means short natural prompts. |
| `PROMPT_SEED` | `1337` | Seed for generated prompts. `measure` varies it per rep. |
| `BENCH_ENDPOINT` | `chat` | `chat` or `completions`. |
| `BENCH_TEMPERATURE` | `0.7` | Sampling temperature used by benchmark harnesses; the accuracy gate remains deterministic. |
| `MIN_LEVEL_SUCCESS` | `0.95` | Minimum successful-request fraction required for a level's repetition to contribute to medians. |
| `TRANSPORT_ABORT_SHARE` | `0.10` | Transport-failure share that abandons the remaining sweep instead of measuring a broken client path. |
| `SERVER_TARGET_QPS` | `1.0` | Target arrival rate for a bare `arrival_rate` harness invocation; `server_qps` sets each configured level. |
| `TTFT_SLA_MS` | `500` | TTFT target used by open-loop reporting unless overridden by protocol or `--ttft-target`. |
| `TPOT_SLA_MS` | `50` | TPOT target used by open-loop reporting unless overridden by protocol or `--tpot-target`. |
| `LOADGEN_OUTDIR` | `loadgen_out` | Scratch directory for MLPerf LoadGen logs; the suite places these outside `docs/results`. |
| `HARNESS_GIT_REV` | unset | Set by the in-cluster launcher, which has no `.git`. |

## Protocols

A protocol is the unit of comparability: it fixes the harness, the prompt profile, the swept axis and the output budget, and every result file opens with the lock that names them. Two numbers may be compared only if they came from the same protocol.

This table is the reference summary. [Which benchmark to run](#which-benchmark-to-run) covers what each protocol returns and does not return.

| Protocol | Harness | Sweeps | What it answers |
| -------- | ------- | ------ | --------------- |
| `corpus` | `throughput.py` | concurrency 1→512 | Sizing: tokens/sec and req/sec against in-flight depth, 150-token outputs. The canonical sweep. |
| `reasoning` | `throughput.py` | concurrency 1→512 | The same sweep with a reasoning-length output budget. Reads higher tok/s than `corpus` because longer decodes amortise per-request overhead, which makes it a different workload rather than a better result. |
| `latency` | `latency.py` | concurrency 1→512 | TTFT / TPOT / ITL, streaming. |
| `latency_longctx` | `latency.py` | input length 4k→28k | First-token cost against context length, at fixed concurrency 8. |
| `latency_sharedprefix` | `latency.py` | concurrency 1→512 | Cache-hit behavior on a shared prefix. Inflated by construction, so not comparable with a unique-prompt baseline. |
| `server_qps` | `arrival_rate.py` | target arrival rate 1→16 QPS | Interactive capacity: does the server hold TTFT p99 ≤ 500 ms and TPOT p99 ≤ 50 ms at a given request rate. Open loop, MLPerf LoadGen. |
| `corpus_aiperf` | `aiperf_harness.py` | concurrency 1→512 | **Calibration only.** The `corpus` workload driven by NVIDIA AIPerf instead of this repo's client, so two independent clients can be checked against each other. |
| `agentx` | `aiperf_harness.py` | one trace replay | Agentic coding traffic: 40 multi-turn sessions with think-time gaps and KV-cache-faithful prefix structure, replayed on its recorded timeline. |
| `trace_replay` | `aiperf_harness.py` | one trace replay | Your own Mooncake-format trace, replayed with its real arrival timing. Pass it with `--trace`. |

## What may be compared with what

Each of these rules reflects a case where two figures are not directly comparable.

- **`corpus_aiperf` numbers are not corpus results.** The protocol exists to check one client against the other, and the lock records which client ran. On an MoE model with NVFP4 weights the two clients agreed within measurement noise at concurrency 512, which supports the canonical `corpus` figures. Quoting the calibration run in place of a `corpus` result reports a different client than the one the protocol pins.
- **AIPerf streaming metrics do not share a column with `latency.py` metrics.** They are different quantities: AIPerf separates client-side queueing from time-to-first-token, `latency.py` folds them together, and the gap reached 41% at concurrency 32 on the same server. `latency.py` is the source for published TTFT/TPOT figures; the AIPerf-backed protocols report their own streaming columns and are read on their own terms.
- **Trace replays compare only against replays of the same trace.** The lock records the trace path and its SHA-256 for this reason. A regenerated trace is a new workload even when the generator and seed are the same.
- **Open-loop figures do not share a table with the concurrency sweeps.** Arrival rate and in-flight depth are different independent variables, and `openloop_tokens_per_sec` is named so the difference is visible in the column itself.
- **A flagged cell is not a usable result.** Any cell whose spread across reps exceeds 10% is marked `distrust this cell` in the file, and any level that lost requests excludes its whole rep from the medians. Both are printed in the file rather than suppressed.

## Trusting the open-loop verdict

`server_qps` is the one protocol whose pass/fail verdict comes from outside this repo, and LoadGen's verdict alone is not sufficient:

- **A percentile has to be certifiable.** LoadGen certifies a p99 only when the query count clears `h + t`: 459 queries with no SLA violations, 662 with one. Below that it reports `UNDERSAMPLED`, which is a sample-size statement rather than an SLA violation. `MIN_QUERY_COUNT` defaults above the floor for this reason.
- **LoadGen can report a pass on a server that returned no successful responses.** Failed requests are completed to LoadGen, which has no failed-query concept, so they enter its latency record as instant responses and improve its percentiles. Measured against a killed server with 84% of requests failing, LoadGen reported `sla=MET, loadgen_result=VALID`. The `REQUEST_SUCCESS` line every harness emits catches this: a level below 95% success excludes its rep, and the aggregate names which and why. `eks-bench selftest-openloop` re-creates that scenario on every run of the checks.

---

## Which benchmark to run

A **harness** is the client program that generates load. A **protocol** is a named configuration that selects one harness and fixes its settings: the prompt profile, the swept axis, the output budget and the request counts. Every result file opens with a lock recording which protocol and harness produced it.

One invocation runs one protocol. To collect both a throughput figure and an interactive-capacity figure, run the tool twice against the same server, changing `--protocol`. Results are comparable only between runs of the same protocol.

`corpus` and `reasoning` run a second pass automatically. Their harness is non-streaming and cannot measure time to first token, so the suite runs the streaming latency harness after the sweep and prints its table in the same file. No other protocol has a companion pass.

### Start here

| Question | Protocol | Notes |
| -------- | -------- | ----- |
| Tokens per second at a given concurrency | `corpus` | Closed-loop sweep. Includes the companion latency pass. |
| Tokens per second for a reasoning model | `reasoning` | 1024-token output cap. The 150-token cap in `corpus` truncates reasoning output. |
| Whether latency targets hold at a given request rate | `server_qps` | Open-loop arrivals. |
| Time to first token and inter-token latency | `latency` | Streaming, across the concurrency sweep. |
| Time to first token as input length grows | `latency_longctx` | Concurrency fixed at 8. |
| Effect of prefix caching on a shared prefix | `latency_sharedprefix` | Requires prefix caching enabled server side. |
| Agreement between two independent clients | `corpus_aiperf` | Reruns the `corpus` workload through NVIDIA AIPerf. |
| Behavior under multi-turn agentic traffic | `agentx` | Ships a fixed trace. |
| Behavior under a recorded production trace | `trace_replay` | Requires `--trace`. |

### Protocols at a glance

Generated from `PROTOCOLS` in [`app/bench/src/config.py`](../../app/bench/src/config.py), the same dict that supplies the `--protocol` choices.

| Protocol | Harness | Swept axis | Output cap | Prompt profile | Companion latency pass |
| -------- | ------- | ---------- | ---------- | -------------- | ---------------------- |
| `corpus` | `throughput` | concurrency 1, 8, 32, 128, 256, 512 | 150 tok | `legacy_corpus` | yes |
| `reasoning` | `throughput` | concurrency 1, 8, 32, 128, 256, 512 | 1024 tok | `legacy_reasoning` | yes |
| `latency` | `latency` | concurrency 1, 8, 32, 128, 256, 512 | 256 tok | `coding_completion` | no |
| `latency_sharedprefix` | `latency` | concurrency 1, 8, 32, 128, 256, 512 | 256 tok | none | no |
| `latency_longctx` | `latency` | input length 4k, 8k, 16k, 28k, concurrency fixed at 8 | 256 tok | none | no |
| `server_qps` | `arrival_rate` | arrival rate 1, 2, 3, 4, 5, 6, 8, 16 QPS | 256 tok | `coding_completion` | no |
| `corpus_aiperf` | `aiperf` | concurrency 1, 8, 32, 128, 256, 512 | 150 tok | `legacy_corpus` | no |
| `agentx` | `aiperf_trace` | none, the trace sets the schedule | per record in the trace | none | no |
| `trace_replay` | `aiperf_trace` | none, the trace sets the schedule | per record in the trace | none | no |

The columns each harness reports in every row of its table. The protocol you select determines which of these column sets the run returns.

| Harness | Columns in every row |
| ------- | -------------------- |
| `throughput` | `tokens_per_sec`, `req_per_sec`, `p50_s`, `p99_s` |
| `aiperf` | `tokens_per_sec`, `req_per_sec`, `p50_s`, `p99_s` |
| `latency` | `ttft_p50_s`, `ttft_p99_s`, `tpot_p50_ms`, `tpot_p99_ms`, `itl_p50_ms`, `itl_p99_ms` |
| `arrival_rate` | `achieved_qps`, `ttft_p50_ms`, `ttft_p99_ms`, `tpot_p50_ms`, `tpot_p99_ms`, `openloop_tokens_per_sec`, `input_tokens_median`, `completed`, `failed` |
| `aiperf_trace` | `tokens_per_sec`, `req_per_sec`, `ttft_p50_s`, `ttft_p99_s`, `itl_p50_ms`, `itl_p99_ms` |

`throughput` and `aiperf` share a column set, which is what allows the two clients to be compared on the same workload. `aiperf_trace` reports ITL where `latency` reports TPOT, because AIPerf exports inter-token latency directly. ITL and TPOT are different quantities.

### What each protocol returns

Every protocol writes one result file to `--out-dir`, opening with the protocol lock and the provenance block, then the tables below. Format details in [`docs/results/README.md`](../results/README.md).

#### `corpus`

Closed-loop concurrency sweep with a 150-token output cap. Holds a fixed number of requests in flight at each level.

* **Returns:** the aggregate sweep keyed by concurrency, reporting the median across reps and flagging any cell whose spread exceeds 10 percent; request success per level per rep; achieved in-flight depth against requested depth; and the companion streaming latency table.
* **Does not report:** behavior at a fixed arrival rate. The 150-token cap truncates reasoning-model output.
* **Note:** `legacy_corpus` repeats five fixed prompts. With prefix caching enabled server side, the run can reflect cache hits rather than uncached serving. Check the `Prefix cache hit` line in the provenance block; a cache-free claim requires a zero observed hit rate.

#### `reasoning`

The `corpus` sweep with a 1024-token output cap, unique prompts and a duration floor.

* **Returns:** the same tables as `corpus`, including the companion latency pass. Reported token counts include reasoning tokens.
* **Does not report:** figures comparable to `corpus`. A longer decode amortises prefill across more output tokens, so tokens per second reads higher on the same hardware.

#### `latency`

Streaming sweep across the concurrency levels, with a 256-token output cap.

* **Returns:** TTFT, TPOT and ITL at p50 and p99 per level, sized to a minimum sample count, with warmup waves excluded.
* **Does not report:** tokens per second.
* **Note:** run this in-cluster. A loopback tunnel adds a fixed cost to every first-token sample, so loopback latency runs are refused unless explicitly permitted, and permitting one is recorded in the lock.

#### `latency_sharedprefix`

Streaming sweep in which all requests share a 4000-token prefix.

* **Returns:** the `latency` column set, measured against a cacheable prompt shape.
* **Does not report:** behavior on unique prompts. Do not compare these figures against a unique-prompt baseline.
* **Note:** prefix caching must be enabled server side.

#### `latency_longctx`

Streaming measurements across input lengths of 4k, 8k, 16k and 28k tokens, at fixed concurrency 8.

* **Returns:** the `latency` column set keyed by input length rather than concurrency.
* **Does not report:** scaling with concurrency.

#### `server_qps`

Interactive capacity, and the only open-loop protocol. Requests arrive on a Poisson schedule through MLPerf LoadGen, independently of whether the server keeps up.

* **Returns:** per arrival-rate level, the achieved QPS against the offered rate, TTFT and TPOT at p50 and p99, `openloop_tokens_per_sec`, median input size, and completed against failed counts. Also a targets section naming which target binds first, and LoadGen's certification state per level.
* **Does not report:** throughput comparable to the concurrency sweeps. Arrival rate and in-flight depth are different independent variables, which is why the column is named `openloop_tokens_per_sec`.
* **Implementation dependency:** `requirements-loadgen.txt`. See [Trusting the open-loop verdict](#trusting-the-open-loop-verdict) before quoting a level as sustainable.

#### `corpus_aiperf`

The `corpus` workload driven by NVIDIA AIPerf rather than this repo's client, for cross-client comparison on the same server.

* **Returns:** the `throughput` column set, measured by AIPerf.
* **Does not report:** a `corpus` result. The lock records which client ran.
* **Implementation dependency:** `requirements-aiperf.txt`.

#### `agentx`

Agentic coding traffic: multi-turn sessions with think-time gaps between turns and prefix structure that behaves like a real KV cache would, replayed on its recorded timeline.

* **Returns:** throughput and streaming columns for the replay. Output budgets come per record from the trace rather than from a single cap.
* **Does not report:** anything keyed to concurrency, since the trace sets the schedule. Not comparable to the sweep protocols.

#### `trace_replay`

Replay of a Mooncake-format trace supplied with `--trace`, using its recorded arrival timing.

* **Returns:** the same columns as `agentx`.
* **Does not report:** figures comparable to a replay of a different trace. The lock records the trace path and a SHA-256 of the file, so two results claiming the same trace can be checked. A regenerated trace is a different workload even with the same generator and seed.
* **Note:** this protocol pins no trace, so `--trace` is required. Keep traces under `app/bench/data/traces/` so the in-cluster payload includes them.

### Running more than one

Bring the server up once and run protocols against it in sequence. Only `--protocol` changes between runs, plus `--profile` where the protocol takes one.

- Throughput and latency for a standard model: `corpus`.
- Throughput and latency for a reasoning model: `reasoning`.
- Cross-client agreement on a throughput figure: `corpus` plus `corpus_aiperf`.
- Whether latency targets hold at a request rate: `server_qps`.
- Prefill cost against context length: `latency_longctx`.
- Prefix cache behavior: `latency_sharedprefix`.
- Agentic or recorded traffic: `agentx` or `trace_replay`.

Each protocol runs its own sweep and consumes its own GPU time. The companion latency pass roughly doubles the duration of a throughput run; `--no-latency` skips it.

### Common errors

- Quoting `openloop_tokens_per_sec` from `server_qps` as a throughput figure. It measures whether a latency target holds at a request rate.
- Comparing `corpus` against `reasoning`. The output caps differ.
- Quoting `corpus_aiperf` as a `corpus` result.
- Running a latency protocol through a port-forward.
- Running `latency_sharedprefix` without prefix caching enabled server side.
- Publishing `corpus` as cache-free while prefix caching is enabled against the repeating `legacy_corpus` profile, or without provenance showing a zero observed hit rate.
