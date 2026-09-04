# Batch vs. real-time inference

**Batch offline inference** processes a pre-collected set of requests—documents to summarize, datasets to annotate, logs to classify—where no user is waiting for an individual response. Optimize completed work per unit time and cost while still meeting the job deadline.

**Real-time online inference** serves requests whose users or downstream systems are waiting. Two streaming metrics dominate:

- **TTFT (time to first token):** queueing plus prompt prefill and scheduling before the first token arrives.
- **TPOT (time per output token):** the pace of generation after the first token. ITL (inter-token latency) is related but is not interchangeable with TPOT.

Throughput matters in both cases, but in a different role. Batch work seeks a high sustainable token rate subject to correctness and deadline constraints. Real-time work seeks latency-SLO compliance at the required arrival rate. High GPU utilization is not by itself evidence of success or queueing; interpret it with offered versus achieved load, request success, and the latency distribution.

## The same server, two views

A closed-loop concurrency sweep normally has a recognizable shape: aggregate throughput rises as batching improves, then approaches a plateau, while queueing, TTFT, and often TPOT worsen as the server saturates. The useful operating point is workload- and SLO-specific; there is no repository-wide concurrency value that represents it.

The `corpus` and `reasoning` protocols collect throughput with the non-streaming `throughput.py` implementation harness and, unless disabled, run a separate streaming-latency pass. These passes target the same server and sweep but send different requests. Read their throughput and latency curves side by side rather than as metrics from a single request.

Run the protocol on the selected recipe and preserve the complete curve. A single peak-throughput or minimum-latency cell omits both the knee and the tradeoff needed for deployment sizing.

---

## Saturation vs. responsiveness

- **Batch work** optimizes aggregate progress against a deadline. Per-request latency can be high if total throughput and reliability are acceptable.
- **Real-time work** optimizes against a per-request response-time target. More batching can improve efficiency, but not if queueing pushes requests outside that target.

The distinction is one of objective and arrival model, not a different class of server.

## Which protocol answers which question

A protocol fixes the harness, prompt shape, output budget, and swept axis. The runtime registry in `app/bench/src/config.py` is authoritative; [`REFERENCE.md`](./REFERENCE.md#protocols-at-a-glance) presents its current generated table and exact levels.

| Protocol | Implementation harness | Axis and default shape | Use it for |
|---|---|---|---|
| `corpus` | `throughput.py` | Canonical concurrency sweep; `legacy_corpus`; short capped output | Historical cross-instance continuity and closed-loop throughput. The five fixed prompts repeat, so server cache configuration and provenance determine whether this is a cache-hit result. The output cap truncates reasoning models. |
| `reasoning` | `throughput.py` | Canonical concurrency sweep; `legacy_reasoning`; reasoning-length output cap | Historical closed-loop reasoning-model capacity. Reported tokens include reasoning tokens. |
| `latency` | `latency.py` | Canonical concurrency sweep; `coding_completion` | Streaming TTFT, TPOT, and ITL as in-flight depth changes. |
| `latency_longctx` | `latency.py` | Configured input-length sweep at fixed concurrency | First-token behavior as unique prompt length grows; not concurrency capacity. |
| `latency_sharedprefix` | `latency.py` | Canonical concurrency sweep; generated shared prefix | A deliberately cache-seeking experiment. Prefix caching must be enabled, and results are not a unique-prompt baseline. |
| `server_qps` | `arrival_rate.py` | Configured offered-QPS levels; `coding_completion` | Open-loop capacity under Poisson arrivals: whether success, achieved-rate, TTFT, and TPOT requirements hold. |
| `corpus_aiperf` | `aiperf_harness.py` | Canonical concurrency sweep; `legacy_corpus` | Cross-client calibration against `corpus`, not a replacement result. |
| `agentx` | `aiperf_harness.py` trace mode | Committed trace schedule and per-record output budgets | Multi-turn agentic traffic with think-time and cache-faithful prefix structure. |
| `trace_replay` | `aiperf_harness.py` trace mode | User-supplied trace schedule and per-record budgets | Replay of recorded traffic; compare only replays of the identical trace digest. |

Closed-loop concurrency, open-loop arrival rate, context length, and trace time are different independent variables. Their numbers do not belong in one comparative table. In particular, `openloop_tokens_per_sec` is not the closed-loop `tokens_per_sec` metric.

Use the public CLI for operator commands:

```text
python3 app/cli.py launch --help
python3 app/cli.py suite --help
python3 app/cli.py validate-manifests <model>/<recipe>.yaml
```

`python3 app/cli.py launch ... --protocol <name>` is the normal in-cluster path for latency claims; `suite` is useful when the endpoint and provenance path are already available. The CLI dispatches to implementation harnesses such as `throughput.py`, `latency.py`, `arrival_rate.py`, and `aiperf_harness.py`. Those names are referenced below to explain metric semantics; the CLI is the supported entry point.

## Prompt length and aggregate prefill pressure

Prompt length affects prefill cost at every load, and aggregate pressure grows with the number of in-flight requests. A useful scheduling approximation is

```text
prefill_pressure = N * L / B
approximate_prefill_iterations = ceil(N * L / B)
```

where:

- `N` is in-flight request count,
- `L` is the measured input tokens per request, and
- `B` is the server's per-iteration batched-token budget.

The rough budget crossing is `N ≈ B / L`: longer prompts reach it at lower concurrency. This is a diagnostic model, not a latency predictor. Chunked prefill, decode interleaving, sequence-length variation, KV capacity, cache state, scheduler policy, model compute, and hardware all affect actual TTFT. Measure the curve rather than converting the ratio directly into milliseconds.

A TTFT figure without its measured input shape is not portable. Nominal profile sizes are labels; the result's measured wire-token count for the model and chat template is the value to report.

## Prompt profiles

Profiles pin both input text and output cap. Choose one that resembles the intended workload with `--profile`; the serving recipe does not determine request shape. The canonical files and tokenizer caveats are documented in the [prompt-profile reference](../../app/bench/data/prompt_profiles/README.md).

| Profile | Prompts | Nominal input | Output cap | Intended shape |
|---|---:|---:|---:|---|
| `chat_short` | 32 | 128 | 512 | Single-turn question, no history |
| `chat_typical` | 32 | 512 | 1,024 | Conversational turn with short history |
| `coding_completion` | 24 | 2,048 | 256 | Inline completion with surrounding-file context |
| `rag_document` | 16 | 4,096 | 512 | Retrieved passages plus a question |
| `coding_assistant` | 12 | 8,192 | 1,024 | Coding chat with open files and history |
| `coding_repo_context` | 8 | 16,384 | 1,024 | Large repository or multi-file context |
| `legacy_corpus` | 5 | 9 | 150 | Historical continuity only |
| `legacy_reasoning` | 32 | 41 | 1,024 | Historical continuity only |

The representative profiles differ at position zero except for their declared intentional duplicates. `legacy_corpus` is different: its five prompts are reused throughout the run. Consequently, the corpus is not automatically cache-free. Whether it measures cold work, warmed cache hits, or a mixture depends on prompt selection, server-side prefix caching, warmup/history, and the provenance recorded with the result.

`latency_sharedprefix` addresses the opposite question. It generates a byte-identical shared prefix and requires prefix caching so the run can characterize cache-hit behavior. Compare cache experiments only when prefix shape, warmup, server configuration, and observed cache state are known, and not directly against unique-prompt latency.

## Reasoning output and user-visible latency

Reasoning tokens are decode work and may delay useful visible content, whether or not the API exposes a literal reasoning trace. Include that time and token budget in the interactive SLO. If the selected model and server support alternate reasoning modes, treat each mode as a distinct configuration and remeasure it rather than assuming reasoning can always be disabled.

---

## Tools and provenance

The public CLI applies protocol locks, runs the accuracy gate, invokes the selected harness, and captures provenance. The implementation pieces have distinct roles:

- **vLLM** is the serving stack under test. Its version, image, arguments, environment, and model configuration belong to the selected manifest and result provenance.
- **`throughput.py`** drives non-streaming closed-loop throughput protocols; **`latency.py`** streams responses for TTFT/TPOT/ITL.
- **`arrival_rate.py`** uses MLPerf LoadGen for open-loop scheduled arrivals and adds the repository's success and achieved-rate checks.
- **NVIDIA AIPerf** provides the independent corpus calibration client and fixed-schedule trace replay.
- **NVIDIA Model Optimizer** produces quantized checkpoints where required; precision changes still require the accuracy gate.
- **NVIDIA Nsight Systems**, **nccl-tests**, and the CUDA Samples pairwise bandwidth test diagnose compute and interconnect ceilings after a benchmark locates them. See the [profiling runbook](../runbooks/studies/g7e.12xlarge-profile-gpus-nsight-systems-runbook.md) and [`experiments/`](../../experiments/).

Recipe and image inventory changes independently of this guide. For current models, instance types, images, parallelism, workload intent, and runbooks, use the [`vllm-recipes` catalog](../../vllm-recipes/README.md). Validate the exact manifest before deployment and preserve its rendered command in provenance.

## vLLM tuning hypotheses, by workload

Treat each setting below as a hypothesis to test on the selected model, image, backend, precision, prompt profile, and hardware. Change one variable at a time, keep the protocol lock fixed, and rerun the accuracy gate where the change can affect output.

### `--max-num-seqs`

This caps how many sequences the scheduler can consider concurrently. More capacity can improve batch fill, but a larger active set can increase per-step work, queueing, and KV demand. For real-time service, set it high enough not to impose a ceiling below the chosen operating point, and measure latency and memory behavior before raising it further.

### `--max-num-batched-tokens`

This is the per-iteration token budget across scheduled sequences. A larger budget can improve GPU fill while making an iteration longer; a smaller budget can improve scheduling responsiveness while sacrificing throughput. Its interaction with prompt length and chunked prefill is workload-specific, so sweep it around the target profile rather than adopting a universal value.

### `--enable-chunked-prefill`

Chunking lets long prefills interleave with decode work and can reduce head-of-line blocking. The benefit and overhead depend on the vLLM version, scheduler, model, and prompt mix. Validate both latency distribution and throughput for the exact recipe rather than treating the flag as universally required.

### `--enable-prefix-caching`

Prefix caching helps only when requests repeat reusable **leading token sequences** after the relevant cache entries are warm. It changes the workload being measured. Record server configuration and cache provenance, distinguish cold and warm behavior, and use `latency_sharedprefix` only for the deliberate cache-hit question. A `legacy_corpus` run with caching enabled is cache-sensitive because its prompts repeat; it is not a unique-prompt baseline.

### `--gpu-memory-utilization`

This controls startup memory reservation and therefore the space available for weights, runtime allocations, and KV cache. A higher value may increase sequence capacity but reduces allocation headroom. Traffic does not change the configured fraction itself, so diagnose request failures from actual KV exhaustion or runtime allocation behavior rather than assuming a lower fraction alone prevents them.

### Quantization, `--kv-cache-dtype`, and `--calculate-kv-scales`

Lower-precision weights or KV cache can change memory footprint, kernel availability, throughput, latency, and output quality. Runtime KV-scale calculation is a recipe-specific alternative to stored or calibrated scales, not a general rule for unpredictable traffic. Compare candidate configurations under the same profile and protocol, and require correctness plus performance evidence before adopting one.

### Parallelism

Tensor, pipeline, and expert parallelism trade local memory/compute against communication. The right topology is constrained first by model fit, then by measured latency and throughput on the instance's actual links. Higher tensor parallelism can reduce per-GPU memory traffic but adds collectives; expert parallelism can improve expert placement while adding all-to-all communication. Neither has a universal batch or real-time verdict.

### Speculative decoding

A draft path can reduce sequential decode steps when proposals are accepted, but it also consumes compute and may select different kernels. Measure acceptance, TTFT, TPOT, throughput, and accuracy for the exact model/backend pair. Do not infer the end-to-end gain from acceptance rate alone.

### Attention backend and request logging

Attention-backend compatibility is scoped to the selected recipe, image/version, model, dtype, and GPU architecture. Inspect the manifest and startup logs. [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md) documents a tested XFormers/sm_120 failure and workaround, which is specific to that stack rather than a requirement for every g7e recipe or evidence that one backend is correct across images.

Per-request logging can add overhead and jitter at high request rates. Preserve enough diagnostics for failures, and measure the effect of any logging-level change rather than assuming all recipes use the same setting.

---

## From a serving recipe to a real-time configuration

1. **Select the current recipe and image.** Read its manifest and catalog entry, then run `python3 app/cli.py validate-manifests <model>/<recipe>.yaml`.
2. **Define the service objective.** Pin the prompt profile or trace, output budget, arrival pattern, TTFT/TPOT percentiles, minimum success rate, required achieved rate, and capacity headroom.
3. **Establish an accuracy-gated baseline.** Run in-cluster with `python3 app/cli.py launch ...` and preserve the protocol lock and provenance.
4. **Measure the closed-loop latency curve.** Use `latency` for a representative profile; use `latency_longctx` or `latency_sharedprefix` only for their specialized questions.
5. **Change one serving variable at a time.** Scheduling, caching, precision, backend, and parallelism changes can interact, so isolate them before combining winners.
6. **Verify sustainable arrivals.** Run `server_qps` to find the highest offered rate that meets latency, success, achieved-rate, and certification requirements.
7. **Repeat the correctness gate and retain the full curve.** A faster configuration that changes answers, or that holds only over a short sample, is not a deployment result.

`legacy_corpus` is not suitable for validating an interactive service: its nominal input is historical, its output cap can truncate reasoning, and its repeated prompts make it cache-sensitive.

## Metrics by workload

| Concern | Batch offline | Real-time online |
|---|---|---|
| Primary objective | Completed tokens or requests per cost and deadline | TTFT/TPOT SLO at required arrival rate |
| Load model | Finite queued work or closed-loop saturation study | Open-loop arrivals, plus closed-loop diagnosis |
| Prompt choice | Representative batch profile; historical protocols only for continuity | Representative interactive profile or trace |
| Cache treatment | Match the real batch's prefix reuse and report it | Report cold/warm mix and reusable-prefix behavior |
| Capacity evidence | Sustainable token/request rate, success, job completion time | Sustainable QPS, achieved/offered rate, success, tail latency |
| Relevant protocols | Representative-profile suite; `corpus`/`reasoning` for historical continuity | `latency` for the curve, `server_qps` for capacity; specialized protocols as needed |

## Picking an operating point and sizing replicas

For interactive service, first choose a latency-feasible operating point on the representative closed-loop curve. Let `C*` be an in-flight depth that meets TTFT/TPOT and success requirements with headroom, and let `W*` be its measured mean request residence time in seconds. The implied per-replica request rate is approximately

```text
mu_est = C* / W*          # requests/second
```

This is Little's-law sizing, not proof of a tail-latency SLO. Prefer the directly measured open-loop sustainable rate `mu_sustainable` from `server_qps`: the highest offered rate that also meets achieved-rate, success, latency, and certification requirements.

For peak expected arrival rate `lambda_peak` and dimensionless headroom factor `H > 1`:

```text
replicas = ceil(H * lambda_peak / mu_sustainable)
```

If only the closed-loop operating point is available, the provisional form is:

```text
replicas = ceil(H * lambda_peak * W* / C*)
```

Then validate that replica count with open-loop traffic and the real load-balancing path. Dividing requests per second by concurrency alone is dimensionally incomplete.

For a batch deadline, derive the required work rate first. If `T_total` output tokens must complete within `D` seconds and one replica sustains `tau` output tokens per second on the representative workload:

```text
required_token_rate = T_total / D
replicas = ceil(H * required_token_rate / tau)
```

Still check request failures, input/prefill work, timeouts, and end-to-end job completion. Peak token throughput alone does not guarantee the batch deadline.

---

*See also: [`REFERENCE.md`](./REFERENCE.md) for current commands, protocol definitions, and comparability rules; [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md) for diagnosed failures and scoped fixes; the [prompt-profile reference](../../app/bench/data/prompt_profiles/README.md) for canonical request shapes; and the [`vllm-recipes` catalog](../../vllm-recipes/README.md) for current recipe and image inventory.*
