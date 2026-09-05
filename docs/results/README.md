# Results

This directory is empty in a fresh clone. `docs/results/*.txt` is gitignored because each result describes a run on your own infrastructure: every file records the endpoint it measured, the cluster context and namespace it read provenance from, the served model path, and the pod name. This README is the only tracked file here.

Files appear as you run recipes: one per measured run, flat, named so that a result and the recipe that produced it share a stem and sort next to each other. Each result records the conditions it was taken under, so it is readable on its own. There is no summary table aggregating results elsewhere in the repo, because the same numbers maintained in two places tend to diverge.

## Naming

```
<model>-<quant>-<instance>[-<config>][-<protocol>][-<profile>][-openloop].txt
```

For example, `glm-5.2-nvfp4-p6-b300.48xlarge-corpus-legacy_corpus.txt` is a `corpus` run of the `legacy_corpus` profile against `vllm-recipes/glm-5.2/nvfp4-p6-b300.48xlarge.yaml`.

- **Use the full instance type, not the family.** `g7e` alone covers 2xlarge, 12xlarge, and 24xlarge, so three different measurements would appear identical.
- **Include `<config>` only to resolve a genuine collision**, naming the axis that differs: `-p2p-phb`, `-p2p-shm`, `-2node`, `-kvscales`, `-aligned`.
- **`-openloop`** is appended for the `server_qps` protocol, so an open-loop result is not read as a closed-loop one.
- **No subdirectories and no `.md` summaries.** `traces/` is the one exception, holding profiler output, and it is gitignored as well: a single `.nsys-rep` can reach 132 MiB, which is over GitHub's per-file limit.
- **`-repN` is not part of the convention.** A run writes one file, the aggregate, with every rep's numbers inside it under `per-rep raw values`. Older local files may carry `-rep1`, `-rep2`, `-rep3` from a suite version that wrote each rep separately.

## What writes them

`measure-local`, or `measure`, which runs the suite in-cluster and brings the files back. `--out-dir` is required on both rather than defaulted, so results cannot be written to an unintended directory.

A bare harness invocation — `harness-throughput`, `harness-latency`, or `harness-arrival-rate` — prints to stdout. Do not redirect that output into this directory. It produces a file that resembles a suite result but carries no protocol lock, no accuracy gate, no reps or medians, and no provenance, and the two are difficult to distinguish once both sit here under a conforming name.

## Format

Every file the suite writes opens with a `PROTOCOL LOCK`. The sections below are what it produces today; a run may skip some of them depending on flags, and `--dry-run` prints the lock alone so you can review the conditions before spending GPU time.

Files from earlier versions of this repo may be missing sections. A section's absence means it was not recorded, not that it passed.

**`PROTOCOL LOCK`** — the conditions, recorded before any measurement so a result cannot be reinterpreted afterward. Includes protocol name and harness, harness git revision, server config, instance, endpoint, served model id, rep count, start time, the full benchmark environment with `[OVERRIDDEN by --profile]` marked, the prompt profile, and the accuracy-gate verdict.

The lock reports two input sizes:

| Line | Means |
|---|---|
| `raw in` | The prompt text alone, measured for this run's tokenizer. Not what the server counts. |
| `wire in` | That text with the model's chat template applied, which is what the server received and billed. **Quote this as the input size.** |

The difference can be substantial: 9 raw tokens become 21 on the wire for GLM-5.2 and 38 for Qwen2.5. Both values come from [`docs/prompt_tokens/`](../prompt_tokens/) and require `--tokenizer`; without it, the line states that the size was not measured rather than estimating one.

**`AGGREGATE over N/N successful reps`** — the headline table, reporting medians across reps. Closed-loop protocols sweep concurrency `[1, 8, 32, 128, 256, 512]`. `tokens_per_sec` counts **output** tokens only.

**`ACHIEVED IN-FLIGHT DEPTH`** — requested concurrency against what was actually applied, time-averaged. A level that never reached its requested depth did not measure the concurrency its label states.

**`MEASURED TOKEN SHAPE`** — server-counted input sizes per level, including the chat template. This is the evidence behind `wire in`.

**`REQUEST SUCCESS`** — per level, per rep. Below 95% success, a level is unusable. A transport failure means the client's connection dropped, as distinct from the server failing.

**`per-rep raw values`** — every rep behind every median, so the spread is visible alongside the summary figure.

**`LATENCY (streaming)`** — present on throughput protocols. This is a **separate measurement** from a second streaming pass, because a non-streaming run cannot observe TTFT, TPOT, or ITL. It covers different requests than the table above, so read the two side by side rather than as one row. `--no-latency` skips it.

**`VARIANCE / VALIDITY WARNINGS`** — every cell whose spread exceeds 10% of its median, named individually. A long list is common at p99 and indicates variance in the measurement rather than a failed run.

**`SERVER PROVENANCE`** — NCCL transport, KV cache sizing, and GPU topology read from the serving pod, so a result can show how the server was configured. When provenance could not be captured, the file records that with a `not captured — …` line. `--no-provenance` is the explicit opt-out.

## Other output you may produce here

Not everything measured is a suite run, and the following do not follow the format above:

- **NCCL microbenchmarks** from [`experiments/`](../../experiments/) report `busbw` in GB/s, measuring the interconnect with no model and no vLLM stack. Their numbers are not comparable to tokens/sec.
- **Nsight Systems breakdowns** show where step time goes — the split across GEMM, the NCCL collective, attention, and sampling — rather than how much throughput was achieved. See the README's Profiling section.
- **Bare harness output.** `harness-throughput`, `harness-latency`, and `harness-arrival-rate` print to stdout. Reading them there is fine; capturing them into this directory is not, for the reason above.

The written reports that interpret these results are gitignored as well; [`docs/reports/README.md`](../reports/README.md) documents that format. The recipes that produced them are tracked, in [`vllm-recipes/`](../../vllm-recipes/).
