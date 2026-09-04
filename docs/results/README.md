# Results

**This directory is empty in a fresh clone, by design.** `docs/results/*.txt` is gitignored: a result is yours, describing the run you performed on your infrastructure. Every file records the endpoint it measured, the cluster context and namespace it read provenance from, the served model path and the pod name — none of which belongs in a public repository. This README is the only tracked file here.

Run a recipe and the files appear. One per measured run, flat, named so a result and the recipe that produced it share a stem and sort next to each other. A result explains the conditions it was taken under, so nothing needs a summary table elsewhere to be readable — and there deliberately isn't one, because the same numbers maintained in two places drift.

## Naming

```
<model>-<quant>-<instance>[-<config>][-<protocol>][-<profile>][-openloop].txt
```

`glm-5.2-nvfp4-p6-b300.48xlarge-corpus-legacy_corpus.txt` is a `corpus` run of the `legacy_corpus` profile against `vllm-recipes/glm-5.2/nvfp4-p6-b300.48xlarge.yaml`.

- **Full instance type, never the family.** `g7e` alone once covered a 2xlarge, a 12xlarge and a 24xlarge in one directory, so three different measurements looked like one.
- **`<config>` only when it breaks a real collision**, naming the axis that differs: `-p2p-phb`, `-p2p-shm`, `-2node`, `-kvscales`, `-aligned`.
- **`-openloop`** is appended for the `server_qps` protocol, because an open-loop result must never be read as a closed-loop one and a filename is more reliable than a warning.
- **No subdirectories, and no `.md` summaries.** `traces/` is the one exception, holding profiler output, and is gitignored too — a single `.nsys-rep` runs to 132 MiB, over GitHub's per-file limit.
- **`-repN` is not part of the convention.** A run writes exactly one file, the aggregate, with every rep's numbers inside it under `per-rep raw values`. Older local files may carry `-rep1`, `-rep2`, `-rep3` from a suite version that wrote each rep separately.

## What writes them

Only `main.py suite`, or `main.py launch`, which runs the suite in-cluster and brings the files back. `--out-dir` is required rather than defaulted on both, because a hardcoded default once wrote one instance's results into another's directory.

A bare harness invocation — `main.py throughput`, `latency`, `arrival_rate` — prints to stdout and must not be redirected in here. It produces a file that looks like every other result while carrying no protocol lock, no accuracy gate, no reps or medians and no provenance, and a reader cannot tell the two apart once both sit in this directory under a conforming name.

## Format

Every file the suite writes opens with a `PROTOCOL LOCK`. The sections below are what it produces today; a run skips some of them by flag, and `--dry-run` prints the lock alone so you can read the conditions before spending GPU time.

If you have older files from a previous version of this repo, some sections will be missing. A section's absence means it was not recorded, not that it passed.

**`PROTOCOL LOCK`** — the conditions, recorded before any measurement so a result cannot be reinterpreted after the fact. Protocol name and harness, harness git revision, server config, instance, endpoint, served model id, rep count, start time, the full benchmark environment with `[OVERRIDDEN by --profile]` marked, the prompt profile, and the accuracy-gate verdict.

Inside the lock, the two input sizes are the part most often misread:

| Line | Means |
|---|---|
| `raw in` | The prompt text alone, measured for this run's tokenizer. Not what the server counts. |
| `wire in` | That text with the model's chat template applied, which is what the server received and billed. **Quote this as the input size.** |

The gap is not small: 9 raw tokens become 21 on the wire for GLM-5.2 and 38 for Qwen2.5. Both come from [`docs/prompt_tokens/`](../prompt_tokens/) and need `--tokenizer`; without it the line says so rather than guessing.

**`AGGREGATE over N/N successful reps`** — the headline table, medians across reps. Closed-loop protocols sweep concurrency `[1, 8, 32, 128, 256, 512]`. `tokens_per_sec` counts **output** tokens only.

**`ACHIEVED IN-FLIGHT DEPTH`** — requested concurrency versus what was actually applied, time-averaged. A level that never reached its requested depth did not measure what its label claims.

**`MEASURED TOKEN SHAPE`** — server-counted input sizes per level, including the chat template. This is the evidence behind `wire in`.

**`REQUEST SUCCESS`** — per level, per rep. Below 95% success a level is unusable, and a transport failure means the client's connection dropped rather than the server failing.

**`per-rep raw values`** — every rep behind every median, so a reader can judge the spread instead of trusting a single number.

**`LATENCY (streaming)`** — present on throughput protocols, a **separate measurement** from a second streaming pass, because a non-streaming run cannot observe TTFT, TPOT or ITL. Different requests from the table above. Read them side by side, never as one row. `--no-latency` skips it.

**`VARIANCE / VALIDITY WARNINGS`** — every cell whose spread exceeds 10% of its median, named individually as "distrust this cell". A long list is normal at p99 and is the file telling on itself rather than a sign the run failed.

**`SERVER PROVENANCE`** — NCCL transport, KV cache sizing and GPU topology read from the serving pod, so a result can show the server was configured as claimed. It records its own absence when it could not be captured, which is why some files carry a `not captured — …` line instead. `--no-provenance` is the deliberate opt-out.

## Other shapes you may produce here

Not everything measured is a suite run, and these do not follow the format above:

- **NCCL microbenchmarks** from [`experiments/`](../../experiments/) report `busbw` in GB/s, measuring the interconnect with no model and no vLLM stack. Never compare their numbers to tokens/sec.
- **Nsight Systems breakdowns** show where step time goes — the split across GEMM, the NCCL collective, attention and sampling — rather than how much throughput was achieved. See the README's Profiling section.
- **Bare harness output.** `main.py throughput`, `latency` and `arrival_rate` print to stdout. Reading them there is fine; capturing them into this directory is not, for the reason above.

The written reports that interpret these results are gitignored too; [`docs/reports/README.md`](../reports/README.md) documents that format. The recipes that produced them are tracked, in [`vllm-recipes/`](../../vllm-recipes/).
