# Architecture

What this repo is, how its pieces fit together, and the integrity mechanisms that make a result trustworthy. See the [root README](../../README.md) for the Quickstart and the Prerequisites needed to run a benchmark.

## What this is

`serving-sweep` is a measurement instrument, not a load generator. A load generator produces a number. This produces a number plus the evidence that the number means what it says. "How fast is my endpoint?" is the wrong question. These are the questions that decide which compute you need, and how much of it:

- Will this model even fit, and at what concurrency does it stop being usable?
- With 128 concurrent users, what does each one see, rather than what does the fleet total?
- At 10 requests per second arriving, does the SLA still hold?
- Which serving flag actually moved the number: precision, attention backend, KV cache dtype, prefix caching, parallelism?
- Can this result be compared against one taken a year from now?
- Is the server serving the configuration it claims to?

## System architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OPERATOR (laptop, CPU only)                     │
│ python3 app/cli.py is the operator CLI for the measurement path.    │
│ app/utils.py bootstraps .venv; app/cli.py execs                     │
│ app/bench/src/main.py, which dispatches validation, accuracy,       │
│ launch, and suite commands.                                         │
│ launch packs the harness into a ConfigMap, renders suite-job.yaml,  │
│ streams Job logs, and writes results under docs/.                   │
└─────────────────────────────────────────────────────────────────────┘
           │ kubectl apply -f                │ python3 app/cli.py launch
           │ vllm-recipes/<model>/<r>.yaml   │ (the bench Job below)
           ▼                                 ▼
 ┌───────────────────────────────┐   ┌───────────────────────────────┐
 │     vLLM SERVER, GPU nodes    │   │      BENCH JOB, CPU node      │
 │ one recipe = one server:      │   │ python:3.12, pip installs     │
 │ Deployment / StatefulSet /    │   │ aiohttp and MLPerf LoadGen    │
 │ LeaderWorkerSet, EFA for      │   │ suite: protocol lock, reps,   │
 │ multi-node TP and PP,         │   │ medians, accuracy gate,       │
 │ serves /v1 on cluster DNS     │   │ AIPerf as a second client     │
 └───────────────────────────────┘   └───────────────────────────────┘
           ▲ mounts /models                  │ logs streamed to the
           │                                 │ operator, extracted
           │                                 │ into docs/
 ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
 │  WEIGHT STAGING JOB │ │     QUANTIZE JOB    │ │       NSYS POD      │
 │ CPU node            │ │ GPU node            │ │ GPU node            │
 │ model-downloads/    │ │ app/quantize image  │ │ app/vllm-nsys image │
 │ one Job per model   │ │ nvidia-modelopt     │ │ nsys wraps its own  │
 │ and precision       │ │ reads BF16, writes  │ │ vllm serve, load    │
 │ writes the claim    │ │ FP8/INT8/AWQ        │ │ driven in-pod       │
 └─────────────────────┘ └─────────────────────┘ └─────────────────────┘
           │ write                                    │ .nsys-rep
           ▼                                          ▼
 ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
 │       WEIGHTS       │ │       RESULTS       │ │        TRACES       │
 │ FSx claim /models   │ │ docs/results/*.txt  │ │ /traces/*.nsys-rep  │
 │ staged or quantized │ │ docs/accuracy/*.json│ │ in-pod, then S3     │
 └─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

An ordinary `python3 app/cli.py launch` dispatches `suite` in the Job. `python3 app/cli.py launch --capture-accuracy-reference ...` instead dispatches `accuracy --write-reference`; capture and benchmark checking remain separate runs.

## Project structure

```
serving-sweep/
├── vllm-recipes/                      # THE vLLM SERVE RECIPES handed to customers.
│                                      #   <model>/<quant>-<instance>[-<config>].yaml, grouped
│                                      #   by model. Each opens with one line saying what it
│                                      #   serves and when to use it.
├── model-downloads/                   # WEIGHT STAGING JOBS, one per model and precision.
│                                      #   Recipes wait for weights, they do not fetch them.
│                                      #   Apply the Job for your model first, once per
│                                      #   filesystem. See its README.
├── app/
│   ├── cli.py                         # Supported operator CLI; styled help, then execs dispatcher.
│   ├── utils.py                       # Bootstraps .venv and resolves app/bench/src/main.py.
│   ├── requirements-cli.txt           # CLI implementation dependency; installer consumes it.
│   ├── quantize/                      # GPU quantization tool (nvidia-modelopt)
│   │   ├── Dockerfile                 # Quantizer image (sm_89+)
│   │   ├── requirements.txt           # nvidia-modelopt[torch], pinned
│   │   └── src/
│   │       ├── config.py              # Quantization-specific env vars
│   │       ├── quantize.py            # FP8/INT8/AWQ quantization via nvidia-modelopt
│   │       └── fix_amax.py            # Repair pass: rename _amax → weight_scale for vLLM
│   ├── bench/                         # Benchmark client (async HTTP, CPU-only)
│   │   ├── Dockerfile                 # Benchmark image
│   │   ├── suite-job.yaml             # Job template `launch` renders to run in-cluster
│   │   ├── requirements.txt           # Core harness implementation dependency (aiohttp)
│   │   ├── requirements-aiperf.txt    # AIPerf adapter implementation dependency
│   │   ├── requirements-loadgen.txt   # Open-loop implementation dependency
│   │   ├── requirements-tokenizer.txt # Tokenizer-check implementation dependency
│   │   ├── src/
│   │   │   ├── main.py                # Benchmark dispatcher and command implementations.
│   │   │   ├── config.py              # EVERY env var + the measurement protocols
│   │   │   ├── stats.py               # Generic helpers (stats, hashing, git, subprocess)
│   │   │   ├── client.py              # Shared request/SSE/reporting logic
│   │   │   ├── workload.py            # Prompt supply: pinned profiles, then generated
│   │   │   ├── throughput.py          # Closed loop, non-streaming: tokens/sec, req/sec
│   │   │   ├── latency.py             # Closed loop, streaming: TTFT / TPOT / ITL
│   │   │   ├── arrival_rate.py        # Open loop (MLPerf LoadGen): SLA at a target QPS
│   │   │   ├── aiperf_harness.py      # Second client (NVIDIA AIPerf) behind the same seam
│   │   │   ├── aiperf_profiles.py     # Pinned profiles -> AIPerf custom-dataset JSONL
│   │   │   ├── accuracy.py            # Correctness gate: fingerprint + GSM8K score
│   │   │   ├── suite.py               # Protocol-locked runner: reps, medians, protocol lock
│   │   │   ├── launcher.py            # `launch`: suite or reference capture over cluster DNS
│   │   │   ├── build_profiles.py      # Generate app/bench/data/prompt_profiles/ (deterministic)
│   │   │   ├── verify_profiles.py     # Measure profiles against a real tokenizer
│   │   │   ├── fetch_eval_data.py     # Pin the GSM8K subset the accuracy gate scores on
│   │   │   ├── mock_server.py         # Local fake server for the checks below. Hostile on
│   │   │   │                          #   purpose: generates to EOS when a request carries
│   │   │   │                          #   no token cap, and injects failures on demand
│   │   │   ├── verify_aiperf.py       # Regression checks for the AIPerf adapter
│   │   │   ├── verify_openloop.py     # Regression checks for the LoadGen harness
│   │   │   ├── verify_accuracy_gate.py # Regression checks for the correctness gate
│   │   │   └── validate_manifests.py  # Prove recipes pass the flags they claim to
│   │   └── data/                      # What the harness reads, owned by this app so the
│   │       ├── prompt_profiles/       #   image is self-contained: pinned prompt text,
│   │       ├── traces/                #   pinned timestamped traces for trace replay,
│   │       ├── gsm8k_subset.jsonl     #   the 200 problems the gate scores, and the
│   │       └── ATTRIBUTION.md         #   upstream digest and selection rule for them
│   └── vllm-nsys/                     # WHERE STEP TIME GOES (see Profiling section)
│       ├── Dockerfile                 # vLLM DLC + Nsight Systems CLI, multi-stage
│       └── profile.sh                 # Launch-wraps `vllm serve` under nsys
├── scripts/
│   ├── install-benchmark-client.sh    # Standard local .venv and benchmark-client setup.
│   ├── install-prerequisites.sh       # Configure EKS namespace, plugin, secret, and node check.
│   ├── push-image.sh                  # Build and push the quantizer image to ECR
│   ├── patch-efa-plugin.sh            # Fixes the EFA device plugin's stale allow-list
│   └── repeat-sweep.sh                # Legacy NCCL noise study helper
├── k8s/examples/                      # EXAMPLES, not runnable. One per purpose, every
│                                      #   identifier a placeholder. Cluster, node groups,
│                                      #   FSx PV/PVC, weight staging, quantization, image
│                                      #   build, diagnostics, profiling. See its README for
│                                      #   what to substitute and the order to apply them.
├── experiments/                       # One-off studies, kept as evidence for a report rather
│   ├── README.md                      #   than as reusable templates. Real cluster names and
│   ├── nccl-tests-p2p-g7e.yaml        #   namespaces on purpose; see its README before
│   └── nccl-tests-p2p-job.yaml        #   changing anything here.
└── docs/
    ├── guides/                        # Architecture, reference, troubleshooting, and methodology.
    ├── reports/                       # Written analyses. Gitignored; README.md documents
    │   └── README.md                  #   the customer report format and what ships with it
    ├── accuracy/                      # Accuracy references per model. Gitignored; capture your
    │   └── README.md                  #   own, because the gate fails closed without one
    ├── prompt_tokens/                 # Measured raw vs wire token counts per tokenizer.
    │   └── README.md                  #   Gitignored; regenerate with verify-profiles
    ├── runbooks/                      # Model overviews and executable runbooks
    │   ├── README.md                  #   Cross-model runbook index
    │   └── <model>/
    │       ├── README.md              #   Model overview and recipe table
    │       └── <precision>-<instance>-runbook.md
    └── results/                       # One .txt per run, flat, named like the recipes.
        └── README.md                  #   The output format, section by section
                                       #   traces/ is the one subdirectory (gitignored).
```
