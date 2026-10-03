# Project rules

## Always

* Commit every checkpoint-staging Job to `model-jobs/<model-name>/`, named `<quant>-model-download.yaml` for published weights or `<quant>-model-quantize.yaml` for weights this repo produces. Use the same model directory name as `vllm-recipes/` and `docs/runbooks/`.
* Commit any manifest you apply to a cluster. A cluster is not source control, and an ad-hoc `kubectl apply` leaves nothing in git to recover.
* Disable prefix caching in every benchmark (`--no-enable-prefix-caching`), no matter what production runs. It roughly doubled throughput on `chat_terse` and only the uncached number is honest.
* Use the placeholder account (`012345678901`), a region placeholder, and a `SUBSTITUTE:` header line in every committed manifest.
* Make a tool fail on its own bad output. A quantize Job that exits 0 having written an invalid checkpoint is worse than one that crashes, because the next reader inherits it silently.
* Put a verification check outside the artifact it verifies when the artifact could itself be stale. A verifier that only runs inside the producing image cannot catch a producing image that is wrong.
* Report end-to-end latency alongside TTFT and TPOT. A p99 sum is not a summed p99, so the two cannot reconstruct it, and end-to-end is the SLA for any caller that does not render a partial answer.
* Measure open-loop arrival rates when the question is capacity. Closed-loop concurrency is a load-generator dial, and it understated SLA-safe capacity by 26% here.
* Prove a feature is active before attributing a result to it. The n-gram deploy scripts fail if no `vllm:spec_decode` metrics appear, because a silently ignored config reads as "the feature didn't help".
* Run speculative decoding as its own labelled config, never folded into a baseline.
* Capture speculation acceptance as a delta between two counter snapshots. `vllm:spec_decode_*` are cumulative and never reset.
* Verify version-specific claims against the running image rather than inheriting them from another model's runbook. Async scheduling and speculation combine fine on 0.29, even though an earlier runbook said they conflict.
* Prefer a flag over an environment variable for anything that must reach an in-cluster Job. Exported variables do not survive the trip.
* Re-read the KV pool size after changing engine features. Enabling speculation cut it from ~135k to ~72k tokens.
* Pass `--pod` explicitly whenever more than one vLLM pod shares the namespace.
* Keep runbook claims in sync with the recipes they describe, and cite the field each one comes from.
* Write every instruction a runbook needs into that runbook. Link to a file as a reference, never as a substitute for the steps. Cross-references also hide divergence: the FP8 runbook inherited a RAID0 launch template from the BF16 one that its recipes do not need.
* Keep a runbook to its established sections and nothing else: `Capacity sizing`, `Before you begin`, `Quickstart`, `Throughput test`, `Latency test`, `Tear down`, `Settings`. Compare against a sibling runbook before adding a heading.
* Make each Quickstart step one short sentence followed by its command, around 20 words. Reasoning belongs in `Settings` as a bullet, never as prose between the steps.
* Write each `Before you begin` bullet as one line: a bulleted markdown link whose text is an imperative phrase and whose target is the relative path, with at most a short parenthetical caveat. The linked file carries the explanation; the bullet does not.
* Split a function once it needs a second `try`/`except` or a second conditional variable, and compose small named helpers instead. Each check in `verify_checkpoint.py` is its own function returning `(problems, notes)`, and `verify()` only concatenates them, so a new check is a new function rather than another branch.
* Write each deploy-script step as a comment describing it, an `echo`, then the command. Use no `if`, loops, or guard blocks: `set -euo pipefail` already stops the script when a step fails, so a check is a plain command such as `kubectl logs ... | grep "Using AttentionBackendEnum.FLASHINFER backend"`. A long-running command may run in the background, but keep it to one tidy line.
* Open every runbook's Quickstart with the same two steps: export the variables the scripts read, then run `./scripts/install-prerequisites.sh`.
* Stage every checkpoint with the script beside its Job, `model-jobs/<model>/<job>.sh`, written like a deploy script. It applies the Job, waits for it, and fails on its own check, so the runbook step is one command with no separate "confirm" step.
* Fill a manifest's placeholders as the script applies it, `sed ... | kubectl apply -f -`, from the variables Quickstart step 1 exports. Never edit a tracked manifest in place with `sed -i`; the files in git stay untouched.
* Serve every ModelOpt checkpoint this repo produces with `--quantization modelopt`. The plain `fp8` loader misreads a ModelOpt export, and on a BF16 checkpoint it silently quantizes at load time instead of failing.
* Tear down GPU deployments and scale the nodegroup to zero as soon as measurement stops.
* Stage model weights on the node's local NVMe instance store, never on the EBS root volume or FSx. Four Muse-Glimmer copies writing 132 GiB to a 125 MB/s gp3 root disk took ~18 minutes to land and stalled `kubectl exec` for the whole window. Serve from FSx or S3 only for checkpoints the size of GLM 5.2 or Kimi K3. On AL2023 managed nodegroups the instance store stays unmounted unless the launch template sets nodeadm `localStorage.strategy: RAID0`; Karpenter's `instanceStorePolicy: RAID0` does the same. Confirm `df /models` shows the NVMe array before trusting a staging step.

## Never

* Never commit measured results or accuracy gates. Both are gitignored (`.gitignore:43` and `:65`) and belong nowhere in git, including inside markdown.
* Never commit a real account ID, cluster name, namespace, or personal alias.
* Never treat an empty git-history search as proof something never existed. Untracked files leave no trace; check the live cluster and the filesystem too.
* Never rely on a human reading logs as the only guard against a silent failure. If the correct outcome and the broken one both exit 0, that is the bug.
* Never attribute output divergence to a change while the baseline is not bit-reproducible. The FP8 baseline diverges from itself, which made the speculation losslessness check unanswerable.
* Never trust the two-pass reproducibility probe. It runs both passes under identical load and has already mis-certified this model.
* Never promote a workaround into the canonical template. The `/app/src` overlay exists because ECR push was unavailable; rebuilding the image is the real fix.
* Never compare closed-loop concurrency figures against open-loop arrival rates, or ITL across configs with different `--stream-interval`.
* Never carry a null result from a short-prompt profile to a long-prompt one. `--max-num-batched-tokens` was immaterial at 96 tokens and will not be at 8k.
* Never claim a benchmark answers a question it did not measure. N-gram on generic chat says nothing about n-gram on autocomplete, where the output echoes the prompt.
* Never start a long or costly run on your own after asking for a go/no-go. Wait for the answer.
* Never report a diff before ruling out an extraction artifact. A jsonpath key containing dots silently produced empty files and three bogus mismatches.
* Never write comments that record why something changed, compare a file with another, or restate obvious logic. "Identical to X except", "Treat results as their own labelled config" and "calibration is not bit-reproducible, so..." belong in a commit message or nowhere. A comment says what a non-obvious setting does.
