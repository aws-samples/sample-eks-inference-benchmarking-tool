# Project rules

## Always

* Commit quantization jobs to git at `/model-quantize`.
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
* Verify version-specific claims against the running image rather than inheriting them from another model's runbook. Async scheduling and speculation combine fine on 0.29 despite the llama3.3-70b note.
* Prefer a flag over an environment variable for anything that must reach an in-cluster Job. Exported variables do not survive the trip.
* Re-read the KV pool size after changing engine features. Enabling speculation cut it from ~135k to ~72k tokens.
* Pass `--pod` explicitly whenever more than one vLLM pod shares the namespace.
* Keep runbook claims in sync with the recipes they describe, and cite the field each one comes from.
* Write every instruction a runbook needs into that runbook. Link to a file as a reference, never as a substitute for the steps. Cross-references also hide divergence: the FP8 runbook inherited a RAID0 launch template from the BF16 one that its recipes do not need.
* Keep a runbook to its established sections and nothing else: `Capacity sizing`, `Before you begin`, `Quickstart`, `Throughput test`, `Latency test`, `Tear down`, `Settings`. Compare against a sibling runbook before adding a heading.
* Make each Quickstart step one short sentence followed by its command, around 20 words. Reasoning belongs in `Settings` as a bullet, never as prose between the steps.
* Write each `Before you begin` bullet as `- [Imperative phrase](relative/path)` on one line, with at most a short parenthetical caveat. The linked file carries the explanation; the bullet does not.
* Split a function once it needs a second `try`/`except` or a second conditional variable, and compose small named helpers instead. Each check in `verify_checkpoint.py` is its own function returning `(problems, notes)`, and `verify()` only concatenates them, so a new check is a new function rather than another branch.
* Tear down GPU deployments and scale the nodegroup to zero as soon as measurement stops.

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
