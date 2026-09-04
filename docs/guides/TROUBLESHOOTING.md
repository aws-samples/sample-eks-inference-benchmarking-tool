# Troubleshooting

Indexed by error string. Each entry states the cause and the fix. Where a fix belongs in a manifest, the recipes already carry it.

- [A recipe applies cleanly but serves the wrong configuration](#a-recipe-applies-cleanly-but-serves-the-wrong-configuration)
- [A GPU node reports `Ready` but advertises `nvidia.com/gpu: 0`](#a-gpu-node-reports-ready-but-advertises-nvidiacomgpu-0)
- [`CUDA error: no kernel image is available for execution on the device` on g7e](#cuda-error-no-kernel-image-is-available-for-execution-on-the-device-on-g7e)
- [Crash during CUDA-graph warmup on g7e](#crash-during-cuda-graph-warmup-on-g7e)
- [`KeyError: 'layers.X.mlp.down_proj.weight_scale'` at vLLM startup](#keyerror-layersxmlpdown_projweight_scale-at-vllm-startup)
- [vLLM crashes with `max_num_batched_tokens < max_model_len`](#vllm-crashes-with-max_num_batched_tokens--max_model_len)
- [`PREFLIGHT FAIL: model ... not served`, or all requests return 404](#preflight-fail-model--not-served-or-all-requests-return-404)
- [A latency result looks off, or TTFT is much higher than expected](#a-latency-result-looks-off-or-ttft-is-much-higher-than-expected)

## A recipe applies cleanly but serves the wrong configuration

`kubectl apply` succeeding proves nothing about which flags actually reached `vllm serve`. A stray comment inside a backslash continuation once truncated a `vllm serve` command, and the server came up healthy with every flag below the comment silently discarded — `bash -n` cannot catch this, because the truncated command is still valid shell. The concrete incident: a truncated argv silently dropped `--kv-cache-dtype fp8` and both parsers, and the server reported clean throughput at the wrong precision. It was caught only because the KV cache was 1.82x its expected size.

Run `python3 app/cli.py validate-manifests` before trusting a deploy. It checks the script's syntax, detects the comment-in-continuation signature directly, re-runs the script with `vllm` and the hardware probes stubbed out to compare the argv that actually arrives against the flags the recipe is written to pass, verifies the Service forwards to a port vLLM is listening on, and flags any variable the script reads that nothing defines. Each recipe is checked with its own `env:` values. Given no argument it checks every recipe, which is the pre-deploy gate; name one to check only what you are about to apply.

## A GPU node reports `Ready` but advertises `nvidia.com/gpu: 0`

Three unrelated causes produce this identical symptom and Kubernetes state cannot tell them apart, so the first step is to leave Kubernetes and look at the host with [`pod-diagnostic-probe.yaml`](../../k8s/examples/pod-diagnostic-probe.yaml) running `nvidia-smi -L`:

| What the host says | Cause | Fix |
| ------------------ | ----- | --- |
| `nvidia-smi -L` lists the GPUs | Device plugin missing or unhealthy; the driver is fine | Run `./scripts/install-prerequisites.sh` to install the pinned plugin, or repair the existing plugin installation |
| `No devices found`, and `dmesg` carries `requires use of the NVIDIA open kernel modules` plus `RmInitAdapter failed!` | The AMI shipped the **closed** kernel module for a Blackwell GPU (g7, g7e), which supports only the **open** one | [`job-node-driver-fix.yaml`](../../k8s/examples/job-node-driver-fix.yaml) to repair one node, or [`daemonset-nvidia-open-driver.yaml`](../../k8s/examples/daemonset-nvidia-open-driver.yaml) to keep every node that joins repaired |
| `nvidia-smi` is not installed at all | The node group was built on a non-GPU AMI | Recreate it with `--ami-type AL2023_x86_64_NVIDIA`; do not patch nodes one at a time |

For the middle case, installing the distribution's open-driver package is not sufficient on its own: DKMS fails on kernel 6.12 because it invokes its hardcoded fallback target `make -j2 module` instead of the `modules` target the package's own `dkms.conf` declares, so the module must be built directly. Both manifests above do that. The headers package is `kernel6.12-devel`; plain `kernel-devel` resolves to a different kernel stream and the build will not match. A node repaired this way loses the fix when its instance is replaced (that is what the DaemonSet is for), and a corrected AMI is the durable answer, so re-test the newest AMI release before adopting either workaround.

## `CUDA error: no kernel image is available for execution on the device` on g7e

For recipes pinned to `public.ecr.aws/deep-learning-containers/vllm:0.21.0-gpu-py312`, the tested XFormers path lacked usable `sm_120` kernels. Those affected old-image manifests set `VLLM_ATTENTION_BACKEND=FLASHINFER`, which supplies the needed kernels and supports an FP8 KV cache. This is not a family-wide g7e rule: current `vllm/vllm-openai:v0.26.0` g7e recipes run without a global backend override. Verify the exact GPU, model, dtype, and image, and use the override only when that stack reproduces the missing-kernel failure.

## Crash during CUDA-graph warmup on g7e

The retained workaround is specific to Qwen2.5-32B FP8 on the vLLM 0.21 DLC: `vllm-recipes/qwen2.5-32b/fp8-g7e.4xlarge.yaml` uses `--enforce-eager` after a reproduced graph-capture failure. Eager mode is a targeted fallback, not a g7e default; current v0.26 g7e recipes keep graphs enabled. If a custom image fails during warmup, first verify backend/cubin support and memory sizing. Do not remove eager mode from the affected old-image recipe without re-testing, and do not prescribe it or `--max-num-seqs=256` to every g7e recipe.

## `KeyError: 'layers.X.mlp.down_proj.weight_scale'` at vLLM startup

**No action needed for checkpoints the current quantizer produced:** kept here because the error message points nowhere near its cause. The checkpoint's `config.json` is missing its `quantization_config` block, so vLLM never registers `weight_scale` parameters and crashes when loading the scale tensors. The quantizer exports through `modelopt`'s `export_hf_checkpoint()`, which writes the block; a checkpoint that hits this predates that, and the fix is to re-quantize it.

## vLLM crashes with `max_num_batched_tokens < max_model_len`

`--max-model-len` is the request/context ceiling; `--max-num-batched-tokens` is the scheduler's per-iteration token budget. When chunked prefill is disabled, affected vLLM versions require the per-iteration budget to cover the maximum model length. With chunked prefill enabled and supported, the context ceiling may legitimately exceed that per-iteration budget because a prefill can be split across iterations. If this startup error appears, check the effective values and chunked-prefill setting for that image and follow that version's validation message; do not impose a repository-wide inequality. Omitting `--max-model-len` merely lets vLLM infer the context ceiling and does not guarantee the inferred values are compatible.

## `PREFLIGHT FAIL: model ... not served`, or all requests return 404

Both are the same mistake: vLLM serves a model under the ID it was launched with, a local filesystem path like `/models/quantized/Qwen2.5-7B-Instruct-fp8`, not the Hugging Face hub name.

In a `suite` or `launch` run this cannot get past preflight: the run stops before measuring and the message lists the served IDs, so copy the right one. The runbooks pre-fill `--model-id` per recipe, which is why their runs do not hit this. Raw 404s remain possible only when driving a harness directly with a wrong `MODEL_ID`, as in the in-pod capture from the [Nsight runbook](../runbooks/studies/g7e.12xlarge-profile-gpus-nsight-systems-runbook.md). Confirm what the server is actually serving in-cluster:

```bash
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" exec deploy/<name> -c vllm -- \
  curl -s localhost:8000/v1/models
```

## A latency result looks off, or TTFT is much higher than expected

Check what reached the server. A `kubectl port-forward` adds roughly 185 ms of connection setup to every TTFT sample (276 ms first byte on a fresh connection against 90 ms warm), which is most of a 500 ms SLA budget — and it is the harness's own tunnel latency being reported as the server's. One run produced a full latency table through a port-forward with every cell carrying a variance flag; a warning printed at the time did not stop it from being read as a real result.

The harness now enforces this rather than advising it: `suite` refuses to measure TTFT through a loopback endpoint. Every measurement and every reference capture instead runs as a Job inside the cluster, created by `python3 app/cli.py launch`, reaching the server over cluster DNS. Capture a reference with `python3 app/cli.py launch --capture-accuracy-reference ...`; run an ordinary benchmark with `python3 app/cli.py launch ...`. Throughput-only runs are unaffected, since tokens/sec does not materially change through a tunnel, so the refusal fires only when the run would report TTFT. The escape hatch, `--allow-loopback-latency`, is recorded in the result so a reader can see it was permitted.

If you hit the refusal, the fix is not the flag — it is to run through `python3 app/cli.py launch` instead of a bare harness invocation over a tunnel.
