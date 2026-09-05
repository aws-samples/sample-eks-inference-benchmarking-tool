# Prompt token counts

This directory holds only this README in a fresh clone, and fills as you run. `docs/prompt_tokens/*.json` is gitignored, because a count belongs to the exact `transformers` release that produced it. You do not need to generate these by hand: pass `--tokenizer` to `measure` or `measure-local` and a missing file is measured and written before the run starts. Every runbook in [`docs/runbooks/`](../runbooks/README.md) already passes the right one.

Automating this is safe in a way that automating [accuracy references](../accuracy/README.md) is not. A token count is a pure function of the pinned prompt text and the tokenizer, with no server involved, so generating one cannot manufacture agreement with anything. An accuracy reference is captured from a server, so auto-capturing would compare a server against itself.

## What they are

A prompt profile in [`app/bench/data/prompt_profiles/`](../../app/bench/data/prompt_profiles/) pins the prompt text so every model is served the same input. This directory records what that text becomes for a particular tokenizer, which is a different number and the one a result should quote.

| | |
|---|---|
| **raw** | The prompt text alone. This is what a profile's `target_input_tokens` labels. |
| **wire** | The same text with that model's chat template applied. This is what the server receives and reports as `usage.prompt_tokens`. |
| **chat_template_overhead** | The difference. |

The gap is not a rounding error. The `legacy_corpus` profile is 9 raw tokens; on the wire it was 21 under one model's chat template and 38 under another's, because the templates differ. Identical text, one number each way.

Tokenizers also differ from each other on the same text. One model's tokenizer read the code profiles roughly 25% longer than the generator's chars-per-token ratio assumed, which is why the nominal target is a label rather than a measurement, and why `verify-prompts` only hard-fails against the reference tokenizer named in `config.PROFILE_REFERENCE_TOKENIZER`.

## Generating them

In the normal path you do not generate these directly. A run with `--tokenizer` measures what is missing and reports it:

```
input sizes: measured Qwen/Qwen2.5-7B-Instruct and wrote Qwen__Qwen2.5-7B-Instruct.json
```

It needs the tokenizer extras, which the benchmark client does not carry. Without them the run continues and records a nominal input size rather than failing:

```
input sizes: NOT recorded for Qwen/Qwen2.5-7B-Instruct and cannot be measured here —
pip install -r app/bench/requirements-tokenizer.txt to enable.
```

For `measure`, the measurement happens on your machine before the payload is packed, because the in-cluster Job runs a stock Python image with aiohttp only and no route to Hugging Face.

To record a tokenizer ahead of time, or to see the full raw-versus-wire table with the tolerance check, run it directly. This needs no GPU, no server, and no cluster:

```bash
pip install -r app/bench/requirements-tokenizer.txt
eks-bench verify-prompts --model Qwen/Qwen2.5-7B-Instruct
```

There is one file per tokenizer, named after the repo id with `/` replaced by `__`, so `Qwen/Qwen2.5-7B-Instruct` becomes `Qwen__Qwen2.5-7B-Instruct.json`. Quantization does not change a tokenizer, so an FP8 or NVFP4 deployment uses its base model's file: `nvidia/GLM-5.2-NVFP4` covers the GLM FP8 recipes too.

Kimi K3's tokenizer needs `tiktoken`, which is why that requirements file pins it.

## How a run uses them

Pass `--tokenizer <repo-id>` to `measure` or `measure-local` and the protocol lock resolves both sizes from the matching file:

```
raw in    : 10320 tok measured for this tokenizer (nominal 8192, +26.0% off nominal)
wire in   : 10333 tok wire (raw 10320 + template 13), from google__gemma-4-31b-it.json
```

Omit the flag, or point it at a tokenizer with no file, and the lock records that and prints the command that would create it, rather than estimating a value. Every runbook already passes the right `--tokenizer` for its model.

## Version pinning

`app/bench/requirements-tokenizer.txt` pins `transformers`, `jinja2` and `tiktoken` exactly, because a different release can tokenize the same text differently or render a different chat template. Each file records the versions that produced it under `measured_with`, so a count stays attributable. Recapture after changing any of those pins.
