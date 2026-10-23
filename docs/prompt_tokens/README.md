# Prompt token counts

Each JSON (schema prompt-profile-tokens/v2) carries the model id, capture time, the exact measured_with library versions, a tolerance, and a profiles block. For every profile (chat_short, chat_typical, coding_assistant, coding_completion, coding_repo_context, legacy_corpus, legacy_reasoning, rag_document) it stores:

| | |
|---|---|
| **raw** | The prompt text alone. This is what a profile's `target_input_tokens` labels. |
| **wire** | The same text with that model's chat template applied. This is what the server receives and reports as `usage.prompt_tokens`. |
| **chat_template_overhead** | The gap between raw and wire. |

## How a run uses them

Follow the runbook to pass `--tokenizer <repo-id>` to `measure` or `measure-local` and the protocol lock resolves both sizes from the matching file:

```
raw in    : 10320 tok measured for this tokenizer (nominal 8192, +26.0% off nominal)
wire in   : 10333 tok wire (raw 10320 + template 13), from google__gemma-4-31b-it.json
```

## Version pinning

`app/bench/requirements-tokenizer.txt` pins `transformers`, `jinja2` and `tiktoken` exactly, because a different release can tokenize the same text differently or render a different chat template. Each file records the versions that produced it under `measured_with`, so a count stays attributable. Recapture after changing any of those pins.
