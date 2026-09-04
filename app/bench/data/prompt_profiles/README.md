# Prompt profiles

Pinned input/output shapes for benchmark runs. One JSON file per profile, each carrying the **literal prompt text** sent to the model.

## Why these exist

Every benchmark this repo published before these profiles used a **41-token input** — a UUID nonce plus one task sentence — and **no result file recorded the input size**. Nothing in the output contradicted the assumption that the headline `1,024` was an input length rather than an output cap. That ambiguity survived multiple report revisions and a customer conversation.

## The two design rules

**1. The prompts are literal, not generated at run time.** Every model receives byte-identical input, so a throughput or latency difference between two models is a property of the models rather than of their prompt streams. It also means a run is reproducible without reproducing a random generator, and you can answer "what exactly did you send?" by opening the file.

**2. Nothing in these files references a model, and nothing ever should.** They are pinned inputs, usable for any model present or future. `target_input_tokens` is a NOMINAL size-class label, not a promise about any tokenizer. Per-model token counts are measured separately and written to `docs/prompt_tokens/<model>.json`.

The same text tokenizes differently per model, and that is the intended tradeoff: identical input is what makes a comparison valid, so the token count is the thing that moves. Measured on the same `coding_assistant` prompts, two tokenizers returned **8,187** and **9,218** tokens — a 12.6% spread with no change to a single character. Quote the measured figure for the model under test, never the nominal one.

## Repetition and prefix caching

Prompts are deliberately made as distinct as possible, because with prefix caching enabled server-side any shared **leading** text is a cache hit that inflates tokens/sec. The first version of these files was bad at this: two `coding_repo_context` prompts shared 63,084 leading characters, and `coding_assistant` prompts had a 0.90 pairwise word overlap.

Three mechanisms fix it. Every prompt opens with a unique reference line, so prompts differ at position zero where prefix matching starts. Code units are mutated per prompt — renamed identifiers, jittered constants — so reused structure is never textually identical. Prose gets per-paragraph reference markers and per-prompt ordering.

Result, excluding the declared duplicates below: **zero prompt pairs share more than 200 leading characters** in any profile, and the longest shared prefix is 13–16 characters.

**A small declared fraction (~8%) repeats an earlier prompt verbatim.** Those indices are listed in `intentional_duplicate_indices`. With prefix caching off they behave like any other prompt. With it on they are the only prompts able to hit the cache, which is how to measure the caching effect without contaminating the uncached figure.

## Profiles

| profile | prompts | nominal in | out | use case |
| ------- | ------- | ---------- | --- | -------- |
| `chat_short` | 32 | 128 | 512 | single-turn question, no history |
| `chat_typical` | 32 | 512 | 1024 | conversational turn with short history |
| `coding_completion` | 24 | 2048 | 256 | inline completion, surrounding file as context |
| `rag_document` | 16 | 4096 | 512 | retrieved passages plus a question |
| `coding_assistant` | 12 | 8192 | 1024 | coding chat: open files plus history |
| `coding_repo_context` | 8 | 16384 | 1024 | large repo or multi-file context |
| `legacy_corpus` | 5 | 9 | 150 | **historical only** |
| `legacy_reasoning` | 32 | 41 | 1024 | **historical only** |

Pool size falls as prompts grow, so any single profile stays small enough for the in-cluster launcher to ship it inside a ConfigMap alongside the harness.

## The two legacy profiles

They exist so the published figures' real shape stays visible instead of being something you reconstruct from source.

`legacy_reasoning` is the shape behind the headline throughput numbers in the published reports: a **41-token input** with a 1,024-token output cap. That input represents no real workload. Treat those numbers as node capacity under a synthetic best case: on one measured configuration, raising input to 13k tokens moved time-to-first-token from 71 ms to 2.58 s on the same server.

`legacy_corpus` is worse in two independent ways. Its 150-token output budget truncates a reasoning model mid-thought: the budget is spent on chain of thought and the request returns `finish_reason=length` with zero visible content, so it measures truncation rather than serving. And its five prompts repeat, so with prefix caching on it is a cache-hit measurement.

## Tooling

```
app/bench/src/main.py build-profiles            # regenerate (deterministic)
app/bench/src/main.py build-profiles --check    # verify committed files match a fresh build
app/bench/src/main.py verify-profiles --list
app/bench/src/main.py verify-profiles --model <hf-repo-or-local-path>
```

`verify_prompt_profiles.py` measures the pinned text with a model's tokenizer, writes `docs/prompt_tokens/<model>.json`, and exits non-zero if a profile's nominal size is off by more than 15%. It downloads tokenizer files only, not weights. Gated models need `HF_TOKEN`.

15% is deliberately loose. Cross-tokenizer variance on identical text is inherent, so this check exists to catch a profile whose label is grossly wrong, not to force tokenizers to agree.
