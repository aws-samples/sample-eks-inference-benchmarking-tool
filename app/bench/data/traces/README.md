# Pinned trace datasets

Timestamped traces for the aiperf trace-replay protocols (`agentx`, `trace_replay`). Like the prompt profiles, these are committed artifacts: the workload a protocol measures must be reviewable and identical on every run, so traces are pinned files, never generated at run time.

## agentx_32k_seed42.jsonl

The SemiAnalysis AgentX agentic-coding scenario, synthesized by the pinned aiperf and scaled to fit a 32k context window. 40 concurrent sessions, 136 requests, multi-turn with per-turn think-time delays and KV-cache-faithful prefix structure (hash_ids). Every session's peak context (input + output) is under 30,000 tokens against the serving recipes' max_model_len of 32,768.

Reproduction (exact):

    aiperf synthesize agentic-code --num-sessions 40 --seed 42 \
        --config agentx_32k_config.json --output <scratch>

aiperf==0.12.0 (app/bench/requirements-aiperf.txt). The config is the stock AgentX manifest with token dimensions scaled to a 32k window; timing and structure params untouched:

| parameter | stock | here | why |
|---|---|---|---|
| max_prompt_tokens | 200000 | 28000 | stock models 200k-context serving; our recipes serve 32k |
| cache.layer1_tokens (repo context) | 32000 | 8000 | quarter scale, keeps the shared-prefix layer meaningful below the window |
| cache.layer1_5_tokens | 20000 | 5000 | quarter scale |
| cache.layer2 mean/median | 10000/5000 | 2500/1250 | quarter scale |
| new_tokens_per_turn mean/median | 3500/1800 | 875/450 | quarter scale |

DO NOT regenerate casually: a new file is a new workload, and results measured against different traces are not comparable. If regeneration is ever needed (aiperf version bump changes the synthesizer), treat it as a protocol change: new filename, and note it in any results comparison.

## trace_replay inputs

The `trace_replay` protocol replays a user-supplied mooncake-format trace passed via `--trace <repo-relative-path>`. Place trace files in this directory so they ride the launcher's payload into the cluster. Nothing is committed for it by default: a trace is your production data, not ours to pin — and the .gitignore here keeps your traces out of git deliberately, since a committed production trace would publish your real workload patterns. The launcher packs from disk, so an ignored trace still reaches the cluster.
