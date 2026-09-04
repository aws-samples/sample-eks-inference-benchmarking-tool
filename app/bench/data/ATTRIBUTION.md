# Third-party data

## `gsm8k_subset.jsonl`

A 200-problem subset of the GSM8K test split, used by the accuracy gate (`app/bench/src/accuracy.py`).

| | |
| --- | --- |
| Source | [openai/grade-school-math](https://github.com/openai/grade-school-math) |
| Upstream file | `grade_school_math/data/test.jsonl` |
| Upstream SHA256 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| Upstream records | 1319 |
| License | MIT, Copyright (c) 2021 OpenAI |
| Subset SHA256 | `05aebd0595d406964bd11afd4ce3a20e53fe0793e5cdcef17c826303a30e73f1` |
| Selection | seeded shuffle, seed 1337, first 200 indices, then sorted |

GSM8K is introduced in [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021).

Regenerate or verify with:

```bash
python3 app/bench/src/main.py fetch-eval-data            # rebuild the subset
python3 app/bench/src/main.py fetch-eval-data --verify   # confirm the committed file reproduces
```

The subset is committed rather than fetched at run time so the gate needs no network access during a benchmark. Every accuracy reference under `docs/accuracy/` is tied to the subset digest above; changing the subset invalidates all of them.

Only the `question` field and the final answer after `####` are retained. The upstream step-by-step solutions are not used, because the gate scores the model's final answer rather than its reasoning trace.
