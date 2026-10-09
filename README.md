# Persona Experiment

A mini replication of **"Playing Pretend: Expert Personas Don't Improve Factual Accuracy"**
(Prompting Science Report 4, Wharton GAIL Generative AI Lab; [arXiv:2512.05858](https://arxiv.org/abs/2512.05858))
run on local models.

**Question:** does telling a model it is an expert make it answer more accurately?
**Answer (this experiment):** no. Personas change behavior, not capability.

## Method

- **Benchmark:** 44 graduate-level questions sampled from the MMLU-Pro test set
  (science, engineering, law, economics, math, history, biology, chemistry,
  physics), seeded and reproducible (`seed=42`).
- **Models:** `qwen3.5:9b` and `qwen3.5:0.8b` via Ollama, temperature 0,
  thinking disabled.
- **Design:** paired — every question is answered under all four conditions, so
  any accuracy difference is attributable to the system prompt.

### Conditions

| Condition | System prompt |
|---|---|
| baseline | *(none)* |
| expert | "You are a world-class expert in physics, chemistry, biology, computer science, engineering, mathematics, economics, law, and history. You have decades of experience and you never make mistakes." |
| layperson | "You are a layperson with no specialized training in any technical or academic subject." |
| toddler | "You are a toddler. You are still learning the basics of the world and have almost no knowledge of science, engineering, law, or mathematics." |

## Results

### Accuracy by condition (44 questions)

| Condition | qwen3.5:9b | qwen3.5:0.8b |
|---|---|---|
| baseline | **19/44 (43%)** | **10/44 (23%)** |
| expert | 20/44 (45%) | 9/44 (20%) |
| layperson | 18/44 (41%) | 9/44 (20%) |
| toddler | 18/44 (41%) | 8/44 (18%) |

### Deltas vs baseline

| Condition | qwen3.5:9b | qwen3.5:0.8b |
|---|---|---|
| expert | +2pp | −3pp |
| layperson | −2pp | −3pp |
| toddler | −2pp | −5pp |

### Interpretation

- On the 9b model, the "world-class expert who never makes mistakes" prompt
  changed answers on **5 of 44** questions and flipped exactly **1** from wrong
  to right. The model was still wrong on 24 of 44.
- On the 0.8b model, the same expert prompt **broke two previously-correct
  answers** and lowered accuracy by 3pp.
- Across both models and 88 question-answers, the expert persona's net effect
  on accuracy was **zero**.
- Personas are not inert: the toddler persona changed 12 answers on the 9b
  model and broke two answers that had been correct (Q27, Q30). Negative-
  capability personas actively degrade performance, matching the study.
- The direction matches the original study: expert personas do not improve
  factual accuracy; low-knowledge personas can hurt it.

### Caveats (read before citing)

- Small sample (44 questions, single seeded draw), single run per condition,
  temperature 0. The ±2-5pp deltas are within noise at this n.
- Results are for accuracy of answers only. Personas can legitimately change
  tone, style, and formatting — the study's explicit caveat.
- Local 9b/0.8b models are far below frontier capability; the absolute scores
  are not the point, the persona effect (or lack of it) is.

## Reproduce

```bash
# 1. Pull the models
ollama pull qwen3.5:9b
ollama pull qwen3.5:0.8b

# 2. (Optional) Regenerate the question sample — requires pyarrow
pip install pyarrow
python3 sample_mmlu.py        # reads mmlu_pro_test.parquet, writes mmlu_pro_sample.json

# 3. Run the experiment (needs a running Ollama server on localhost:11434)
python3 experiment.py qwen3.5:9b
python3 experiment.py qwen3.5:0.8b
```

The 44-question sample is committed (`mmlu_pro_sample.json`); full transcripts
per condition are in `results/`.

## OpenRouter re-run with thinking enabled

The local think runs (`experiment.py --think`) were flooded with non-answers
(`?`): with Ollama, `num_predict=2048` caps the *entire* completion, thinking
included, so long-reasoning runs were cut off mid-thought and returned empty
content (the toddler condition never answered once - 44/44 empty).

`experiment_openrouter_think.py` re-runs the same experiment on
`qwen/qwen3.5-9b` via OpenRouter, where the output budget is 262k tokens, so
thinking gets real headroom:

- Same 44 questions, same four conditions, same `temperature 0`, paired design
- Reasoning enabled (`reasoning: {"effort": "xhigh", "exclude": false}`),
  chain-of-thought captured per answer
- `max_tokens=16384` (observed local thinking was ~2-2.5k tokens)
- Retries with backoff on 429/5xx; aborts after 3 consecutive failures
- Live progress bar (ETA, rate, per-condition counts); Ctrl+C saves and
  resumes on rerun; results saved incrementally every 10 tasks

```bash
export OPENROUTER_API_KEY=sk-or-...   # or it auto-loads from ~/.bashrc
python3 experiment_openrouter_think.py              # full run: 176 calls, ~$0.08
python3 experiment_openrouter_think.py --limit 5    # smoke test, 5 questions
python3 experiment_openrouter_think.py --workers 30 # probe for more headroom
```

Results land in `results/results_qwen3.5-9b_openrouter_think.json`.

### Worker scaling (measured 2026-08-17, xhigh reasoning)

Empirical throughput for this model, so future runs start at the right count:

| Workers | Sustained rate | Notes |
|---|---|---|
| 10 | ~0.12/s (~20m for 176 calls) | Under-utilized the providers |
| 25 | ~0.22/s (~7m) | ~1.8x throughput; no 429s at either count |

Paid-model variants have no OpenRouter platform request cap, so the binding
constraint is upstream provider concurrency. Start at 20-25 workers. The
progress bar's `FAILS=` counter is the tripwire: if it starts climbing, back
off. If you want the run faster than worker scaling can give, lower
`reasoning.effort` (xhigh -> high/medium) - that cuts per-call latency
directly.

## Files

- `experiment.py` — main experiment (std-lib only + Ollama HTTP)
- `experiment_openrouter_think.py` — OpenRouter re-run with thinking enabled
- `analyze_results.py` — CLI analysis of any results file (summary, deltas,
  answer changes, per-category, per-question, reasoning stats, compare,
  `--json`; handles both result shapes and partial in-progress files)
- `sample_mmlu.py` — seeded sampler from the MMLU-Pro parquet (needs pyarrow)
- `mmlu_pro_sample.json` — the 44 questions used
- `results/results_qwen3.5-9b.json`, `results/results_qwen3.5-0.8b.json` — full results with raw model output
