# grpo-math-reasoning

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hydaspex/grpo-math-reasoning/blob/main/notebooks/colab_grpo.ipynb)

Reinforcement learning from **verifiable rewards**: **GRPO** (Group Relative Policy Optimization) applied to Qwen2.5-1.5B on GSM8K math reasoning, with a **supervised fine-tuning control arm** so the result is a controlled comparison rather than a before/after. Built on Unsloth QLoRA, tracked in MLflow, and evaluated with bootstrap confidence intervals and paired significance testing.

## Why GRPO

DPO and other preference methods learn offline from a fixed set of human-labelled pairs. GRPO learns *online*: it samples a group of completions per prompt, scores each one with a programmatic verifier, and pushes the policy toward the above-average members of the group. No reward model and no human labels are needed — just a function that can check an answer. Math is the clean case for this, because correctness is decidable.

It also needs no critic network, unlike PPO. The baseline is the group mean, which is what makes it cheap enough to run a real experiment on one small GPU.

## Pipeline

```text
GSM8K (openai/gsm8k, main)
        │
        ├──► prompt-only records ──► GRPO ──┐   sample group, score, advantage
        │                                    │
        └──► supervised records  ──► SFT ───┤   same data, same target format
                                             │
                                             ▼
                          base vs SFT vs GRPO on held-out split
                          accuracy + bootstrap CI + paired McNemar
```

Both arms train on identical examples and target the identical output format, so a difference in final scores reflects the training method rather than one arm being penalised for formatting differently.

## The reward

The reward is deliberately **stacked** rather than correctness-only:

| Component | Weight | Why |
| --------- | -----: | --- |
| correctness | 2.0 | The signal that actually matters |
| integer | 0.5 | GSM8K answers are bare integers |
| strict format | 0.5 | Exact `<reasoning>`/`<answer>` layout |
| soft format | 0.5 | Same, whitespace-tolerant |
| XML tag count | 0.5 | Partial credit while structure is still forming |

Correctness alone is far too sparse for a 1.5B model early in training: it fires almost never for the first hundred-odd steps, so every completion in a sampled group scores zero, the within-group advantage collapses to zero, and no gradient flows. The format rewards are dense from step one and give the policy something to climb while correctness is still silent.

## Repository layout

```text
configs/grpo_qwen25_1_5b.yaml   # one config drives every stage
src/mathrl/config.py            # validated pydantic settings
src/mathrl/data.py              # GSM8K -> chat records, deterministic split
src/mathrl/rewards.py           # the verifiable reward functions
src/mathrl/evaluate.py          # metrics, bootstrap CI, McNemar
scripts/prepare_data.py
scripts/train_sft.py            # control arm
scripts/train_grpo.py           # the main event
scripts/compare_models.py       # base vs SFT vs GRPO
notebooks/colab_grpo.ipynb      # one-click smoke run
tests/                          # CPU-only unit tests
```

## Quick start

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

python scripts/prepare_data.py --config configs/grpo_qwen25_1_5b.yaml
python scripts/train_sft.py    --config configs/grpo_qwen25_1_5b.yaml
python scripts/train_grpo.py   --config configs/grpo_qwen25_1_5b.yaml
python scripts/compare_models.py \
  --config configs/grpo_qwen25_1_5b.yaml \
  --sft-adapter outputs/qwen25-gsm8k-sft \
  --grpo-adapter outputs/qwen25-gsm8k-grpo
```

The Colab notebook runs a reduced smoke configuration end to end — enough to prove the pipeline works, not enough to produce meaningful scores. Headline numbers come from a longer run; see below.

## Evaluation

Two metrics, both computed per example so they can be aggregated statistically:

- **accuracy** — the extracted answer matches the gold answer within relative tolerance
- **format_valid** — the completion honours the reasoning/answer contract

Reported with a percentile **bootstrap confidence interval**, plus a one-sided **paired McNemar test** between adjacent arms. The paired test is what answers "is GRPO actually better than SFT": examples both models get right (or both get wrong) carry no information about which is better and drop out of the test entirely, so all the statistical power comes from the discordant pairs. That resolves a real difference at a much smaller sample size than comparing two independent confidence intervals would.

Answer extraction prefers the `<answer>` block and falls back to the **last** number in the completion, not the first. Chain-of-thought output walks through intermediate arithmetic before landing on the result, so first-number extraction reliably grabs a working value instead of the answer — a lesson carried over from an earlier project in this portfolio where exactly that bug was live.

## Results

_Pending the full training run._ This section will report accuracy and format adherence for base, SFT and GRPO with confidence intervals and McNemar p-values, alongside the GPU, wall-clock and config used to produce them.

## What I learned / what I'd do differently

_To be written against the real run._

## Licence

MIT. GSM8K is distributed by OpenAI under the MIT licence; review dataset terms before redistribution.
