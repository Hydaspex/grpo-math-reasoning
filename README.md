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

Answer extraction is **strict**: the answer is read only from inside the `<answer>` block, with no fallback to a loose number elsewhere in the completion. An earlier last-number fallback silently inflated accuracy to ~1.0, because chain-of-thought output nearly always contains, somewhere, the small integer that GSM8K golds tend to be, so almost any completion "matched". Tying accuracy to the answer contract removes that confound. The consequence is deliberate and stated in the results below: a model that has not learned to emit the `<answer>` block scores zero on accuracy, whatever its underlying arithmetic ability.

## Results

Evaluated on the held-out GSM8K split (373 examples). GRPO was trained for 250 steps and SFT for one epoch, both branching from the same Qwen2.5-1.5B base under identical QLoRA settings, on a single Kaggle T4 GPU. Accuracy is exact-answer match under strict extraction; the 95% interval is a percentile bootstrap.

| Model | Accuracy | 95% CI | Format valid |
| ----- | -------: | :----: | -----------: |
| Base  | 0.0%     | [0.0, 0.0]      | 0.0%  |
| SFT   | 60.9%    | [55.5, 65.7]    | 100%  |
| GRPO  | **70.8%**| [66.0, 75.3]    | 100%  |

Reinforcement from a verifiable reward outperforms supervised imitation on the same data and target format by **9.9 points** (70.8% vs 60.9%). The gain is significant under a one-sided paired McNemar test, and the effect is corroborated by non-overlapping bootstrap intervals:

| Comparison | Improved | Regressed | p-value |
| ---------- | -------: | --------: | ------: |
| Base → SFT  | 227 | 0  | <1e-4  |
| Base → GRPO | 264 | 0  | <1e-4  |
| SFT → GRPO  | 80  | 43 | 4e-4   |

Two points warrant emphasis.

**The base score reflects format compliance, not arithmetic ability.** Strict extraction credits an answer only when it appears inside the `<answer>` block, which the untrained base model never produces — hence both its accuracy and its format-valid rate are zero. The base row should therefore be read as a starting point on the *output contract* both trained arms are taught, not as a measurement of the base model's latent reasoning. The scientifically meaningful comparison is SFT vs GRPO, where both arms satisfy the format perfectly (100%) and the accuracy difference is attributable to the training objective alone.

**Optimisation is not monotone.** Against SFT, GRPO corrects 80 examples but regresses 43 that SFT had answered correctly. The net improvement (+37 discordant pairs) is significant, but the regressions are the more instructive observation: pushing the policy toward the reward relocates probability mass in ways that recover many previously failed examples at the cost of a minority of previously solved ones — a concrete instance of the general tension between imitation and reward optimisation.

## What I learned / what I'd do differently

**A permissive metric can manufacture a perfect score.** The first full run reported ~1.0 accuracy for every arm, base included. The scorer was extracting the answer from the `<answer>` block where present, but falling back to the last number anywhere in the completion when it was not. That fallback was the flaw. GSM8K answers are typically small integers, and a chain of reasoning passes through many such numbers on its way to a result — so the final number in almost any completion, correct or not, happened to match the gold. The metric was rewarding the presence of *a* plausible number, not *the* answer. Removing the fallback and reading strictly from the `<answer>` block collapsed the illusion and restored a metric that discriminates. The general lesson: design a metric to fail loudly on a degenerate model before trusting it on a trained one.

**A "zero" is only interpretable alongside what it measures.** Strict extraction ties accuracy to the output contract, so the untrained base scores zero — not because it cannot do arithmetic, but because it never emits the required format. This is the right behaviour for a fair SFT-vs-GRPO comparison, provided the base row is read as a contract baseline rather than a reasoning measurement. The alternative, reporting a lenient score for base and a strict score for the trained arms, would compare arms on different rulers.

**Reward optimisation trades correctness around, not only upward.** GRPO's net gain over SFT (+9.9 points) coexists with 43 regressions on examples SFT had solved. The aggregate number alone would hide this; the paired analysis surfaces it. Reporting the discordant counts, not just the delta, is what makes the "cost of optimisation" visible rather than rhetorical.

**Infrastructure was the larger share of the effort.** Getting a colocated-vLLM GRPO run onto a free T4 meant resolving a FlashInfer link-time failure (the CUDA driver stub is absent from the toolkit's stubs directory on Kaggle, so `-lcuda` could not resolve), and falling back from FlashAttention-2 to xformers because the T4's compute capability (7.5) is below the 8.0 that FA2 requires. Neither is visible in the final numbers, and both would have been cheaper to anticipate than to debug mid-run.

**What I would do differently.** Report a lenient accuracy alongside the strict one, so the base model's latent arithmetic ability is visible as a separate line rather than absent; add an unrewarded validation split to check that the reward is not being gamed; and log completion-length and clipped-ratio distributions from the first step, since a rising clipped ratio silently starves correct answers of their `<answer>` block and is invisible in the reward curve alone.

## Licence

MIT. GSM8K is distributed by OpenAI under the MIT licence; review dataset terms before redistribution.
