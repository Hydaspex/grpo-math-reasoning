"""Verifiable reward functions for GRPO on GSM8K.

The reward is deliberately *stacked* rather than correctness-only. Exact-answer
correctness is a very sparse signal for a 1.5B model: it fires almost never for
the first hundred-odd training steps, so a group of sampled completions tends to
score uniformly zero, the within-group advantage collapses, and no gradient
flows. The format rewards are dense from step one and give the policy something
to climb while correctness is still silent.

Scoring lives in pure ``score_*`` helpers that take plain strings, with thin
adapters wrapping them in TRL's reward-function signature. That keeps every
reward unit-testable without importing trl, torch or vllm — which is what lets
CI stay CPU-only.
"""

import re
from collections.abc import Callable

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

_STRICT_FORMAT = re.compile(
    r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$", re.DOTALL
)
_SOFT_FORMAT = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)


def extract_xml_answer(text: str) -> str:
    """Pull the contents of the <answer> block out of a completion."""
    if "<answer>" not in text:
        return ""
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def extract_hash_answer(text: str) -> str | None:
    """Extract the gold answer from a GSM8K ``answer`` field (after '####')."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def score_correctness(completion: str, gold: str) -> float:
    """1.0 when the extracted answer matches the gold answer exactly."""
    return 1.0 if extract_xml_answer(completion) == gold.strip() else 0.0


def score_integer(completion: str) -> float:
    """1.0 when the extracted answer is a bare integer, as GSM8K answers are."""
    answer = extract_xml_answer(completion)
    return 1.0 if answer.lstrip("-").isdigit() else 0.0


def score_strict_format(completion: str) -> float:
    """1.0 for the exact XML layout, newlines included."""
    return 1.0 if _STRICT_FORMAT.match(completion) else 0.0


def score_soft_format(completion: str) -> float:
    """1.0 when both XML blocks are present, tolerating whitespace drift."""
    return 1.0 if _SOFT_FORMAT.search(completion) else 0.0


def score_xmlcount(completion: str) -> float:
    """Partial credit per well-formed tag, minus a nudge for trailing junk.

    Rewards the model for getting *part* of the structure right, which matters
    early when it rarely produces all four tags cleanly. The small trailing
    penalty discourages rambling on after </answer>.
    """
    count = 0.0
    if completion.count("<reasoning>\n") == 1:
        count += 0.25
    if completion.count("\n</reasoning>\n") == 1:
        count += 0.25
    if completion.count("\n<answer>\n") == 1:
        count += 0.25
        count -= len(completion.split("\n</answer>\n")[-1]) * 0.001
    if completion.count("\n</answer>") == 1:
        count += 0.25
        count -= (len(completion.split("\n</answer>")[-1]) - 1) * 0.001
    return max(count, 0.0)


def completion_text(completion: object) -> str:
    """Normalise a TRL completion to plain text.

    Conversational datasets hand back ``[{"role": ..., "content": ...}]`` per
    completion; plain-text datasets hand back a bare string.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion)


def build_reward_funcs(config) -> list[Callable]:
    """Build the weighted TRL reward functions from a RewardConfig.

    TRL calls each with ``(prompts, completions, completion_ids, **kwargs)`` and
    routes any extra dataset columns through ``**kwargs`` — which is how
    ``answer`` reaches the correctness reward.
    """

    def correctness_reward(completions, answer, **kwargs) -> list[float]:
        return [
            config.correctness * score_correctness(completion_text(c), a)
            for c, a in zip(completions, answer)
        ]

    def integer_reward(completions, **kwargs) -> list[float]:
        return [config.integer * score_integer(completion_text(c)) for c in completions]

    def strict_format_reward(completions, **kwargs) -> list[float]:
        return [config.strict_format * score_strict_format(completion_text(c)) for c in completions]

    def soft_format_reward(completions, **kwargs) -> list[float]:
        return [config.soft_format * score_soft_format(completion_text(c)) for c in completions]

    def xmlcount_reward(completions, **kwargs) -> list[float]:
        return [config.xmlcount * score_xmlcount(completion_text(c)) for c in completions]

    return [
        correctness_reward,
        integer_reward,
        strict_format_reward,
        soft_format_reward,
        xmlcount_reward,
    ]
