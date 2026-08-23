"""Evaluation metrics for GSM8K answers, with bootstrap CIs and paired testing."""

import random
import re
from collections.abc import Sequence
from math import erf, sqrt

from mathrl.rewards import extract_xml_answer, score_soft_format

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> float | None:
    """Extract the final numeric token from model output.

    Deliberately the *last* number, not the first. Chain-of-thought output walks
    through intermediate arithmetic before landing on the answer, so first-number
    extraction reliably grabs a working value instead of the result.
    """
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def extract_answer(text: str) -> float | None:
    """Extract the answer strictly from the <answer> block.

    No last-number fallback: a prediction that never emits a valid <answer>
    block scores as wrong. The earlier fallback -- grab the last number
    anywhere in the text -- silently inflated accuracy to ~1.0, because a
    truncated chain of thought almost always contains the small integer that
    GSM8K golds tend to be, so nearly any output "matched". Tying accuracy to
    the answer contract keeps it honest and makes the format cost visible.
    """
    tagged = extract_xml_answer(text)
    if tagged:
        return last_number(tagged)
    return None


def numeric_match(prediction: str, gold: str, relative_tolerance: float = 1e-3) -> bool:
    """Compare extracted answers using relative tolerance."""
    predicted, expected = extract_answer(prediction), last_number(gold)
    if predicted is None or expected is None:
        return False
    if expected == 0:
        return abs(predicted) < relative_tolerance
    return abs(predicted - expected) / abs(expected) <= relative_tolerance


def format_valid(prediction: str) -> bool:
    """Whether the completion follows the reasoning/answer XML contract."""
    return bool(score_soft_format(prediction))


def score(predictions: list[str], golds: list[str]) -> dict[str, float]:
    """Return answer accuracy and format adherence."""
    if len(predictions) != len(golds) or not golds:
        raise ValueError("predictions and golds must have the same non-zero length")
    accuracy = sum(numeric_match(p, g) for p, g in zip(predictions, golds)) / len(golds)
    formatted = sum(format_valid(p) for p in predictions) / len(predictions)
    return {"accuracy": accuracy, "format_valid": formatted}


def per_example_results(predictions: list[str], golds: list[str]) -> list[dict]:
    """Per-example prediction/gold/correctness records.

    Persists enough per example to support bootstrap CIs, paired significance
    testing and error analysis downstream.
    """
    if len(predictions) != len(golds) or not golds:
        raise ValueError("predictions and golds must have the same non-zero length")
    return [
        {
            "prediction": prediction,
            "gold": gold,
            "accuracy": numeric_match(prediction, gold),
            "format_valid": format_valid(prediction),
        }
        for prediction, gold in zip(predictions, golds)
    ]


def bootstrap_ci(
    values: Sequence[bool],
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of a per-example metric vector.

    Returns (point_estimate, ci_low, ci_high).
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    n = len(values)
    point = sum(values) / n
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    alpha = (1 - confidence) / 2
    lo_idx = max(int(alpha * n_boot), 0)
    hi_idx = min(int((1 - alpha) * n_boot) - 1, n_boot - 1)
    return point, means[lo_idx], means[hi_idx]


def mcnemar_one_sided(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float]:
    """One-sided McNemar test: is B better than A on paired per-example outcomes?

    Pairing is what makes this sample-efficient. Examples both models get right
    (or both get wrong) carry no information about which is better and drop out;
    all the signal lives in the discordant pairs, so a real difference shows up
    at a far smaller n than comparing two independent confidence intervals.

    correct_a/correct_b must be aligned — same examples, same order.
    """
    if len(correct_a) != len(correct_b) or not correct_a:
        raise ValueError("correct_a and correct_b must have the same non-zero length")
    n10 = sum(a and not b for a, b in zip(correct_a, correct_b))
    n01 = sum(b and not a for a, b in zip(correct_a, correct_b))
    n_discordant = n10 + n01
    if n_discordant == 0:
        return {"n10": n10, "n01": n01, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(n01 - n10) - 1) ** 2 / n_discordant
    z = (n01 - n10) / sqrt(n_discordant)
    p_one_sided = (1 - erf(abs(z) / sqrt(2))) / 2 if n01 > n10 else 1.0
    return {"n10": n10, "n01": n01, "statistic": statistic, "p_value": p_one_sided}
