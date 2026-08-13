"""GSM8K record preparation for the GRPO and SFT stages."""

import json
import random
import re
from pathlib import Path
from typing import Any

from mathrl.rewards import SYSTEM_PROMPT, extract_hash_answer

# GSM8K interleaves calculator annotations like <<48/2=24>> into its reasoning.
# They are an artefact of how the dataset was built, not something we want the
# model to imitate.
_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def reasoning_text(answer: str) -> str:
    """Strip calculator annotations and the '#### N' tail from a GSM8K answer."""
    body = answer.split("####")[0]
    return _CALC_ANNOTATION.sub("", body).strip()


def grpo_record(example: dict[str, Any]) -> dict | None:
    """Build a prompt-only record; GRPO samples the completions itself."""
    gold = extract_hash_answer(example.get("answer", ""))
    if gold is None or not example.get("question"):
        return None
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"].strip()},
        ],
        "answer": gold,
    }


def sft_record(example: dict[str, Any]) -> dict | None:
    """Build a supervised record targeting the same XML format GRPO rewards.

    Holding the output format constant across both arms is what makes the
    comparison a controlled one: any difference in scores reflects the training
    method rather than one arm being penalised for formatting differently.
    """
    gold = extract_hash_answer(example.get("answer", ""))
    if gold is None or not example.get("question"):
        return None
    completion = (
        f"<reasoning>\n{reasoning_text(example['answer'])}\n</reasoning>\n"
        f"<answer>\n{gold}\n</answer>"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"].strip()},
            {"role": "assistant", "content": completion},
        ],
        "answer": gold,
    }


def split_records(records: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split deterministically into (train, validation)."""
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    n_validation = max(1, int(len(records) * fraction))
    validation = set(indices[:n_validation])
    return (
        [r for i, r in enumerate(records) if i not in validation],
        [r for i, r in enumerate(records) if i in validation],
    )


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write newline-delimited JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        stream.writelines(json.dumps(record) + "\n" for record in records)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read newline-delimited JSON."""
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
