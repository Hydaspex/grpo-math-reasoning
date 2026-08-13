"""Prepare GSM8K records for the GRPO and SFT stages.

Writes three files: prompt-only records for GRPO, supervised records for the
SFT control arm, and a shared held-out validation split. Both training files
are built from the same underlying examples so the two arms are trained on
identical data.

Usage:
    python scripts/prepare_data.py --config configs/grpo_qwen25_1_5b.yaml
"""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathrl.config import load_config
from mathrl.data import grpo_record, sft_record, split_records, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_dataset(config.data.dataset_name, config.data.dataset_config, split="train")

    examples = []
    for example in raw:
        if config.data.max_samples and len(examples) >= config.data.max_samples:
            break
        if grpo_record(example) is not None:
            examples.append(example)

    train, validation = split_records(examples, config.data.validation_fraction, config.seed)

    write_jsonl([grpo_record(e) for e in train], config.data.grpo_path)
    write_jsonl([sft_record(e) for e in train], config.data.sft_path)
    # Validation carries the supervised shape so the gold answer travels with it.
    write_jsonl([sft_record(e) for e in validation], config.data.validation_path)

    print(
        f"Prepared {len(train)} training records (GRPO and SFT) "
        f"and {len(validation)} validation records"
    )


if __name__ == "__main__":
    main()
