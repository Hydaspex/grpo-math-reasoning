"""Supervised fine-tuning control arm.

This exists to make the GRPO result a controlled comparison rather than a
before/after. It trains on the same examples, targeting the same
reasoning/answer XML format the GRPO reward pays out for, so a difference in
final scores reflects the training method rather than one arm being penalised
for formatting differently.

Usage:
    python scripts/train_sft.py --config configs/grpo_qwen25_1_5b.yaml
"""

import argparse
import sys
from pathlib import Path

import mlflow
import torch
import unsloth  # noqa: F401  (import first so its patches land)
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathrl.config import load_config
from mathrl.data import read_jsonl


def precision_flags() -> dict[str, bool]:
    """Prefer bf16 where supported, else fp16 (T4-class cards)."""
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": not bf16}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name_or_path,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        target_modules=config.model.target_modules,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    # Render to text up front rather than mapping lazily: a uniform string
    # column keeps the dataset schema stable across shards.
    records = read_jsonl(config.data.sft_path)
    texts = [
        tokenizer.apply_chat_template(record["messages"], tokenize=False) for record in records
    ]
    dataset = Dataset.from_list([{"text": text} for text in texts])

    training_args = SFTConfig(
        output_dir=str(config.sft.output_dir),
        num_train_epochs=config.sft.epochs,
        per_device_train_batch_size=config.sft.batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        learning_rate=config.sft.learning_rate,
        warmup_ratio=config.sft.warmup_ratio,
        max_steps=config.sft.max_steps,
        max_length=config.model.max_seq_length,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        seed=config.seed,
        logging_steps=10,
        report_to=[],
        # A hosted GPU session can end mid-run without warning, so checkpoint
        # often enough that losing it costs minutes, not hours.
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        **precision_flags(),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Auto-resume from the last checkpoint if this script was killed mid-run.
    last_checkpoint = None
    if Path(config.sft.output_dir).exists():
        last_checkpoint = get_last_checkpoint(str(config.sft.output_dir))
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment)

    with mlflow.start_run(run_name=f"{config.experiment_name}-sft"):
        mlflow.log_param("stage", "sft")
        mlflow.log_params(config.model_dump(mode="json", exclude={"mlflow"}))
        result = trainer.train(resume_from_checkpoint=last_checkpoint)
        mlflow.log_metrics({f"train_{k}": v for k, v in result.metrics.items()})
        trainer.save_model(str(config.sft.output_dir))
        mlflow.log_artifacts(str(config.sft.output_dir), artifact_path="sft_adapter")

    print(f"SFT adapter written to {config.sft.output_dir}")


if __name__ == "__main__":
    main()
