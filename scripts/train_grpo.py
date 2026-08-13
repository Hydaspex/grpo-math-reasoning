"""GRPO training on GSM8K with verifiable rewards.

GRPO samples a group of completions per prompt, scores each with the reward
functions in mathrl.rewards, and pushes the policy toward the above-average
members of the group. Unsloth's contribution here is the loading path: it
spins up a colocated vLLM engine so sampling the group is fast enough to be
practical on a single small GPU. The training loop itself is stock TRL.

Expect reward to sit near zero for the first 100-200 steps. That is normal --
correctness fires rarely at first and the format rewards carry the early
signal -- and is the most common reason people kill a healthy run early.

Usage:
    python scripts/train_grpo.py --config configs/grpo_qwen25_1_5b.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Let the vLLM sampling phase and the training phase share GPU memory rather
# than statically partitioning it. Must be set before unsloth is imported.
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import mlflow
import unsloth  # noqa: F401  (import first so its patches land)
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathrl.config import load_config
from mathrl.data import read_jsonl
from mathrl.rewards import build_reward_funcs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name_or_path,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        # GRPO-specific: activates vLLM for group sampling, and tells it the
        # LoRA rank up front so it can size its adapter slots.
        fast_inference=config.model.fast_inference,
        max_lora_rank=config.model.lora_rank,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
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

    dataset = Dataset.from_list(read_jsonl(config.data.grpo_path))

    training_args = GRPOConfig(
        output_dir=str(config.grpo.output_dir),
        learning_rate=config.grpo.learning_rate,
        num_generations=config.grpo.num_generations,
        max_prompt_length=config.grpo.max_prompt_length,
        max_completion_length=config.grpo.max_completion_length,
        max_steps=config.grpo.max_steps,
        beta=config.grpo.beta,
        temperature=config.grpo.temperature,
        per_device_train_batch_size=config.grpo.batch_size,
        gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
        warmup_ratio=config.grpo.warmup_ratio,
        max_grad_norm=config.grpo.max_grad_norm,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        use_vllm=config.model.fast_inference,
        seed=config.seed,
        logging_steps=1,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=build_reward_funcs(config.reward),
        args=training_args,
        train_dataset=dataset,
    )

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment)

    with mlflow.start_run(run_name=f"{config.experiment_name}-grpo"):
        mlflow.log_param("stage", "grpo")
        mlflow.log_params(config.model_dump(mode="json", exclude={"mlflow"}))
        result = trainer.train()
        mlflow.log_metrics({f"train_{k}": v for k, v in result.metrics.items()})
        trainer.save_model(str(config.grpo.output_dir))
        mlflow.log_artifacts(str(config.grpo.output_dir), artifact_path="grpo_adapter")

    print(f"GRPO adapter written to {config.grpo.output_dir}")


if __name__ == "__main__":
    main()
