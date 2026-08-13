"""Validated YAML configuration for the SFT and GRPO pipeline."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    name_or_path: str = "unsloth/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    # Unsloth spins up a colocated vLLM engine for GRPO's group sampling. This
    # is GRPO-specific; the SFT stage ignores it.
    fast_inference: bool = True
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class DataConfig(BaseModel):
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    grpo_path: Path = Path("data/grpo_train.jsonl")
    sft_path: Path = Path("data/sft_train.jsonl")
    validation_path: Path = Path("data/validation.jsonl")
    max_samples: int | None = None
    validation_fraction: float = Field(default=0.05, gt=0, lt=0.5)


class SftStageConfig(BaseModel):
    output_dir: Path = Path("outputs/qwen25-gsm8k-sft")
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    max_steps: int = -1


class GrpoStageConfig(BaseModel):
    output_dir: Path = Path("outputs/qwen25-gsm8k-grpo")
    learning_rate: float = 5.0e-6
    # Group size. GRPO's advantage is (r_i - mean(r)) / std(r) computed within
    # the group, so a group of one carries no signal.
    num_generations: int = Field(default=8, ge=2)
    max_prompt_length: int = 256
    max_completion_length: int = 200
    max_steps: int = 250
    # KL coefficient against the reference model. At 0.0 TRL skips loading a
    # reference model entirely; raise it if the policy starts reward hacking.
    beta: float = 0.0
    temperature: float = 1.0
    # Per-device batch of 1 keeps activation memory low on a 15GB T4; the
    # accumulation steps carry the effective batch up to the group size so the
    # divisibility rule below is satisfied.
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.1

    @model_validator(mode="after")
    def check_batch_divides_by_group(self) -> "GrpoStageConfig":
        effective = self.batch_size * self.gradient_accumulation_steps
        if effective % self.num_generations:
            raise ValueError(
                f"batch_size * gradient_accumulation_steps ({effective}) must be divisible "
                f"by num_generations ({self.num_generations}); TRL fails deep in the "
                "trainer with an opaque error otherwise"
            )
        return self


class RewardConfig(BaseModel):
    """Weights for the stacked reward.

    A correctness-only reward is too sparse to learn from early on, since the
    model rarely lands the exact answer in the first hundred-odd steps. The
    format rewards supply dense signal until correctness starts firing.
    """

    correctness: float = 2.0
    integer: float = 0.5
    strict_format: float = 0.5
    soft_format: float = 0.5
    xmlcount: float = 0.5


class MLflowConfig(BaseModel):
    tracking_uri: str | None = None
    experiment: str = "/Shared/grpo-math-reasoning"


class PipelineConfig(BaseModel):
    experiment_name: str = "qwen25-1.5b-gsm8k-grpo"
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sft: SftStageConfig = Field(default_factory=SftStageConfig)
    grpo: GrpoStageConfig = Field(default_factory=GrpoStageConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline configuration."""
    with open(path, encoding="utf-8") as stream:
        return PipelineConfig.model_validate(yaml.safe_load(stream))
