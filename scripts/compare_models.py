"""Compare base, SFT and GRPO models on the held-out GSM8K split.

Generates answers from each model on identical prompts, scores them, and
reports both effect size (bootstrap confidence intervals) and significance
(one-sided paired McNemar between adjacent arms). The paired test is the one
that answers "is GRPO actually better than SFT", and it needs far fewer
examples to say so than comparing two independent intervals would.

Models are loaded sequentially to fit limited VRAM; within each model, prompts
are generated in batches.

Usage:
    python scripts/compare_models.py \
        --config configs/grpo_qwen25_1_5b.yaml \
        --sft-adapter outputs/qwen25-gsm8k-sft \
        --grpo-adapter outputs/qwen25-gsm8k-grpo

Note on batched generation: left-padding plus an explicit pad_token_id keeps
batched decoding equivalent to single-example decoding for real (non-pad)
tokens in exact arithmetic, but kernel selection can vary with batch size in
practice. Before trusting a full-split run, spot-check by generating the same
~15 examples with --batch-size 1 and your intended batch size and diffing.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import mlflow
import torch
import unsloth  # noqa: F401  (import first so its patches land)
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathrl.config import load_config
from mathrl.data import read_jsonl
from mathrl.evaluate import bootstrap_ci, mcnemar_one_sided, per_example_results, score


@torch.inference_mode()
def generate_predictions(
    model_path: str,
    records: list[dict],
    max_seq_length: int,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> list[str]:
    """Load a model, batch-generate answers, then release VRAM."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Left padding is required for batched causal-LM generation: with every
    # row's real tokens right-aligned, generate() appends new tokens after the
    # same index for every row, so one prompt_len slice is correct for the
    # whole batch.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        tokenizer.apply_chat_template(
            record["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        for record in records
    ]

    predictions: list[str] = []
    for start in range(0, len(prompts), batch_size):
        inputs = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        for row in outputs:
            predictions.append(
                tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--grpo-adapter", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of validation examples (0 = use the full split).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="outputs/comparison_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    records = read_jsonl(str(config.data.validation_path))
    if args.limit > 0:
        records = records[: args.limit]
    golds = [record["answer"] for record in records]

    stages = {
        "base": config.model.name_or_path,
        "sft": args.sft_adapter,
        "grpo": args.grpo_adapter,
    }

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment)

    results: dict[str, dict] = {}
    correctness: dict[str, list[bool]] = {}

    with mlflow.start_run(run_name=f"{config.experiment_name}-comparison"):
        mlflow.log_params({
            "n_eval": len(records),
            "base_model": config.model.name_or_path,
            "sft_adapter": args.sft_adapter,
            "grpo_adapter": args.grpo_adapter,
        })

        for stage, model_path in stages.items():
            print(f"\n=== Generating with {stage}: {model_path} ===")
            predictions = generate_predictions(
                model_path,
                records,
                config.model.max_seq_length,
                batch_size=args.batch_size,
            )
            metrics = score(predictions, golds)
            example_results = per_example_results(predictions, golds)
            correctness[stage] = [r["accuracy"] for r in example_results]

            _, acc_lo, acc_hi = bootstrap_ci(correctness[stage])
            _, fmt_lo, fmt_hi = bootstrap_ci([r["format_valid"] for r in example_results])
            ci = {
                "accuracy_ci_low": acc_lo,
                "accuracy_ci_high": acc_hi,
                "format_valid_ci_low": fmt_lo,
                "format_valid_ci_high": fmt_hi,
            }
            results[stage] = {"metrics": metrics, "ci": ci}

            with mlflow.start_run(run_name=stage, nested=True):
                mlflow.log_param("model_path", model_path)
                mlflow.log_param("n_eval", len(records))
                mlflow.log_metrics(metrics)
                mlflow.log_metrics(ci)
                pred_path = Path(f"outputs/predictions_{stage}.jsonl")
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                pred_path.write_text(
                    "\n".join(
                        json.dumps({"index": i, **r}) for i, r in enumerate(example_results)
                    ),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(pred_path), artifact_path="predictions")

        # Paired significance between adjacent arms. Examples both models get
        # right (or both wrong) carry no information about which is better and
        # drop out, so this resolves smaller gaps than the CIs above.
        comparisons = {}
        for lower, upper in (("base", "sft"), ("sft", "grpo"), ("base", "grpo")):
            test = mcnemar_one_sided(correctness[lower], correctness[upper])
            comparisons[f"{lower}_vs_{upper}"] = test
            mlflow.log_metrics({
                f"mcnemar_p_{lower}_vs_{upper}": test["p_value"],
                f"mcnemar_n01_{lower}_vs_{upper}": test["n01"],
                f"mcnemar_n10_{lower}_vs_{upper}": test["n10"],
            })
        results["comparisons"] = comparisons

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Comparison ===")
    print(f"{'Stage':<8} {'accuracy':>10} {'95% CI':>18} {'format':>10}")
    print("-" * 50)
    for stage in stages:
        m, c = results[stage]["metrics"], results[stage]["ci"]
        ci_text = f"[{c['accuracy_ci_low']:.3f}, {c['accuracy_ci_high']:.3f}]"
        print(f"{stage:<8} {m['accuracy']:>10.4f} {ci_text:>18} {m['format_valid']:>10.4f}")

    print("\n=== Paired McNemar (one-sided) ===")
    for name, test in comparisons.items():
        print(
            f"{name:<16} p={test['p_value']:.4f} "
            f"(improved {test['n01']}, regressed {test['n10']})"
        )
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
