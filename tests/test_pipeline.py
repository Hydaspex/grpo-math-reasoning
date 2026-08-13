from pathlib import Path

import pytest
from pydantic import ValidationError

from mathrl.config import GrpoStageConfig, PipelineConfig, RewardConfig, load_config
from mathrl.data import grpo_record, reasoning_text, sft_record, split_records
from mathrl.evaluate import (
    bootstrap_ci,
    extract_answer,
    last_number,
    mcnemar_one_sided,
    numeric_match,
    per_example_results,
    score,
)
from mathrl.rewards import (
    build_reward_funcs,
    extract_hash_answer,
    extract_xml_answer,
    score_correctness,
    score_integer,
    score_soft_format,
    score_strict_format,
    score_xmlcount,
)

WELL_FORMED = "<reasoning>\nNatalia sold 24 clips.\n</reasoning>\n<answer>\n72\n</answer>"

GSM8K_EXAMPLE = {
    "question": "Natalia sold clips to 48 friends.",
    "answer": "She sold 48/2 = <<48/2=24>>24 clips in May.\n#### 72",
}


# --- config -----------------------------------------------------------------


def test_repo_config_loads():
    config = load_config(Path(__file__).parents[1] / "configs" / "grpo_qwen25_1_5b.yaml")
    assert config.model.name_or_path == "unsloth/Qwen2.5-1.5B-Instruct"
    assert config.grpo.num_generations == 8


def test_defaults_are_valid():
    config = PipelineConfig()
    assert config.seed == 42
    assert config.reward.correctness > config.reward.integer


def test_group_of_one_rejected():
    with pytest.raises(ValidationError):
        GrpoStageConfig(num_generations=1)


def test_batch_not_divisible_by_group_rejected():
    with pytest.raises(ValidationError):
        GrpoStageConfig(num_generations=8, batch_size=3, gradient_accumulation_steps=1)


# --- data -------------------------------------------------------------------


def test_reasoning_text_strips_calc_annotations_and_answer():
    assert "<<" not in reasoning_text(GSM8K_EXAMPLE["answer"])
    assert "####" not in reasoning_text(GSM8K_EXAMPLE["answer"])


def test_grpo_record_is_prompt_only():
    record = grpo_record(GSM8K_EXAMPLE)
    assert record["answer"] == "72"
    assert [m["role"] for m in record["prompt"]] == ["system", "user"]


def test_sft_record_targets_the_rewarded_format():
    record = sft_record(GSM8K_EXAMPLE)
    completion = record["messages"][-1]["content"]
    assert score_strict_format(completion) == 1.0
    assert extract_xml_answer(completion) == "72"


def test_records_rejected_without_gold_answer():
    assert grpo_record({"question": "q", "answer": "no delimiter"}) is None
    assert sft_record({"question": "q", "answer": "no delimiter"}) is None


def test_split_is_deterministic():
    records = [{"id": i} for i in range(20)]
    first = split_records(records, 0.2, 42)
    second = split_records(records, 0.2, 42)
    assert first == second
    assert {r["id"] for r in first[0]}.isdisjoint({r["id"] for r in first[1]})


# --- rewards ----------------------------------------------------------------


def test_extract_hash_answer():
    assert extract_hash_answer(GSM8K_EXAMPLE["answer"]) == "72"
    assert extract_hash_answer("no delimiter here") is None


def test_score_correctness_requires_exact_match():
    assert score_correctness(WELL_FORMED, "72") == 1.0
    assert score_correctness(WELL_FORMED, "71") == 0.0


def test_score_integer_accepts_negatives_rejects_prose():
    assert score_integer("<answer>\n-5\n</answer>") == 1.0
    assert score_integer("<answer>\nabout five\n</answer>") == 0.0


def test_strict_format_is_stricter_than_soft():
    sloppy = "<reasoning>some working</reasoning> <answer>72</answer>"
    assert score_strict_format(sloppy) == 0.0
    assert score_soft_format(sloppy) == 1.0


def test_xmlcount_gives_partial_credit():
    partial = "<reasoning>\nworking\n</reasoning>\n"
    assert 0.0 < score_xmlcount(partial) < score_xmlcount(WELL_FORMED)


def test_xmlcount_never_negative():
    assert score_xmlcount("<answer>\n1\n</answer>\n" + "junk" * 500) >= 0.0


def test_build_reward_funcs_applies_weights():
    funcs = build_reward_funcs(RewardConfig())
    correctness = funcs[0]
    rewards = correctness(completions=[WELL_FORMED], answer=["72"])
    assert rewards == [2.0]


def test_reward_funcs_handle_conversational_completions():
    funcs = build_reward_funcs(RewardConfig())
    correctness = funcs[0]
    chat = [[{"role": "assistant", "content": WELL_FORMED}]]
    assert correctness(completions=chat, answer=["72"]) == [2.0]


# --- evaluate ---------------------------------------------------------------


def test_last_number_beats_first_for_chain_of_thought():
    cot = "First 48/2 = 24, then 24 * 3 = 72"
    assert last_number(cot) == 72.0


def test_extract_answer_prefers_the_answer_block():
    text = "I computed 24 along the way.\n<answer>\n72\n</answer>"
    assert extract_answer(text) == 72.0


def test_numeric_match_tolerates_commas():
    assert numeric_match("<answer>\n1,200\n</answer>", "1200")


def test_score_reports_accuracy_and_format():
    metrics = score([WELL_FORMED, "just 5"], ["72", "5"])
    assert metrics["accuracy"] == 1.0
    assert metrics["format_valid"] == 0.5


def test_score_rejects_empty():
    with pytest.raises(ValueError):
        score([], [])


def test_per_example_results_shape():
    results = per_example_results([WELL_FORMED], ["72"])
    assert results[0]["accuracy"] is True
    assert results[0]["format_valid"] is True


def test_bootstrap_ci_is_deterministic_and_bounded():
    values = [True, False, True, True, False]
    point, lo, hi = bootstrap_ci(values, n_boot=200, seed=7)
    assert bootstrap_ci(values, n_boot=200, seed=7) == (point, lo, hi)
    assert lo <= point <= hi


def test_bootstrap_ci_all_correct_is_tight():
    point, lo, hi = bootstrap_ci([True] * 20, n_boot=200, seed=1)
    assert (point, lo, hi) == (1.0, 1.0, 1.0)


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_mcnemar_detects_directional_improvement():
    a = [True, True, False, False, False, False, True, False]
    b = [True, True, True, True, True, False, True, False]
    result = mcnemar_one_sided(a, b)
    assert result["n01"] > result["n10"]
    assert result["p_value"] < 0.5


def test_mcnemar_no_discordant_pairs_returns_p_one():
    assert mcnemar_one_sided([True, False], [True, False])["p_value"] == 1.0


def test_mcnemar_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mcnemar_one_sided([True], [True, False])
