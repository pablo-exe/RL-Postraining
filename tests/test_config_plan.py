from __future__ import annotations

from rl_countdown.cli import build_parser
from rl_countdown.config import load_config, load_experiments
from rl_countdown.training import planned_training_steps


def test_experiment_matrix_is_explicit_and_unique() -> None:
    config = load_config()
    experiments = load_experiments(config)
    assert len(experiments) == 6
    assert len({experiment.experiment_id for experiment in experiments}) == 6
    assert [experiment.method for experiment in experiments].count("grpo") == 4


def test_training_plan_uses_measured_profile_and_omits_controls_if_needed() -> None:
    config = load_config()
    steps, plan = planned_training_steps(
        config,
        {
            "pilot_grpo_seconds_per_step": 300,
            "sft_seconds_per_step": 100,
            "model_load_seconds_per_run": 60,
            "validation_seconds_per_checkpoint": 10,
        },
    )
    assert steps == min(config.grpo["final_step_candidates"])
    assert plan["selected_grpo_seeds"] == [42]
    assert plan["run_validity_ablation"] is False
    assert plan["run_sft_control"] is False


def test_cli_exposes_documented_commands() -> None:
    parser = build_parser()
    command_names = parser._subparsers._group_actions[0].choices
    assert set(command_names) == {
        "doctor",
        "prepare-data",
        "baseline",
        "smoke",
        "pilot",
        "train-grpo",
        "train-sft",
        "evaluate",
        "report",
        "predict",
    }
