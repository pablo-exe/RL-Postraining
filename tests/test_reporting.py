from __future__ import annotations

import json

from rl_countdown.config import ProjectConfig
from rl_countdown.reporting import _paired_bootstrap_vs_base, _update_readme_results


def _write_predictions(path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"problem_id": problem_id, "correct": correct}) + "\n"
            for problem_id, correct in rows
        ),
        encoding="utf-8",
    )


def test_secondary_bootstrap_pairs_shared_problem_ids_and_is_reproducible(tmp_path) -> None:
    config = ProjectConfig(
        root=tmp_path,
        values={
            "project": {"results_root": "results"},
            "evaluation": {"bootstrap_samples": 500, "bootstrap_seed": 7},
        },
    )
    evaluations = tmp_path / "results" / "evaluations"
    _write_predictions(
        evaluations / "base_qwen25_15b" / "predictions_test_id_greedy.jsonl",
        [("a", 0), ("b", 1), ("c", 0), ("unpaired", 1)],
    )
    _write_predictions(
        evaluations / "grpo_validity_seed_42" / "predictions_test_id_greedy.jsonl",
        [("a", 1), ("b", 1), ("c", 0), ("candidate_only", 1)],
    )

    first = _paired_bootstrap_vs_base(config, "grpo_validity_seed_42", seed=1008)
    second = _paired_bootstrap_vs_base(config, "grpo_validity_seed_42", seed=1008)

    assert first == second
    assert first is not None
    assert first["n_paired"] == 3
    assert first["mean_accuracy_difference"] == 1 / 3
    assert first["ci_95_lower"] <= first["mean_accuracy_difference"]
    assert first["mean_accuracy_difference"] <= first["ci_95_upper"]


def test_secondary_bootstrap_requires_shared_predictions(tmp_path) -> None:
    config = ProjectConfig(
        root=tmp_path,
        values={
            "project": {"results_root": "results"},
            "evaluation": {"bootstrap_samples": 100, "bootstrap_seed": 7},
        },
    )
    evaluations = tmp_path / "results" / "evaluations"
    _write_predictions(
        evaluations / "base_qwen25_15b" / "predictions_test_id_greedy.jsonl",
        [("base", 1)],
    )
    _write_predictions(
        evaluations / "sft_witness_seed_42" / "predictions_test_id_greedy.jsonl",
        [("sft", 1)],
    )

    assert _paired_bootstrap_vs_base(config, "sft_witness_seed_42", seed=1009) is None


def test_readme_table_shows_correct_count_not_evaluated_count(tmp_path) -> None:
    config = ProjectConfig(
        root=tmp_path,
        values={
            "project": {"results_root": "results"},
            "evaluation": {},
        },
    )
    (tmp_path / "README.md").write_text(
        "before\n<!-- MEASURED_RESULTS_START -->\nold\n<!-- MEASURED_RESULTS_END -->\nafter\n",
        encoding="utf-8",
    )
    _update_readme_results(
        config,
        [
            {
                "experiment_id": "base_qwen25_15b",
                "test_id": {
                    "complete": True,
                    "count": 1024,
                    "expected_count": 1024,
                    "correct_count": 10,
                    "solve_rate": 10 / 1024,
                },
                "test_ood": {
                    "complete": True,
                    "count": 512,
                    "expected_count": 512,
                    "correct_count": 2,
                    "solve_rate": 2 / 512,
                },
            }
        ],
    )

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "1.0% (10/1024)" in readme
    assert "0.4% (2/512)" in readme
    assert "1024/1024" not in readme
