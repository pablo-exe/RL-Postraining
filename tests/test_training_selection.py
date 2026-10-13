from __future__ import annotations

import json

from rl_countdown.training import select_checkpoint


def test_checkpoint_selection_reads_phase_local_validation_log(tmp_path) -> None:
    run_dir = tmp_path / "run-001"
    phase_dir = run_dir / "phase-01"
    evaluations = []
    for step, solve_rate, invalid_rate in ((2, 0.0, 0.5), (4, 0.125, 0.75)):
        checkpoint = phase_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        evaluations.append(
            {
                "checkpoint_tag": f"phase-01-checkpoint-{step}",
                "phase": "phase-01",
                "step": step,
                "validation_complete": True,
                "metrics": {"solve_rate": solve_rate, "invalid_rate": invalid_rate},
            }
        )
    (phase_dir / "checkpoint_evaluations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in evaluations),
        encoding="utf-8",
    )

    selected_dir, selected_metrics = select_checkpoint(run_dir, (phase_dir,))

    assert selected_dir == phase_dir / "checkpoint-4"
    assert selected_metrics["global_step"] == 4
    assert selected_metrics["metrics"]["solve_rate"] == 0.125
