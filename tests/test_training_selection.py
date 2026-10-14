from __future__ import annotations

import json

from rl_countdown.training import (
    _checkpoint_global_step,
    _completed_phase_record,
    _record_resume_revision,
    select_checkpoint,
)


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


def test_resume_skips_only_a_phase_with_a_complete_step_budget(tmp_path) -> None:
    marker = tmp_path / "phase_complete.json"
    marker.write_text(
        json.dumps({"phase_complete": True, "completed_steps": 2}),
        encoding="utf-8",
    )
    assert _completed_phase_record(marker, requested_steps=2) == {
        "phase_complete": True,
        "completed_steps": 2,
    }
    assert _completed_phase_record(marker, requested_steps=3) is None


def test_resume_reads_step_from_latest_trainer_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-200"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 200}),
        encoding="utf-8",
    )
    assert _checkpoint_global_step(checkpoint) == 200
    assert _checkpoint_global_step(tmp_path / "missing") is None


def test_resume_manifest_records_each_new_code_revision_once() -> None:
    manifest = {"git_revision": "old"}
    _record_resume_revision(manifest, "old")
    _record_resume_revision(manifest, "new")
    _record_resume_revision(manifest, "new")
    assert len(manifest["resume_git_revisions"]) == 1
    assert manifest["resume_git_revisions"][0]["git_revision"] == "new"
    assert manifest["resume_git_revisions"][0]["resumed_at_utc"]
