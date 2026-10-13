"""Command-line workflow for the complete, budgeted portfolio experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from rl_countdown.budget import GpuBudget
from rl_countdown.config import ProjectConfig, load_config
from rl_countdown.dataset import load_dataset, prepare_data
from rl_countdown.evaluation import evaluate_baseline, run_full_evaluation
from rl_countdown.modeling import cuda_diagnostic, discover_adapter_registry, load_base_model
from rl_countdown.reporting import generate_report
from rl_countdown.training import (
    finalize_grpo_probe,
    generate_for_problem,
    planned_training_steps,
    train_grpo_experiment,
    train_sft_experiment,
    write_json,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value of type {type(value).__name__} cannot be serialized as JSON.")


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default))


def _project_fingerprint(config: ProjectConfig, dataset_manifest: dict[str, Any]) -> str:
    frozen = json.dumps(
        {"configuration": config.values, "dataset_manifest": dataset_manifest},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(frozen.encode("utf-8")).hexdigest()


def _apply_runtime_overrides(config: ProjectConfig) -> ProjectConfig:
    path = config.results_dir / "runtime_overrides.json"
    if not path.is_file():
        return config
    record = json.loads(path.read_text(encoding="utf-8"))
    overrides = record.get("grpo", {})
    allowed = {"generation_batch_size", "num_generations"}
    if set(overrides) - allowed:
        raise ValueError("Only the preregistered GRPO memory fallback may be stored as a runtime override.")
    values = copy.deepcopy(config.values)
    values["grpo"].update(overrides)
    return ProjectConfig(root=config.root, values=values)


def _load_bundle(config: ProjectConfig):
    if not config.data_dir.is_dir():
        raise FileNotFoundError("Frozen data not found. Run `rl-countdown prepare-data` first.")
    return load_dataset(config.data_dir)


def _budget_limit(config: ProjectConfig, stage: str) -> float:
    budget = config.budget
    if stage == "preflight":
        return float(budget["preflight_and_pilots_hours"])
    if stage == "training":
        return float(budget["preflight_and_pilots_hours"]) + float(budget["final_training_hours"])
    if stage in {"evaluation", "prediction"}:
        return float(budget["gpu_hours_total"])
    raise ValueError(f"Unknown budget stage: {stage}")


def _run_directory(config: ProjectConfig, manifest: dict[str, Any]) -> Path:
    return config.runs_dir / "experiments" / str(manifest["experiment_id"]) / str(manifest["run_id"])


def _registered_manifest(config: ProjectConfig, experiment_id: str) -> dict[str, Any] | None:
    registry = discover_adapter_registry(config.root)
    entries = registry["experiments"].get(experiment_id, [])
    valid = []
    for entry in entries:
        path = config.root / entry["run_manifest"]
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") in {"completed", "budget_stopped"}:
                valid.append(manifest)
    if not valid:
        return None
    return max(valid, key=lambda item: str(item.get("completed_at_utc", "")))


def _completed_run(config: ProjectConfig, experiment_id: str) -> dict[str, Any] | None:
    root = config.runs_dir / "experiments" / experiment_id
    if not root.is_dir():
        return None
    candidates = []
    for path in root.iterdir():
        manifest_path = path / "run_manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            candidates.append(manifest)
    if not candidates:
        return None
    return max(candidates, key=lambda item: str(item.get("completed_at_utc", "")))


def _run_rollout_statistics(run_dir: Path, problem_limit: int | None = None) -> dict[str, Any]:
    path = run_dir / "rollouts.jsonl"
    groups: dict[tuple[str, int, str], list[float]] = {}
    selected_problems: set[str] = set()
    rows = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = str(record["problem_id"])
            if problem_id not in selected_problems:
                if problem_limit is not None and len(selected_problems) >= problem_limit:
                    continue
                selected_problems.add(problem_id)
            rows += 1
            key = (str(record["phase"]), int(record["global_step"]), problem_id)
            groups.setdefault(key, []).append(float(record["reward"]))
    comparable = [rewards for rewards in groups.values() if len(rewards) > 1]
    varied = [rewards for rewards in comparable if len(set(rewards)) > 1]
    return {
        "completion_count": rows,
        "reward_groups": len(comparable),
        "reward_varying_groups": len(varied),
        "reward_variation_rate": len(varied) / len(comparable) if comparable else 0.0,
        "groups_with_positive_reward": sum(any(reward > 0 for reward in group) for group in comparable),
        "positive_group_rate": (
            sum(any(reward > 0 for reward in group) for group in comparable) / len(comparable)
            if comparable
            else 0.0
        ),
    }


def _checkpoint_metrics(run_dir: Path, global_step: int | None = None) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.glob("phase-*/checkpoint_evaluations.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if global_step is None or int(record["step"]) == global_step:
                    records.append(record)
    return records


def _pilot_progress_path(config: ProjectConfig) -> Path:
    return config.results_dir / "calibration_progress.json"


def _pilot_existing_record(config: ProjectConfig, experiment_id: str) -> dict[str, Any] | None:
    path = _pilot_progress_path(config)
    if not path.is_file():
        return None
    progress = json.loads(path.read_text(encoding="utf-8"))
    return progress.get("runs", {}).get(experiment_id)


def _save_pilot_record(
    config: ProjectConfig,
    experiment_id: str,
    record: dict[str, Any],
    fingerprint: str,
) -> None:
    path = _pilot_progress_path(config)
    progress = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema_version": 1, "runs": {}, "project_fingerprint": fingerprint}
    )
    if progress.get("project_fingerprint") != fingerprint:
        raise ValueError("Pilot progress belongs to a different frozen project fingerprint.")
    progress["runs"][experiment_id] = record
    write_json(path, progress)


def _check_resume_fingerprint(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}. Run `rl-countdown pilot` first.")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("project_fingerprint") != fingerprint:
        raise ValueError("The pilot calibration belongs to a different model, dataset, or project configuration.")
    return record


def _run_baseline(config: ProjectConfig) -> dict[str, Any]:
    bundle = _load_bundle(config)
    tracker = GpuBudget(
        config.runs_dir,
        "baseline-validation",
        max_hours=_budget_limit(config, "preflight"),
    )
    with tracker:
        metrics = evaluate_baseline(config, bundle, tracker)
    tracker.write_summary(config.results_dir / "gpu_budget_summary.json")
    _emit({"validation_metrics": metrics, "budget": tracker.snapshot()})
    return metrics


def _run_smoke(config: ProjectConfig) -> dict[str, Any]:
    from peft import PeftModel

    from rl_countdown.core import Problem

    smoke_values = copy.deepcopy(config.values)
    smoke_values["grpo"]["curriculum_phase_evaluation_size"] = min(
        16, int(config.grpo["curriculum_phase_evaluation_size"])
    )
    config = ProjectConfig(root=config.root, values=smoke_values)
    bundle = _load_bundle(config)
    baseline_path = config.results_dir / "evaluations" / "base_qwen25_15b" / "baseline_validation_metrics.json"
    if not baseline_path.is_file():
        raise FileNotFoundError("Run `rl-countdown baseline` before smoke training so the holdout is fixed.")
    steps = int(config.grpo["smoke_steps"])
    learning_rate = float(config.grpo["learning_rate_candidates"][0])
    tracker = GpuBudget(config.runs_dir, "smoke-grpo", max_hours=_budget_limit(config, "preflight"))

    def execute_smoke(active_config: ProjectConfig, experiment_id: str) -> dict[str, Any]:
        completed = _completed_run(active_config, experiment_id)
        if completed is not None:
            return completed
        probe = train_grpo_experiment(
            active_config,
            bundle,
            tracker,
            experiment_id=experiment_id,
            reward_kind="exact_plus_validity",
            seed=42,
            steps=steps,
            learning_rate=learning_rate,
            resume=True,
            stop_at_global_step=4,
            checkpoint_save_steps=2,
        )
        if probe.get("status") != "probe_complete":
            raise RuntimeError(f"The smoke resume probe ended unexpectedly: {probe.get('status')}.")
        return train_grpo_experiment(
            active_config,
            bundle,
            tracker,
            experiment_id=experiment_id,
            reward_kind="exact_plus_validity",
            seed=42,
            steps=steps,
            learning_rate=learning_rate,
            resume=True,
            checkpoint_save_steps=2,
        )

    with tracker:
        run_id = (
            "smoke_direct_grpo_shaped_seed_42_fallback"
            if int(config.grpo["num_generations"]) == int(config.grpo["num_generations_fallback"])
            else "smoke_direct_grpo_shaped_seed_42"
        )
        try:
            manifest = execute_smoke(config, run_id)
        except Exception as error:
            is_cuda_oom = "out of memory" in str(error).lower()
            if not is_cuda_oom or int(config.grpo["num_generations"]) == int(config.grpo["num_generations_fallback"]):
                raise
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            fallback_values = copy.deepcopy(config.values)
            fallback_values["grpo"]["generation_batch_size"] = int(config.grpo["generation_batch_size_fallback"])
            fallback_values["grpo"]["num_generations"] = int(config.grpo["num_generations_fallback"])
            config = ProjectConfig(root=config.root, values=fallback_values)
            overrides = {
                "schema_version": 1,
                "reason": f"CUDA memory fallback after {type(error).__name__}: {error}",
                "grpo": {
                    "generation_batch_size": int(config.grpo["generation_batch_size"]),
                    "num_generations": int(config.grpo["num_generations"]),
                },
                "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            write_json(config.results_dir / "runtime_overrides.json", overrides)
            run_id = "smoke_direct_grpo_shaped_seed_42_fallback"
            manifest = execute_smoke(config, run_id)
        run_dir = _run_directory(config, manifest)
        rollouts = _run_rollout_statistics(
            run_dir,
            problem_limit=int(config.budget["uninformative_group_probe_size"]),
        )
        phases = manifest.get("phase_runs", [])
        changed = bool(phases) and all(
            phase.get("adapter_sha256_before") != phase.get("adapter_sha256_after")
            for phase in phases
            if phase.get("completed_steps", 0) > 0
        )
        base_unchanged = bool(phases) and all(
            phase.get("frozen_base_probe_sha256_before") == phase.get("frozen_base_probe_sha256_after")
            for phase in phases
        )
        train_metrics_path = run_dir / "training_metrics.jsonl"
        train_metrics = [
            json.loads(line)
            for line in train_metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        finite_metrics = all(
            math.isfinite(float(value))
            for row in train_metrics
            for value in row["metrics"].values()
            if isinstance(value, (int, float))
        )
        phase_two = next((phase for phase in phases if phase["phase"] == "phase-02"), {})
        resume_checkpoint = phase_two.get("resumed_from_checkpoint")
        reload_base, reload_tokenizer = load_base_model(config)
        selected_adapter = run_dir / "adapter_selected"
        reload_model = PeftModel.from_pretrained(reload_base, str(selected_adapter), is_trainable=False)
        reload_probe = generate_for_problem(
            reload_model,
            reload_tokenizer,
            Problem("smoke-reload", "validation", (2, 3, 7), 20),
            config,
        )
        report = {
            "experiment_id": run_id,
            "runtime_overrides": json.loads((config.results_dir / "runtime_overrides.json").read_text(encoding="utf-8"))
            if (config.results_dir / "runtime_overrides.json").is_file()
            else {},
            "status": manifest.get("status"),
            "requested_steps": steps,
            "validation_subset_size": int(config.grpo["curriculum_phase_evaluation_size"]),
            "completed_steps": sum(int(phase.get("completed_steps", 0)) for phase in phases),
            "adapter_changed_in_each_phase": changed,
            "sampled_base_weights_unchanged": base_unchanged,
            "finite_training_metrics": finite_metrics,
            "resume_test_passed": bool(resume_checkpoint and Path(resume_checkpoint).is_dir()),
            "adapter_reload_passed": bool(reload_model is not None),
            "reload_probe": reload_probe,
            "rollouts": rollouts,
            "baseline_validation": json.loads(baseline_path.read_text(encoding="utf-8")),
            "smoke_validation": manifest.get("selected_checkpoint", {}).get("metrics"),
            "note": (
                "No within-group reward variation was observed; direct exact-reward GRPO has no useful signal yet."
                if rollouts["reward_varying_groups"] == 0
                else "Within-group reward variation was observed in the smoke run."
            ),
        }
        write_json(config.results_dir / "smoke_report.json", report)
    tracker.write_summary(config.results_dir / "gpu_budget_summary.json")
    _emit({"smoke": report, "budget": tracker.snapshot()})
    return report


def _run_pilot(config: ProjectConfig) -> dict[str, Any]:
    bundle = _load_bundle(config)
    fingerprint = _project_fingerprint(config, bundle.manifest.to_dict())
    calibration_path = config.results_dir / "calibration.json"
    progress_path = _pilot_progress_path(config)
    if progress_path.is_file():
        existing_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if existing_progress.get("project_fingerprint") != fingerprint:
            raise ValueError("Pilot progress belongs to a different frozen project fingerprint.")
    if calibration_path.is_file():
        saved = json.loads(calibration_path.read_text(encoding="utf-8"))
        if saved.get("project_fingerprint") == fingerprint:
            _emit(saved)
            return saved

    smoke_path = config.results_dir / "smoke_report.json"
    if not smoke_path.is_file():
        raise FileNotFoundError("Run `rl-countdown smoke` before calibration pilots.")
    tracker = GpuBudget(config.runs_dir, "pilot-calibration", max_hours=_budget_limit(config, "preflight"))
    learning_rates = [float(value) for value in config.grpo["learning_rate_candidates"]]
    expected_steps = int(config.grpo["pilot_steps"])
    pilot_results = []
    with tracker:
        for learning_rate in learning_rates:
            learning_rate_name = f"{learning_rate:.0e}".replace("+", "")
            experiment_id = f"pilot_lr_{learning_rate_name.replace('-', '_')}"
            record = _pilot_existing_record(config, experiment_id)
            if record is None:
                completed = _completed_run(config, experiment_id)
                started = time.perf_counter()
                if completed is None:
                    manifest = train_grpo_experiment(
                        config,
                        bundle,
                        tracker,
                        experiment_id=experiment_id,
                        reward_kind="exact_plus_validity",
                        seed=int(config.grpo["pilot_seed"]),
                        steps=expected_steps,
                        learning_rate=learning_rate,
                        resume=True,
                        learning_rate_pilot=experiment_id,
                        pilot=True,
                    )
                else:
                    manifest = completed
                outer_seconds = time.perf_counter() - started
                run_dir = _run_directory(config, manifest)
                record = {
                    "experiment_id": experiment_id,
                    "learning_rate": learning_rate,
                    "manifest": manifest,
                    "outer_seconds": outer_seconds,
                    "run_directory": str(run_dir.relative_to(config.root).as_posix()),
                }
                _save_pilot_record(config, experiment_id, record, fingerprint)
            pilot_results.append(record)

        scored = []
        for record in pilot_results:
            manifest = record["manifest"]
            selected = manifest.get("selected_checkpoint", {}).get("metrics", {})
            run_dir = config.root / record["run_directory"]
            scored.append(
                {
                    **record,
                    "validation": selected,
                    "rollouts": _run_rollout_statistics(run_dir),
                }
            )
        best = max(
            scored,
            key=lambda record: (
                float(record["validation"].get("solve_rate", 0.0)),
                -float(record["validation"].get("invalid_rate", 1.0)),
                -float(record["learning_rate"]),
            ),
        )
        sft_experiment_id = "pilot_sft_calibration_seed_42"
        sft_record = _pilot_existing_record(config, sft_experiment_id)
        if sft_record is None:
            completed_sft = _completed_run(config, sft_experiment_id)
            started = time.perf_counter()
            if completed_sft is None:
                sft_manifest = train_sft_experiment(
                    config,
                    bundle,
                    tracker,
                    experiment_id=sft_experiment_id,
                    seed=42,
                    steps=8,
                    resume=True,
                )
            else:
                sft_manifest = completed_sft
            sft_outer_seconds = time.perf_counter() - started
            sft_run_dir = _run_directory(config, sft_manifest)
            sft_record = {
                "experiment_id": sft_experiment_id,
                "manifest": sft_manifest,
                "outer_seconds": sft_outer_seconds,
                "run_directory": str(sft_run_dir.relative_to(config.root).as_posix()),
            }
            _save_pilot_record(config, sft_experiment_id, sft_record, fingerprint)
        sft_run_dir = config.root / sft_record["run_directory"]

        def per_step(record: dict[str, Any]) -> float:
            manifest = record["manifest"]
            run_dir = config.root / record["run_directory"]
            phases = manifest.get("phase_runs", [])
            actual_steps = sum(int(phase.get("completed_steps", 0)) for phase in phases)
            training_wall = sum(float(phase.get("train_wall_seconds", 0.0)) for phase in phases)
            eval_wall = sum(
                float(row.get("elapsed_seconds", 0.0))
                for row in _checkpoint_metrics(run_dir)
            )
            if actual_steps <= 0:
                raise RuntimeError(f"Calibration run {manifest['experiment_id']} made no optimizer updates.")
            return max(0.01, (training_wall - eval_wall) / actual_steps)

        grpo_step_seconds = statistics.median(per_step(record) for record in scored)
        sft_step_seconds = per_step(sft_record)
        load_seconds = statistics.median(
            max(
                0.0,
                float(record["outer_seconds"])
                - sum(
                    float(phase.get("train_wall_seconds", 0.0))
                    for phase in record["manifest"].get("phase_runs", [])
                ),
            )
            for record in [*scored, sft_record]
        )
        evaluation_records = [
            row
            for record in [*scored, sft_record]
            for row in _checkpoint_metrics(config.root / record["run_directory"])
        ]
        evaluation_seconds = statistics.median(
            float(record.get("elapsed_seconds", 0.0)) for record in evaluation_records
        ) if evaluation_records else 0.0
        profile = {
            "pilot_grpo_seconds_per_step": grpo_step_seconds,
            "sft_seconds_per_step": sft_step_seconds,
            "model_load_seconds_per_run": load_seconds,
            "validation_seconds_per_checkpoint": evaluation_seconds,
        }
        steps, plan = planned_training_steps(config, profile)
        calibration = {
            "schema_version": 1,
            "project_fingerprint": fingerprint,
            "selected_learning_rate": float(best["learning_rate"]),
            "selected_learning_rate_pilot": best["experiment_id"],
            "pilot_comparison": [
                {
                    "experiment_id": record["experiment_id"],
                    "learning_rate": record["learning_rate"],
                    "validation": record["validation"],
                    "rollouts": record["rollouts"],
                }
                for record in scored
            ],
            "sft_calibration": {
                "experiment_id": sft_experiment_id,
                "steps": 8,
                "run_directory": sft_record["run_directory"],
            },
            "measured_profile": profile,
            "selected_final_step_plan": plan,
            "selected_final_steps": steps,
            "preflight_budget": tracker.snapshot(),
            "calibrated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(calibration_path, calibration)
    tracker.write_summary(config.results_dir / "gpu_budget_summary.json")
    _emit(calibration)
    return calibration


def _load_calibration(config: ProjectConfig, bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    return _check_resume_fingerprint(
        config.results_dir / "calibration.json",
        _project_fingerprint(config, bundle_manifest),
    )


def _early_holdout_decision(config: ProjectConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    run_dir = _run_directory(config, manifest)
    probe_step = int(config.budget["holdout_probe_steps"])
    records = _checkpoint_metrics(run_dir, probe_step)
    baseline_path = config.results_dir / "evaluations" / "base_qwen25_15b" / "baseline_validation_metrics.json"
    if not baseline_path.is_file():
        return {"available": False, "reason": "baseline validation is missing"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not records:
        return {"available": False, "reason": f"no validation record at global step {probe_step}"}
    best = max(records, key=lambda row: float(row["metrics"].get("solve_rate", 0.0)))
    rollout_stats = _run_rollout_statistics(
        run_dir,
        problem_limit=int(config.budget["uninformative_group_probe_size"]),
    )
    delta_pp = 100 * (float(best["metrics"]["solve_rate"]) - float(baseline["solve_rate"]))
    minimum_gain = float(config.budget["minimum_early_gain_percentage_points"])
    minimum_positive_rate = float(config.budget["uninformative_group_min_accuracy"])
    has_signal = (
        rollout_stats["reward_varying_groups"] > 0
        and rollout_stats["positive_group_rate"] >= minimum_positive_rate
    )
    return {
        "available": True,
        "probe_global_step": probe_step,
        "baseline_validation_accuracy": float(baseline["solve_rate"]),
        "probe_validation_accuracy": float(best["metrics"]["solve_rate"]),
        "accuracy_change_percentage_points": delta_pp,
        "minimum_gain_percentage_points": minimum_gain,
        "reward_signal": has_signal,
        "minimum_positive_group_rate": minimum_positive_rate,
        "rollout_statistics": rollout_stats,
        "continue_all_seeds": has_signal and delta_pp >= minimum_gain,
        "note": (
            "Continue with three seeds."
            if has_signal and delta_pp >= minimum_gain
            else (
                "Keep the completed seed-42 model as a measured pilot result; "
                "prioritize controls and honest evaluation."
            )
        ),
    }


def _run_grpo(config: ProjectConfig, resume: bool) -> list[dict[str, Any]]:
    bundle = _load_bundle(config)
    calibration = _load_calibration(config, bundle.manifest.to_dict())
    plan = calibration["selected_final_step_plan"]
    steps = int(plan["selected_steps_per_experiment"])
    seeds = [int(seed) for seed in plan["selected_grpo_seeds"]]
    learning_rate = float(calibration["selected_learning_rate"])
    tracker = GpuBudget(config.runs_dir, "final-grpo-training", max_hours=_budget_limit(config, "training"))
    results = []
    continue_primary_seeds = True
    with tracker:
        for seed in seeds:
            experiment_id = f"grpo_correct_seed_{seed}"
            completed = _registered_manifest(config, experiment_id)
            if completed and completed.get("status") == "completed":
                results.append(completed)
                continue
            if completed and completed.get("probe_accepted"):
                results.append(completed)
                continue_primary_seeds = False
                break
            probe_manifest = None
            if seed == 42 and steps > int(config.budget["holdout_probe_steps"]):
                candidate_probe = _completed_run(config, experiment_id)
                if candidate_probe is None:
                    candidate_probe = train_grpo_experiment(
                        config,
                        bundle,
                        tracker,
                        experiment_id=experiment_id,
                        reward_kind="exact_plus_validity",
                        seed=seed,
                        steps=steps,
                        learning_rate=learning_rate,
                        resume=resume,
                        learning_rate_pilot=calibration["selected_learning_rate_pilot"],
                        stop_at_global_step=int(config.budget["holdout_probe_steps"]),
                    )
                if candidate_probe.get("status") == "probe_complete":
                    probe_manifest = candidate_probe
                    decision = _early_holdout_decision(config, probe_manifest)
                    write_json(config.results_dir / "early_holdout_decision.json", decision)
                    _emit({"early_holdout_decision": decision})
                    if not decision.get("continue_all_seeds", False):
                        continue_primary_seeds = False
                        manifest = finalize_grpo_probe(config, candidate_probe)
                    else:
                        manifest = train_grpo_experiment(
                            config,
                            bundle,
                            tracker,
                            experiment_id=experiment_id,
                            reward_kind="exact_plus_validity",
                            seed=seed,
                            steps=steps,
                            learning_rate=learning_rate,
                            resume=True,
                            learning_rate_pilot=calibration["selected_learning_rate_pilot"],
                        )
                else:
                    manifest = candidate_probe
            else:
                manifest = train_grpo_experiment(
                    config,
                    bundle,
                    tracker,
                    experiment_id=experiment_id,
                    reward_kind="exact_plus_validity",
                    seed=seed,
                    steps=steps,
                    learning_rate=learning_rate,
                    resume=resume,
                    learning_rate_pilot=calibration["selected_learning_rate_pilot"],
                )
            results.append(manifest)
            if seed == 42 and not continue_primary_seeds:
                break
            if (manifest.get("status") != "completed" and not manifest.get("probe_accepted")) or tracker.should_stop:
                break

        if plan.get("run_validity_ablation") and not tracker.should_stop:
            experiment_id = "grpo_validity_seed_42"
            completed = _registered_manifest(config, experiment_id)
            if completed is None or completed.get("status") != "completed":
                results.append(
                    train_grpo_experiment(
                        config,
                        bundle,
                        tracker,
                        experiment_id=experiment_id,
                        reward_kind="expression_validity",
                        seed=int(config.grpo["ablation_seed"]),
                        steps=steps,
                        learning_rate=learning_rate,
                        resume=resume,
                        learning_rate_pilot=calibration["selected_learning_rate_pilot"],
                    )
                )
        if tracker.should_stop:
            print("Training reached its reserved GPU-hour boundary; the last checkpoint is resumable.")
    tracker.write_summary(config.results_dir / "gpu_budget_summary.json")
    _emit({"runs": results, "budget": tracker.snapshot()})
    return results


def _run_sft(config: ProjectConfig, resume: bool) -> dict[str, Any] | None:
    bundle = _load_bundle(config)
    calibration = _load_calibration(config, bundle.manifest.to_dict())
    if not calibration["selected_final_step_plan"].get("run_sft_control", False):
        print("The measured 56-hour training forecast reserved the available time for the primary GRPO run only.")
        return None
    experiment_id = "sft_witness_seed_42"
    completed = _registered_manifest(config, experiment_id)
    if completed and completed.get("status") == "completed":
        _emit(completed)
        return completed
    tracker = GpuBudget(config.runs_dir, "final-sft-control", max_hours=_budget_limit(config, "training"))
    with tracker:
        manifest = train_sft_experiment(
            config,
            bundle,
            tracker,
            experiment_id=experiment_id,
            seed=42,
            steps=int(calibration["selected_final_steps"]),
            resume=resume,
        )
    tracker.write_summary(config.results_dir / "gpu_budget_summary.json")
    _emit({"sft_run": manifest, "budget": tracker.snapshot()})
    return manifest


def _run_evaluate(config: ProjectConfig) -> dict[str, Any]:
    bundle = _load_bundle(config)
    tracker = GpuBudget(config.runs_dir, "final-evaluation", max_hours=_budget_limit(config, "evaluation"))
    with tracker:
        results = run_full_evaluation(config, bundle, tracker)
    tracker.write_summary(config.results_dir / "gpu_budget_summary.json")
    _emit({"evaluation": results, "budget": tracker.snapshot()})
    return results


def _run_predict(config: ProjectConfig, numbers: list[int], target: int, adapter: str | None) -> dict[str, Any]:
    from peft import PeftModel

    from rl_countdown.core import Problem
    from rl_countdown.training import generate_for_problem

    if len(numbers) not in (3, 4) or any(number <= 0 for number in numbers):
        raise ValueError("Pass exactly three or four positive integers after --nums.")
    tracker = GpuBudget(config.runs_dir, "local-prediction", max_hours=_budget_limit(config, "prediction"))
    with tracker:
        model, tokenizer = load_base_model(config)
        if adapter:
            adapter_path = Path(adapter).resolve()
            if not (adapter_path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"No PEFT adapter_config.json under {adapter_path}.")
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        problem = Problem("local-prediction", "validation", tuple(numbers), target)
        prediction = generate_for_problem(model, tokenizer, problem, config)
        prediction["adapter"] = adapter
        prediction["budget"] = tracker.snapshot()
    tracker.write_summary(config.results_dir / "gpu_budget_summary.json")
    _emit(prediction)
    return prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-countdown", description="Run the frozen Countdown GRPO study.")
    commands = parser.add_subparsers(dest="command", required=True)

    def with_config(command: str, help_text: str) -> argparse.ArgumentParser:
        child = commands.add_parser(command, help=help_text)
        child.add_argument("--config", default="configs/default.yaml", help="Project configuration file.")
        return child

    with_config("doctor", "Check CUDA, BF16, GPU VRAM, and runtime prerequisites.")
    with_config("prepare-data", "Generate and verify the frozen synthetic dataset.")
    with_config("baseline", "Measure the unmodified model on the fixed validation subset.")
    with_config("smoke", "Run an 8-step direct-GRPO GPU smoke test.")
    with_config("pilot", "Compare learning rates and calibrate the GPU-hour training plan.")
    grpo = with_config("train-grpo", "Run the preregistered direct-GRPO experiments.")
    grpo.add_argument("--resume", action="store_true", help="Resume matching incomplete checkpoints.")
    sft = with_config("train-sft", "Train the independent witness-supervised control.")
    sft.add_argument("--resume", action="store_true", help="Resume matching incomplete checkpoints.")
    with_config("evaluate", "Evaluate the frozen ID and OOD test sets, resuming raw predictions.")
    with_config("report", "Build the measured-results report and figures.")
    predict = with_config("predict", "Generate and verify one expression locally.")
    predict.add_argument("--nums", nargs="+", required=True, type=int)
    predict.add_argument("--target", required=True, type=int)
    predict.add_argument("--adapter", help="Path to a saved PEFT adapter directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = _apply_runtime_overrides(load_config())
        if arguments.config != "configs/default.yaml":
            custom = Path(arguments.config)
            if custom.is_absolute():
                config_path = custom
            else:
                config_path = (config.root / custom).resolve()
            expected = (config.root / "configs" / "default.yaml").resolve()
            if config_path != expected:
                raise ValueError(
                    "The protocol freezes one shared configs/default.yaml file; "
                    "alternate configuration paths are unsupported."
                )

        if arguments.command == "doctor":
            report = cuda_diagnostic()
            report["project_root"] = str(config.root)
            report["data_ready"] = config.data_dir.is_dir()
            report["model_revision"] = config.project["model_revision"]
            import shutil

            disk = shutil.disk_usage(config.root)
            report["disk_free_gib"] = round(disk.free / (1024**3), 2)
            report["model_download_space_recommended_gib"] = 50
            if disk.free < 50 * (1024**3):
                report["warning"] = "Free disk space is below the 50 GiB model and checkpoint recommendation."
            _emit(report)
            return 0 if report["ok"] else 2
        if arguments.command == "prepare-data":
            manifest = prepare_data(config.values, config.data_dir)
            _emit({"data_directory": config.data_dir, "manifest": manifest})
            return 0
        if arguments.command == "baseline":
            _run_baseline(config)
            return 0
        if arguments.command == "smoke":
            _run_smoke(config)
            return 0
        if arguments.command == "pilot":
            _run_pilot(config)
            return 0
        if arguments.command == "train-grpo":
            _run_grpo(config, arguments.resume)
            return 0
        if arguments.command == "train-sft":
            _run_sft(config, arguments.resume)
            return 0
        if arguments.command == "evaluate":
            _run_evaluate(config)
            return 0
        if arguments.command == "report":
            _emit(generate_report(config))
            return 0
        if arguments.command == "predict":
            _run_predict(config, arguments.nums, arguments.target, arguments.adapter)
            return 0
        parser.error(f"Unknown command: {arguments.command}")
    except KeyboardInterrupt:
        print("Interrupted. The last saved checkpoint or prediction log can be resumed.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0
