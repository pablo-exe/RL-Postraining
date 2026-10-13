"""Resumable evaluation for the frozen Countdown test sets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rl_countdown.budget import GpuBudget
from rl_countdown.config import ProjectConfig
from rl_countdown.core import Problem
from rl_countdown.dataset import DatasetBundle, balanced_hash_subset, get_split
from rl_countdown.modeling import discover_adapter_registry, load_base_model, require_cuda
from rl_countdown.training import generate_for_problem, now_utc, write_json

FINAL_EXPERIMENT_IDS = (
    "base_qwen25_15b",
    "grpo_correct_seed_42",
    "grpo_correct_seed_43",
    "grpo_correct_seed_44",
    "grpo_validity_seed_42",
    "sft_witness_seed_42",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith("\n") and index == len(lines) - 1:
            break
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    write_json(path, metrics)


def _single_pass_metrics(records: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    correct_ids = {str(row["problem_id"]) for row in records if row["correct"]}
    valid_ids = {str(row["problem_id"]) for row in records if row["valid"]}
    by_arity: dict[str, dict[str, float | int]] = {}
    for arity in (3, 4):
        selected = [row for row in records if int(row["arity"]) == arity]
        count = len(selected)
        by_arity[str(arity)] = {
            "count": count,
            "valid_rate": sum(bool(row["valid"]) for row in selected) / count if count else 0.0,
            "solve_rate": sum(bool(row["correct"]) for row in selected) / count if count else 0.0,
        }
    count = len(records)
    error_counts = Counter(str(row["error_code"]) for row in records if row["error_code"] is not None)
    finish_counts = Counter(str(row["finish_reason"]) for row in records)
    return {
        "metric": "greedy_accuracy",
        "count": count,
        "expected_count": expected_count,
        "complete": count == expected_count,
        "valid_count": len(valid_ids),
        "correct_count": len(correct_ids),
        "valid_rate": len(valid_ids) / count if count else None,
        "invalid_rate": 1.0 - len(valid_ids) / count if count else None,
        "solve_rate": len(correct_ids) / count if count else None,
        "conditional_accuracy": len(correct_ids) / len(valid_ids) if valid_ids else 0.0,
        "error_counts": dict(sorted(error_counts.items())),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "max_new_tokens_rate": finish_counts.get("max_new_tokens", 0) / count if count else None,
        "mean_completion_tokens": (
            sum(int(row["completion_token_count"]) for row in records) / count if count else None
        ),
        "mean_latency_seconds": (
            sum(float(row["latency_seconds"]) for row in records) / count if count else None
        ),
        "by_arity": by_arity,
        "updated_at_utc": now_utc(),
    }


def _sampled_metrics(records: list[dict[str, Any]], expected_problems: int, samples_per_problem: int) -> dict[str, Any]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_problem[str(row["problem_id"])].append(row)
    complete_groups = [
        sorted(group, key=lambda row: int(row["sample_index"]))
        for group in by_problem.values()
        if len(group) == samples_per_problem
    ]
    total_samples = sum(len(group) for group in complete_groups)
    return {
        "metric": "sampled_pass_at_k",
        "count": len(complete_groups),
        "expected_count": expected_problems,
        "complete": len(complete_groups) == expected_problems,
        "samples_per_problem": samples_per_problem,
        "pass_at_1": (
            sum(bool(row["correct"]) for group in complete_groups for row in group) / total_samples
            if total_samples
            else None
        ),
        "pass_at_k": (
            sum(any(bool(row["correct"]) for row in group) for group in complete_groups) / len(complete_groups)
            if complete_groups
            else None
        ),
        "valid_rate": (
            sum(bool(row["valid"]) for group in complete_groups for row in group) / total_samples
            if total_samples
            else None
        ),
        "mean_completion_tokens": (
            sum(int(row["completion_token_count"]) for group in complete_groups for row in group) / total_samples
            if total_samples
            else None
        ),
        "mean_latency_seconds": (
            sum(float(row["latency_seconds"]) for group in complete_groups for row in group) / total_samples
            if total_samples
            else None
        ),
        "updated_at_utc": now_utc(),
    }


def _evaluate_file(
    model: Any,
    tokenizer: Any,
    problems: tuple[Problem, ...],
    config: ProjectConfig,
    tracker: GpuBudget,
    *,
    experiment_id: str,
    split: str,
    sampled: bool,
    samples_per_problem: int,
    seed: int,
) -> dict[str, Any]:
    experiment_dir = config.results_dir / "evaluations" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    mode = f"sampled_k{samples_per_problem}" if sampled else "greedy"
    prediction_path = experiment_dir / f"predictions_{split}_{mode}.jsonl"
    metrics_path = experiment_dir / f"metrics_{split}_{mode}.json"
    progress_path = experiment_dir / f"progress_{split}_{mode}.json"
    existing = _read_jsonl(prediction_path)
    completed_keys = {
        (str(row["problem_id"]), int(row["sample_index"])) for row in existing
    }
    total_calls = len(problems) * samples_per_problem
    records = list(existing)
    with prediction_path.open("a", encoding="utf-8", newline="\n") as stream:
        calls_since_sync = 0
        for problem in problems:
            for sample_index in range(samples_per_problem):
                key = (problem.id, sample_index)
                if key in completed_keys:
                    continue
                sample_seed = (
                    seed + int(hashlib.sha256(f"{problem.id}:{sample_index}".encode()).hexdigest()[:8], 16)
                    if sampled
                    else None
                )
                record = generate_for_problem(
                    model,
                    tokenizer,
                    problem,
                    config,
                    greedy=not sampled,
                    seed=sample_seed,
                )
                record.update(
                    experiment_id=experiment_id,
                    sample_index=sample_index,
                    evaluation_mode=mode,
                )
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                calls_since_sync += 1
                if calls_since_sync >= 8:
                    stream.flush()
                    calls_since_sync = 0
                    tracker.flush()
                    tracker.ensure_available()
                    records.append(record)
                    provisional = (
                        _single_pass_metrics(records, len(problems))
                        if not sampled
                        else _sampled_metrics(records, len(problems), samples_per_problem)
                    )
                    write_json(
                        progress_path,
                        {
                            "experiment_id": experiment_id,
                            "split": split,
                            "mode": mode,
                            "completed_calls": len(records),
                            "expected_calls": total_calls,
                            "partial_metrics": provisional,
                            "updated_at_utc": now_utc(),
                        },
                    )
                    if tracker.should_stop:
                        break
            if tracker.should_stop:
                break
        stream.flush()

    records = _read_jsonl(prediction_path)
    metrics = (
        _single_pass_metrics(records, len(problems))
        if not sampled
        else _sampled_metrics(records, len(problems), samples_per_problem)
    )
    metrics.update(
        experiment_id=experiment_id,
        split=split,
        evaluation_mode=mode,
        predictions_path=prediction_path.relative_to(config.root).as_posix(),
    )
    _write_metrics(metrics_path, metrics)
    if metrics["complete"] and progress_path.exists():
        progress_path.unlink()
    return metrics


def _selected_adapters(config: ProjectConfig) -> list[dict[str, Any]]:
    registry = discover_adapter_registry(config.root)
    selected: list[dict[str, Any]] = []
    for experiment_id in FINAL_EXPERIMENT_IDS[1:]:
        candidates = registry["experiments"].get(experiment_id, [])
        valid: list[dict[str, Any]] = []
        for entry in candidates:
            manifest_path = (config.root / entry["run_manifest"]).resolve()
            adapter_path = (config.root / entry["adapter_path"]).resolve()
            if config.root not in manifest_path.parents or config.root not in adapter_path.parents:
                raise ValueError("Experiment registry paths must stay inside the project checkout.")
            if not manifest_path.is_file() or not (adapter_path / "adapter_config.json").is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") not in {"completed", "budget_stopped"}:
                continue
            entry_with_manifest = dict(entry)
            entry_with_manifest["manifest"] = manifest
            valid.append(entry_with_manifest)
        if valid:
            selected.append(max(valid, key=lambda entry: str(entry.get("updated_at_utc", ""))))
    return selected


def _load_all_adapters(model: Any, adapters: list[dict[str, Any]], config: ProjectConfig) -> tuple[Any, dict[str, str]]:
    from peft import PeftModel

    if not adapters:
        return model, {}
    names: dict[str, str] = {}
    first = adapters[0]
    first_name = first["experiment_id"].replace("-", "_")
    model = PeftModel.from_pretrained(
        model,
        str((config.root / first["adapter_path"]).resolve()),
        adapter_name=first_name,
        is_trainable=False,
    )
    names[first["experiment_id"]] = first_name
    for entry in adapters[1:]:
        adapter_name = entry["experiment_id"].replace("-", "_")
        model.load_adapter(
            str((config.root / entry["adapter_path"]).resolve()),
            adapter_name=adapter_name,
            is_trainable=False,
        )
        names[entry["experiment_id"]] = adapter_name
    return model, names


def run_full_evaluation(config: ProjectConfig, bundle: DatasetBundle, tracker: GpuBudget) -> dict[str, Any]:
    """Evaluate base plus every registered final adapter on frozen ID and OOD tests."""
    require_cuda()
    adapters = _selected_adapters(config)
    model, tokenizer = load_base_model(config)
    models: list[tuple[str, str | None]] = [(FINAL_EXPERIMENT_IDS[0], None)]
    models.extend((entry["experiment_id"], entry["experiment_id"]) for entry in adapters)
    split_problems = {name: get_split(bundle, name) for name in ("validation", "test_id", "test_ood")}
    sample_sizes = {
        "test_id": int(config.evaluation["sampled_test_id_subset_size"]),
        "test_ood": int(config.evaluation["sampled_test_ood_subset_size"]),
    }
    sampled_problems: dict[str, tuple[Problem, ...]] = {}
    for split in ("test_id", "test_ood"):
        sampled_problems[split] = balanced_hash_subset(
            split_problems[split],
            min(sample_sizes[split], len(split_problems[split])),
            int(config.evaluation["bootstrap_seed"]) + (0 if split == "test_id" else 1),
        )
    all_metrics: dict[str, Any] = {
        "schema_version": 1,
        "model_revision": config.project["model_revision"],
        "dataset_manifest": bundle.manifest.to_dict(),
        "planned_experiments": [experiment_id for experiment_id, _ in models],
        "results": {},
        "started_at_utc": now_utc(),
    }
    adapter_names: dict[str, str] = {}
    for experiment_id, adapter_name in models:
        if tracker.should_stop:
            break
        if adapter_name is not None:
            if not adapter_names:
                model, adapter_names = _load_all_adapters(model, adapters, config)
            adapter_name = adapter_names[experiment_id]
            model.set_adapter(adapter_name)
        results: dict[str, Any] = {}
        for split, problems in split_problems.items():
            if tracker.should_stop:
                break
            results[f"{split}_greedy"] = _evaluate_file(
                model,
                tokenizer,
                problems,
                config,
                tracker,
                experiment_id=experiment_id,
                split=split,
                sampled=False,
                samples_per_problem=1,
                seed=int(config.evaluation["bootstrap_seed"]),
            )
            if tracker.should_stop:
                break
            if split == "validation":
                continue
            results[f"{split}_sampled"] = _evaluate_file(
                model,
                tokenizer,
                sampled_problems[split],
                config,
                tracker,
                experiment_id=experiment_id,
                split=split,
                sampled=True,
                samples_per_problem=int(config.evaluation["sampled_count"]),
                seed=int(config.evaluation["bootstrap_seed"]),
            )
        all_metrics["results"][experiment_id] = results
        write_json(config.results_dir / "evaluation_summary.json", all_metrics)
    all_metrics.update(
        complete=len(all_metrics["results"]) == len(models)
        and all(
            metric.get("complete", False)
            for model_results in all_metrics["results"].values()
            for metric in model_results.values()
        ),
        finished_at_utc=now_utc(),
        budget=tracker.snapshot(),
    )
    write_json(config.results_dir / "evaluation_summary.json", all_metrics)
    return all_metrics


def evaluate_baseline(config: ProjectConfig, bundle: DatasetBundle, tracker: GpuBudget) -> dict[str, Any]:
    """Measure a quick 128-example validation baseline before starting RL."""
    from rl_countdown.training import evaluate_problems

    require_cuda()
    model, tokenizer = load_base_model(config)
    problems = balanced_hash_subset(
        get_split(bundle, "validation"),
        int(config.grpo["eval_subset_size"]),
        int(config.evaluation["bootstrap_seed"]),
    )
    metrics, predictions = evaluate_problems(
        model,
        tokenizer,
        problems,
        config,
        tracker=tracker,
        experiment_id=FINAL_EXPERIMENT_IDS[0],
        seed=int(config.evaluation["bootstrap_seed"]),
    )
    destination = config.results_dir / "evaluations" / FINAL_EXPERIMENT_IDS[0]
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "baseline_validation_metrics.json", metrics)
    path = destination / "predictions_validation_baseline.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    return metrics
