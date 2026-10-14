"""Generate compact, honest portfolio artifacts from measured run files."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from rl_countdown.config import ProjectConfig
from rl_countdown.evaluation import FINAL_EXPERIMENT_IDS
from rl_countdown.modeling import discover_adapter_registry
from rl_countdown.training import now_utc, write_json


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def paired_bootstrap_interval(
    differences: list[float], *, samples: int, seed: int
) -> tuple[float, float, float]:
    """Percentile interval for paired, per-problem accuracy differences."""
    if not differences:
        raise ValueError("At least one paired problem is required for a bootstrap interval.")
    generator = random.Random(seed)
    count = len(differences)
    means = [
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    means.sort()
    lower = means[max(0, math.floor(0.025 * samples))]
    upper = means[min(samples - 1, math.ceil(0.975 * samples) - 1)]
    return sum(differences) / count, lower, upper


def _predictions(config: ProjectConfig, experiment_id: str, split: str) -> dict[str, dict[str, Any]]:
    path = (
        config.results_dir
        / "evaluations"
        / experiment_id
        / f"predictions_{split}_greedy.jsonl"
    )
    return {str(row["problem_id"]): row for row in _read_jsonl(path)}


def _metric(config: ProjectConfig, experiment_id: str, split: str, mode: str = "greedy") -> dict[str, Any] | None:
    path = config.results_dir / "evaluations" / experiment_id / f"metrics_{split}_{mode}.json"
    return _read_json(path)


def _experiment_runs(config: ProjectConfig) -> dict[str, Path]:
    registry = discover_adapter_registry(config.root)
    runs: dict[str, Path] = {}
    for experiment_id in FINAL_EXPERIMENT_IDS[1:]:
        entries = registry["experiments"].get(experiment_id, [])
        for entry in entries:
            manifest_path = config.root / entry["run_manifest"]
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") in {"completed", "budget_stopped"}:
                    runs[experiment_id] = manifest_path.parent
    return runs


def _main_seed_bootstrap(config: ProjectConfig, seed: int) -> dict[str, Any] | None:
    base = _predictions(config, "base_qwen25_15b", "test_id")
    candidate = _predictions(config, f"grpo_correct_seed_{seed}", "test_id")
    shared = sorted(base.keys() & candidate.keys())
    if not shared:
        return None
    values = [float(candidate[key]["correct"]) - float(base[key]["correct"]) for key in shared]
    average, lower, upper = paired_bootstrap_interval(
        values,
        samples=int(config.evaluation["bootstrap_samples"]),
        seed=int(config.evaluation["bootstrap_seed"]) + seed,
    )
    return {
        "n_paired": len(shared),
        "mean_accuracy_difference": average,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
    }


def _aggregate_main_seeds(config: ProjectConfig) -> dict[str, Any] | None:
    base = _predictions(config, "base_qwen25_15b", "test_id")
    by_seed = {
        seed: _predictions(config, f"grpo_correct_seed_{seed}", "test_id")
        for seed in (42, 43, 44)
    }
    shared = sorted(set(base).intersection(*(set(rows) for rows in by_seed.values())))
    if not shared:
        return None
    per_problem_delta = [
        sum(float(by_seed[seed][key]["correct"]) for seed in by_seed) / len(by_seed)
        - float(base[key]["correct"])
        for key in shared
    ]
    average, lower, upper = paired_bootstrap_interval(
        per_problem_delta,
        samples=int(config.evaluation["bootstrap_samples"]),
        seed=int(config.evaluation["bootstrap_seed"]),
    )
    return {
        "n_paired": len(shared),
        "seed_count": len(by_seed),
        "mean_accuracy_difference": average,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
    }


def _as_percent(value: Any) -> str:
    return "pending" if value is None else f"{100.0 * float(value):.1f}%"


def _figure_accuracy(config: ProjectConfig) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_labels = ["Base", "GRPO 42", "GRPO 43", "GRPO 44", "Validity", "SFT"]
    measured = []
    for label, identifier in zip(all_labels, FINAL_EXPERIMENT_IDS, strict=True):
        metric_id = _metric(config, identifier, "test_id")
        metric_ood = _metric(config, identifier, "test_ood")
        if any(metric and metric.get("count", 0) for metric in (metric_id, metric_ood)):
            measured.append((label, metric_id, metric_ood))
    if not measured:
        return None
    labels = [row[0] for row in measured]
    id_values = [row[1] for row in measured]
    ood_values = [row[2] for row in measured]
    x = list(range(len(labels)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5.5), layout="constrained")
    axis.bar(
        [value - width / 2 for value in x],
        [float(metric["solve_rate"]) if metric and metric.get("solve_rate") is not None else 0 for metric in id_values],
        width,
        label="ID test",
        color="#315ca8",
    )
    axis.bar(
        [value + width / 2 for value in x],
        [
            float(metric["solve_rate"])
            if metric and metric.get("solve_rate") is not None
            else 0
            for metric in ood_values
        ],
        width,
        label="OOD test",
        color="#e09b38",
    )
    axis.set_ylabel("Greedy exact-solution accuracy")
    axis.set_ylim(0, 1)
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path = config.results_dir / "figures" / "accuracy_id_ood.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _figure_validation(config: ProjectConfig, runs: dict[str, Path]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    plotted = 0
    colors = {42: "#315ca8", 43: "#49885a", 44: "#b26042"}
    for experiment_id, run_dir in runs.items():
        if not experiment_id.startswith("grpo_correct_seed_"):
            continue
        seed = int(experiment_id.rsplit("_", 1)[1])
        points: dict[int, float] = {}
        for path in sorted(run_dir.glob("phase-*/checkpoint_evaluations.jsonl")):
            for row in _read_jsonl(path):
                if "metrics" in row:
                    points[int(row["step"])] = float(row["metrics"]["solve_rate"])
        if points:
            ordered = sorted(points.items())
            axis.plot(
                [step for step, _ in ordered],
                [score for _, score in ordered],
                marker="o",
                label=f"seed {seed}",
                color=colors.get(seed),
            )
            plotted += 1
    if plotted == 0:
        plt.close(fig)
        return None
    axis.set_xlabel("Global optimizer step")
    axis.set_ylabel("Validation exact-solution accuracy")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    path = config.results_dir / "figures" / "validation_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_report(config: ProjectConfig) -> dict[str, Any]:
    """Summarize registered training outcomes and currently available evaluation evidence."""
    summary_path = config.results_dir / "evaluation_summary.json"
    evaluation_summary = _read_json(summary_path) or {}
    runs = _experiment_runs(config)
    metric_rows = []
    for experiment_id in FINAL_EXPERIMENT_IDS:
        metric_rows.append(
            {
                "experiment_id": experiment_id,
                "test_id": _metric(config, experiment_id, "test_id"),
                "test_ood": _metric(config, experiment_id, "test_ood"),
                "test_id_sampled": _metric(config, experiment_id, "test_id", "sampled_k4"),
                "test_ood_sampled": _metric(config, experiment_id, "test_ood", "sampled_k4"),
            }
        )
    bootstrap = {
        str(seed): _main_seed_bootstrap(config, seed) for seed in (42, 43, 44)
    }
    aggregate = _aggregate_main_seeds(config)
    budget_path = config.runs_dir / "accounting" / "gpu_budget.sqlite3"
    used_gpu_hours = None
    if budget_path.is_file():
        import sqlite3

        with sqlite3.connect(budget_path) as connection:
            row = connection.execute("SELECT COALESCE(SUM(seconds), 0) FROM sessions").fetchone()
        used_gpu_hours = float(row[0]) / 3600
    figure_files = [path for path in (_figure_accuracy(config), _figure_validation(config, runs)) if path is not None]
    figures = [str(path.relative_to(config.root).as_posix()) for path in figure_files]
    report_figures = [path.relative_to(config.results_dir).as_posix() for path in figure_files]
    results = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "evaluation_complete": bool(evaluation_summary.get("complete", False)),
        "planned_experiments": evaluation_summary.get("planned_experiments", []),
        "measured_experiments": list(evaluation_summary.get("results", {}).keys()),
        "metrics": metric_rows,
        "paired_bootstrap_vs_base": bootstrap,
        "three_seed_primary_estimate": aggregate,
        "gpu_hours_used": used_gpu_hours,
        "gpu_hours_limit": float(config.budget["gpu_hours_total"]),
        "training_runs": {
            experiment_id: str(run_dir.relative_to(config.root).as_posix())
            for experiment_id, run_dir in runs.items()
        },
        "figures": figures,
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.results_dir / "summary.json", results)

    lines = [
        "# Measured results",
        "",
        f"Generated at {results['generated_at_utc']}. The frozen test set is only used after checkpoint selection.",
        f"Full evaluation status: **{'complete' if results['evaluation_complete'] else 'partial'}**.",
        "",
        "## Greedy exact-solution accuracy",
        "",
        "| Experiment | ID accuracy | ID valid | ID n | OOD accuracy | OOD valid | OOD n | ID pass@4 | OOD pass@4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        identifier = row["experiment_id"]
        metrics_id = row["test_id"] or {}
        metrics_ood = row["test_ood"] or {}
        sampled_id = row["test_id_sampled"] or {}
        sampled_ood = row["test_ood_sampled"] or {}
        lines.append(
            "| {name} | {id_acc} | {id_valid} | {id_n}/{id_expected} | {ood_acc} | {ood_valid} | "
            "{ood_n}/{ood_expected} | {id_pass4} | {ood_pass4} |".format(
                name=identifier,
                id_acc=_as_percent(metrics_id.get("solve_rate")),
                id_valid=_as_percent(metrics_id.get("valid_rate")),
                id_n=metrics_id.get("count", 0),
                id_expected=metrics_id.get("expected_count", "—"),
                ood_acc=_as_percent(metrics_ood.get("solve_rate")),
                ood_valid=_as_percent(metrics_ood.get("valid_rate")),
                ood_n=metrics_ood.get("count", 0),
                ood_expected=metrics_ood.get("expected_count", "—"),
                id_pass4=_as_percent(sampled_id.get("pass_at_k")),
                ood_pass4=_as_percent(sampled_ood.get("pass_at_k")),
            )
        )
    lines.extend(["", "## Paired uncertainty", ""])
    if aggregate is None:
        lines.append(
            "The three-seed primary estimate is pending until the base and all three GRPO seed predictions "
            "share test IDs."
        )
    else:
        lines.append(
            "A paired bootstrap over the shared ID test problems estimates the mean three-seed change at "
            f"**{100*aggregate['mean_accuracy_difference']:+.2f} percentage points** "
            f"(95% percentile interval {100*aggregate['ci_95_lower']:+.2f} to "
            f"{100*aggregate['ci_95_upper']:+.2f}; n={aggregate['n_paired']})."
        )
    lines.extend(["", "Per-seed paired changes:", ""])
    for seed, interval in bootstrap.items():
        if interval is None:
            lines.append(f"- Seed {seed}: pending.")
        else:
            lines.append(
                f"- Seed {seed}: {100*interval['mean_accuracy_difference']:+.2f} pp "
                f"(95% CI {100*interval['ci_95_lower']:+.2f}, {100*interval['ci_95_upper']:+.2f}; "
                f"n={interval['n_paired']})."
            )
    decision = _read_json(config.results_dir / "early_holdout_decision.json")
    if decision is not None and not decision.get("continue_all_seeds", True):
        lines.extend(
            [
                "",
                "The pre-registered step-200 validation gate did not pass, so seeds 43 and 44 were not run. "
                f"Seed 42 changed validation accuracy by "
                f"{float(decision['accuracy_change_percentage_points']):+.2f} percentage points "
                f"against the {float(decision['minimum_gain_percentage_points']):.1f}-point threshold. "
                "The result is a budget-stopped single-seed experiment, not a three-seed estimate.",
            ]
        )
    lines.extend(["", "## Run evidence", ""])
    if used_gpu_hours is not None:
        lines.append(f"Accumulated GPU time: {used_gpu_hours:.2f} / {config.budget['gpu_hours_total']} hours.")
    else:
        lines.append("GPU time ledger not found.")
    if runs:
        for experiment_id, run_dir in sorted(runs.items()):
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            selected = manifest.get("selected_checkpoint", {})
            manifest_relative = (run_dir / "run_manifest.json").relative_to(config.root).as_posix()
            lines.append(
                f"- `{experiment_id}`: {manifest.get('status')}, seed {manifest.get('seed')}, "
                f"{manifest.get('max_train_steps')} planned steps, selected step "
                f"{selected.get('global_step', 'unknown')}; manifest `{manifest_relative}`."
            )
    else:
        lines.append("No final training runs have been registered yet.")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "A successful result supports only the stated Countdown task and prompt. It does not show a general "
            "increase in reasoning ability. OOD tests change the integer range only. SFT has privileged witness "
            "labels; GRPO does not. Report negative or partial outcomes as measured.",
            "",
            "## Figures",
            "",
        ]
    )
    if figures:
        lines.extend(f"![{Path(path).stem}]({path})" for path in report_figures)
    else:
        lines.append("Figures will appear after evaluation produces predictions.")
    lines.append("")
    (config.results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _update_readme_results(config, metric_rows)
    return results


def _update_readme_results(config: ProjectConfig, rows: list[dict[str, Any]]) -> None:
    readme_path = config.root / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    start_marker = "<!-- MEASURED_RESULTS_START -->"
    end_marker = "<!-- MEASURED_RESULTS_END -->"
    if start_marker not in content or end_marker not in content:
        return
    table = [
        "| Model / run | ID greedy accuracy | OOD greedy accuracy | Status |",
        "|---|---:|---:|---|",
    ]
    names = {
        "base_qwen25_15b": "Public instruction checkpoint",
        "grpo_correct_seed_42": "Direct GRPO, seed 42",
        "grpo_correct_seed_43": "Direct GRPO, seed 43",
        "grpo_correct_seed_44": "Direct GRPO, seed 44",
        "grpo_validity_seed_42": "Validity-reward ablation, seed 42",
        "sft_witness_seed_42": "SFT control, seed 42",
    }
    for row in rows:
        metric_id = row["test_id"] or {}
        metric_ood = row["test_ood"] or {}
        status = "measured" if metric_id.get("complete") and metric_ood.get("complete") else "partial / pending"
        table.append(
            f"| {names[row['experiment_id']]} | {_as_percent(metric_id.get('solve_rate'))} "
            f"({metric_id.get('count', 0)}/{metric_id.get('expected_count', '—')}) | "
            f"{_as_percent(metric_ood.get('solve_rate'))} "
            f"({metric_ood.get('count', 0)}/{metric_ood.get('expected_count', '—')}) | {status} |"
        )
    prefix, remainder = content.split(start_marker, 1)
    _, suffix = remainder.split(end_marker, 1)
    readme_path.write_text(
        prefix + start_marker + "\n" + "\n".join(table) + "\n" + end_marker + suffix,
        encoding="utf-8",
    )
