"""Load and validate committed project and experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    method: str
    seeds: tuple[int, ...]
    steps: int | str
    reward: str | None
    curriculum: str


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    values: dict[str, Any]

    @property
    def project(self) -> dict[str, Any]:
        return self.values["project"]

    @property
    def data(self) -> dict[str, Any]:
        return self.values["data"]

    @property
    def grpo(self) -> dict[str, Any]:
        return self.values["grpo"]

    @property
    def lora(self) -> dict[str, Any]:
        return self.values["lora"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.values["evaluation"]

    @property
    def budget(self) -> dict[str, Any]:
        return self.values["budget"]

    @property
    def data_dir(self) -> Path:
        return self.root / self.project["data_root"] / "countdown-v1"

    @property
    def runs_dir(self) -> Path:
        return self.root / self.project["runs_root"]

    @property
    def results_dir(self) -> Path:
        return self.root / self.project["results_root"]


def find_project_root(start: Path | None = None) -> Path:
    """Find the checkout that declares this package."""
    base = (start or Path.cwd()).resolve()
    for candidate in (base, *base.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs" / "default.yaml").is_file():
            return candidate
    raise FileNotFoundError("Project root not found. Run this command inside the rl-countdown checkout.")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return values


def load_config(root: Path | None = None) -> ProjectConfig:
    project_root = (root or find_project_root()).resolve()
    values = _load_yaml(project_root / "configs" / "default.yaml")
    required = {"project", "data", "prompt", "verifier", "lora", "grpo", "sft_control", "evaluation", "budget"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"configs/default.yaml is missing sections: {', '.join(missing)}")
    if values["project"]["model_revision"] != values["project"]["tokenizer_revision"]:
        raise ValueError("Pin the model and tokenizer to the same immutable Hugging Face revision.")
    return ProjectConfig(root=project_root, values=values)


def load_experiments(config: ProjectConfig) -> tuple[Experiment, ...]:
    rows = _load_yaml(config.root / "configs" / "experiments.yaml").get("experiments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("configs/experiments.yaml must define one or more experiments.")
    parsed: list[Experiment] = []
    identifiers: set[str] = set()
    for row in rows:
        experiment = Experiment(
            experiment_id=str(row["id"]),
            method=str(row["method"]),
            seeds=tuple(int(seed) for seed in row["seeds"]),
            steps=row["train_steps"],
            reward=row.get("reward"),
            curriculum=str(row["curriculum"]),
        )
        if experiment.experiment_id in identifiers:
            raise ValueError(f"Experiment identifiers must be unique: {experiment.experiment_id}")
        if experiment.method not in {"pretrained_instruct", "grpo", "sft"}:
            raise ValueError(f"Unsupported experiment method: {experiment.method}")
        if experiment.method == "grpo" and experiment.reward not in {"exact_solution", "expression_validity"}:
            raise ValueError(f"Unsupported reward for {experiment.experiment_id}: {experiment.reward}")
        if experiment.method == "pretrained_instruct" and (experiment.steps != 0 or experiment.seeds):
            raise ValueError("The original-model reference must not perform training updates.")
        identifiers.add(experiment.experiment_id)
        parsed.append(experiment)
    return tuple(parsed)

