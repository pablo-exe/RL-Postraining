"""Direct GRPO, the supervised control, checkpoint evaluation, and run manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_countdown.budget import GpuBudget
from rl_countdown.config import ProjectConfig
from rl_countdown.core import Problem, verify_expression
from rl_countdown.dataset import DatasetBundle, ReferenceSolution, balanced_hash_subset, get_split
from rl_countdown.modeling import (
    discover_adapter_registry,
    load_base_model,
    require_cuda,
    save_adapter_registry,
)
from rl_countdown.prompting import render_completion


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def adapter_state_sha256(model: Any) -> str:
    """Hash trainable LoRA tensors to verify that an optimizer update changed the adapter."""
    import torch
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model)
    if not state:
        state = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not state:
        raise RuntimeError("No trainable adapter parameters were found in the model.")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_base_probe_sha256(model: Any) -> str:
    """Hash fixed samples from the first, middle, and last frozen base tensors."""
    import torch

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    frozen = [(name, parameter) for name, parameter in base_model.named_parameters() if not parameter.requires_grad]
    if not frozen:
        raise RuntimeError("No frozen base parameters were found; LoRA must not unfreeze the base model.")
    digest = hashlib.sha256()
    for name, parameter in (frozen[0], frozen[len(frozen) // 2], frozen[-1]):
        flat = parameter.detach().reshape(-1)
        sample = flat[[0, len(flat) // 2, len(flat) - 1]].to(device="cpu", dtype=torch.float32)
        digest.update(name.encode("utf-8"))
        digest.update(sample.numpy().tobytes())
    return digest.hexdigest()


def rollout_group_statistics(path: Path) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], list[float]] = {}
    completion_count = 0
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                completion_count += 1
                key = (str(row["phase"]), int(row["global_step"]), str(row["problem_id"]))
                groups.setdefault(key, []).append(float(row["reward"]))
    comparable = [values for values in groups.values() if len(values) > 1]
    varied = [values for values in comparable if len(set(values)) > 1]
    positive = sum(any(value > 0 for value in values) for values in comparable)
    return {
        "completion_count": completion_count,
        "reward_groups": len(comparable),
        "reward_varying_groups": len(varied),
        "reward_variation_rate": len(varied) / len(comparable) if comparable else 0.0,
        "groups_with_positive_reward": positive,
        "positive_group_rate": positive / len(comparable) if comparable else 0.0,
    }


def _runtime_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    packages = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
    versions = {"python": os.sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return path


class JsonlWriter:
    """Flush one audit row at a time so interrupted GPU jobs retain their work."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            self._stream.close()


def make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(3).hex()


def current_git_revision(root: Path) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _data_manifest(bundle: DatasetBundle) -> dict[str, Any]:
    return bundle.manifest.to_dict()


def _save_registry_record(
    config: ProjectConfig,
    experiment_id: str,
    run_id: str,
    manifest_path: Path,
    adapter_path: Path,
    method: str,
    seed: int,
    selected_validation: dict[str, Any],
) -> None:
    registry = discover_adapter_registry(config.root)
    experiments: dict[str, Any] = registry["experiments"]
    entry = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "method": method,
        "seed": seed,
        "run_manifest": manifest_path.relative_to(config.root).as_posix(),
        "adapter_path": adapter_path.relative_to(config.root).as_posix(),
        "selected_validation": selected_validation,
        "updated_at_utc": now_utc(),
    }
    old_entries = experiments.get(experiment_id, [])
    if not isinstance(old_entries, list):
        raise ValueError(f"Malformed adapter registry entry for {experiment_id}.")
    old_entries = [old for old in old_entries if old.get("run_id") != run_id]
    old_entries.append(entry)
    experiments[experiment_id] = old_entries
    save_adapter_registry(config.root, registry)


def _messages_for(problem: Problem, config: ProjectConfig) -> list[dict[str, str]]:
    from rl_countdown.prompting import chat_messages

    return chat_messages(problem, config)


def _trainer_records(problems: tuple[Problem, ...], config: ProjectConfig) -> list[dict[str, Any]]:
    return [
        {
            "problem_id": problem.id,
            "prompt": _messages_for(problem, config),
            "nums": list(problem.nums),
            "target": problem.target,
            "arity": problem.arity,
        }
        for problem in problems
    ]


def extract_completion_text(completion: Any) -> str:
    """Recover raw assistant text from the chat-style completion returned by TRL."""
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list) or not completion:
        return ""
    parts: list[str] = []
    for message in completion:
        if isinstance(message, str):
            parts.append(message)
        elif isinstance(message, dict) and isinstance(message.get("content"), str):
            parts.append(message["content"])
        else:
            return ""
    return "\n".join(parts)


def make_reward_function(
    reward_kind: str,
    rollouts: JsonlWriter,
    phase_id: str,
) -> Callable[..., list[float]]:
    if reward_kind not in {"exact_plus_validity", "expression_validity"}:
        raise ValueError(f"Unsupported reward function: {reward_kind}")

    def reward_function(
        completions: list[Any],
        problem_id: list[str],
        nums: list[list[int]],
        target: list[int],
        arity: list[int],
        trainer_state: Any | None = None,
        **kwargs: Any,
    ) -> list[float]:
        sample_count = len(completions)
        if not (len(problem_id) == len(nums) == len(target) == len(arity) == sample_count):
            raise ValueError("TRL returned mismatched completion and problem metadata batches.")
        step = int(getattr(trainer_state, "global_step", 0))
        rewards: list[float] = []
        for raw_completion, identifier, provided_numbers, objective, provided_arity in zip(
            completions,
            problem_id,
            nums,
            target,
            arity,
            strict=True,
        ):
            text = extract_completion_text(raw_completion)
            verification = verify_expression(text, provided_numbers, int(objective))
            if reward_kind == "exact_plus_validity":
                reward = 1.0 if verification.correct else 0.2 if verification.valid else 0.0
            else:
                reward = float(verification.valid)
            rewards.append(reward)
            rollouts.write(
                {
                    "phase": phase_id,
                    "global_step": step,
                    "problem_id": str(identifier),
                    "nums": [int(value) for value in provided_numbers],
                    "target": int(objective),
                    "arity": int(provided_arity),
                    "completion": raw_completion,
                    "normalized_expression": text.strip(),
                    "valid": verification.valid,
                    "correct": verification.correct,
                    "error_code": verification.error_code,
                    "value_numerator": verification.value_numerator,
                    "value_denominator": verification.value_denominator,
                    "reward_kind": reward_kind,
                    "correctness_component": float(verification.correct),
                    "validity_component": float(verification.valid),
                    "reward": reward,
                }
            )
        return rewards

    reward_function.__name__ = f"countdown_{reward_kind}"
    return reward_function


def _score_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot select a checkpoint without validation predictions.")
    total = len(records)
    valid = sum(bool(record["valid"]) for record in records)
    correct = sum(bool(record["correct"]) for record in records)
    by_arity = {}
    for arity in (3, 4):
        group = [record for record in records if record["arity"] == arity]
        by_arity[str(arity)] = {
            "count": len(group),
            "valid_rate": sum(bool(record["valid"]) for record in group) / len(group) if group else 0.0,
            "solve_rate": sum(bool(record["correct"]) for record in group) / len(group) if group else 0.0,
        }
    return {
        "count": total,
        "valid_rate": valid / total,
        "invalid_rate": 1.0 - valid / total,
        "solve_rate": correct / total,
        "correct_count": correct,
        "valid_count": valid,
        "by_arity": by_arity,
    }


def generate_for_problem(
    model: Any,
    tokenizer: Any,
    problem: Problem,
    config: ProjectConfig,
    *,
    greedy: bool = True,
    seed: int | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    import torch

    messages = _messages_for(problem, config)
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    generation_args: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens or config.evaluation["greedy_max_new_tokens"]),
        "do_sample": not greedy,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if not greedy:
        generation_args.update(
            temperature=float(config.evaluation["sampled_temperature"]),
            top_p=float(config.evaluation["sampled_top_p"]),
            top_k=int(config.evaluation["sampled_top_k"]),
        )

    rng_context = torch.random.fork_rng(devices=[model.device.index or 0]) if seed is not None else None
    previous_mode = bool(model.training)
    model.eval()
    try:
        if rng_context is not None:
            rng_context.__enter__()
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**generation_args)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        if rng_context is not None:
            rng_context.__exit__(None, None, None)
        if previous_mode:
            model.train()
    latency_seconds = time.perf_counter() - started
    completion_ids = output_ids[input_ids.shape[-1] :].detach().cpu().tolist()
    raw_completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    eos = tokenizer.eos_token_id
    finish_reason = "eos" if eos is not None and eos in completion_ids else "max_new_tokens"
    verification = verify_expression(raw_completion, problem.nums, problem.target)
    return {
        "problem_id": problem.id,
        "split": problem.split,
        "nums": list(problem.nums),
        "target": problem.target,
        "arity": problem.arity,
        "completion": raw_completion,
        "normalized_expression": raw_completion.strip(),
        "completion_token_count": len(completion_ids),
        "finish_reason": finish_reason,
        "latency_seconds": latency_seconds,
        "valid": verification.valid,
        "correct": verification.correct,
        "error_code": verification.error_code,
        "value_numerator": verification.value_numerator,
        "value_denominator": verification.value_denominator,
    }


def evaluate_problems(
    model: Any,
    tokenizer: Any,
    problems: tuple[Problem, ...],
    config: ProjectConfig,
    *,
    greedy: bool = True,
    samples_per_problem: int = 1,
    seed: int = 1234,
    tracker: GpuBudget | None = None,
    progress_path: Path | None = None,
    experiment_id: str = "validation",
    checkpoint_step: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    for index, problem in enumerate(problems, start=1):
        for sample_index in range(samples_per_problem):
            sample_seed = (
                seed + int(hashlib.sha256(f"{problem.id}:{sample_index}".encode()).hexdigest()[:8], 16)
                if not greedy
                else None
            )
            prediction = generate_for_problem(
                model,
                tokenizer,
                problem,
                config,
                greedy=greedy,
                seed=sample_seed,
            )
            prediction.update(
                experiment_id=experiment_id,
                sample_index=sample_index,
                checkpoint_step=checkpoint_step,
            )
            predictions.append(prediction)
        if tracker is not None and index % 8 == 0:
            tracker.flush()
            tracker.ensure_available()
            if tracker.should_stop:
                break
        if progress_path is not None and index % 32 == 0:
            provisional = _score_predictions(predictions[-index * samples_per_problem :])
            write_json(
                progress_path,
                {
                    "experiment_id": experiment_id,
                    "evaluated_problems": index,
                    "total_problems": len(problems),
                    "checkpoint_step": checkpoint_step,
                    "partial_metrics": provisional,
                    "updated_at_utc": now_utc(),
                },
            )
    return _score_predictions(predictions), predictions


class ValidationCallback:
    """Log held-out validation at checkpoints and request a safe GPU-budget stop."""

    def __new__(
        cls,
        validation_problems: tuple[Problem, ...],
        tokenizer: Any,
        config: ProjectConfig,
        tracker: GpuBudget,
        run_dir: Path,
        experiment_id: str,
        phase_id: str,
        global_step_offset: int = 0,
        model: Any | None = None,
        stop_at_global_step: int | None = None,
    ) -> Any:
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self) -> None:
                self._last_saved_validation_step: int | None = None
                self._last_flush = time.monotonic()
                self.validation_problems = validation_problems
                self.tokenizer = tokenizer
                self.config = config
                self.tracker = tracker
                self.run_dir = run_dir
                self.experiment_id = experiment_id
                self.phase_id = phase_id
                self.global_step_offset = global_step_offset
                self.model = model
                self.stop_at_global_step = stop_at_global_step

            def _record_validation(self, model: Any, step: int, tag: str) -> None:
                if self._last_saved_validation_step == step:
                    return
                started = time.perf_counter()
                metrics, _ = evaluate_problems(
                    model,
                    self.tokenizer,
                    self.validation_problems,
                    self.config,
                    tracker=self.tracker,
                    progress_path=self.run_dir / "validation-progress.json",
                    experiment_id=self.experiment_id,
                    checkpoint_step=step,
                )
                phase_dir = self.run_dir / self.phase_id
                phase_dir.mkdir(parents=True, exist_ok=True)
                writer = JsonlWriter(phase_dir / "checkpoint_evaluations.jsonl")
                try:
                    writer.write(
                        {
                            "experiment_id": self.experiment_id,
                            "phase": self.phase_id,
                            "checkpoint_tag": tag,
                            "step": step,
                            "metrics": metrics,
                            "validation_complete": int(metrics["count"]) == len(self.validation_problems),
                            "elapsed_seconds": time.perf_counter() - started,
                            "evaluated_at_utc": now_utc(),
                        }
                    )
                finally:
                    writer.close()
                self._last_saved_validation_step = step
                if self.tracker.should_stop:
                    write_json(
                        self.run_dir / "budget-stop.json",
                        {
                            "requested_global_step": step,
                            "phase": self.phase_id,
                            "reason": "project GPU budget reached during validation",
                            "timestamp_utc": now_utc(),
                            "budget": self.tracker.snapshot(),
                        },
                    )

            def on_log(
                self,
                args: Any,
                state: Any,
                control: Any,
                logs: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> Any:
                values = logs or {}
                for name, value in values.items():
                    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                        raise FloatingPointError(
                            f"Non-finite trainer metric {name}={value} at step {state.global_step}."
                        )
                writer = JsonlWriter(self.run_dir / "training_metrics.jsonl")
                try:
                    writer.write(
                        {
                            "experiment_id": self.experiment_id,
                            "phase": self.phase_id,
                            "step": self.global_step_offset + int(state.global_step),
                            "metrics": values,
                            "logged_at_utc": now_utc(),
                        }
                    )
                finally:
                    writer.close()
                return control

            def on_step_end(
                self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any
            ) -> Any:
                if time.monotonic() - self._last_flush >= min(20.0, self.tracker.heartbeat_seconds):
                    self.tracker.flush()
                    self._last_flush = time.monotonic()
                    write_json(
                        self.run_dir / "progress.json",
                        {
                            "experiment_id": self.experiment_id,
                            "phase": self.phase_id,
                            "completed_phase_steps": int(state.global_step),
                            "global_step": self.global_step_offset + int(state.global_step),
                            "max_phase_steps": int(args.max_steps),
                            "timestamp_utc": now_utc(),
                            "budget": self.tracker.snapshot(),
                        },
                    )
                if self.tracker.should_stop:
                    control.should_training_stop = True
                if (
                    self.stop_at_global_step is not None
                    and self.global_step_offset + int(state.global_step) >= self.stop_at_global_step
                ):
                    control.should_training_stop = True
                return control

            def on_save(
                self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any
            ) -> Any:
                model_for_evaluation = model if model is not None else self.model
                if model_for_evaluation is None:
                    model_for_evaluation = kwargs.get("model")
                if model_for_evaluation is not None:
                    self._record_validation(
                        model_for_evaluation,
                        self.global_step_offset + int(state.global_step),
                        f"{self.phase_id}-checkpoint-{state.global_step}",
                    )
                return control

            def on_train_end(
                self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any
            ) -> Any:
                model_for_evaluation = model if model is not None else self.model
                if model_for_evaluation is None:
                    model_for_evaluation = kwargs.get("model")
                if model_for_evaluation is not None:
                    self._record_validation(
                        model_for_evaluation,
                        self.global_step_offset + int(state.global_step),
                        f"{self.phase_id}-final",
                    )
                return control

        return _Callback()


def _run_metadata(
    config: ProjectConfig,
    bundle: DatasetBundle,
    experiment_id: str,
    method: str,
    seed: int,
    total_steps: int,
    learning_rate: float,
    learning_rate_pilot: str | None,
    hardware: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "method": method,
        "seed": seed,
        "run_id": run_id,
        "model_id": config.project["model_id"],
        "model_revision": config.project["model_revision"],
        "tokenizer_revision": config.project["tokenizer_revision"],
        "dataset_manifest": _data_manifest(bundle),
        "configuration": config.values,
        "git_revision": current_git_revision(config.root),
        "python": os.sys.version,
        "software_versions": _runtime_versions(),
        "hardware": hardware,
        "max_train_steps": total_steps,
        "learning_rate": learning_rate,
        "selected_learning_rate_pilot": learning_rate_pilot,
        "started_at_utc": now_utc(),
    }


def _step_split(
    problems: tuple[Problem, ...],
    _phase_steps: int,
    _first_phase_fraction: float,
    easy_arity: int,
) -> tuple[tuple[Problem, ...], tuple[Problem, ...]]:
    if not any(problem.arity == easy_arity for problem in problems):
        raise ValueError(f"Train data does not contain the requested initial curriculum arity {easy_arity}.")
    return (
        tuple(problem for problem in problems if problem.arity == easy_arity),
        tuple(problem for problem in problems if problem.arity in (3, 4)),
    )


def _step_evaluations(
    run_dir: Path,
    fallback_path: Path | None = None,
) -> list[dict[str, Any]]:
    source = run_dir / "checkpoint_evaluations.jsonl"
    if not source.exists() and fallback_path is not None:
        source = fallback_path
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if "metrics" in record and "step" in record:
                    if not record.get("validation_complete", True):
                        continue
                    records.append(record)
    return records


def _completed_phase_record(path: Path, requested_steps: int) -> dict[str, Any] | None:
    """Return the completion marker only when its recorded update budget is satisfied."""
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("phase_complete") or int(record.get("completed_steps", 0)) < requested_steps:
        return None
    return record


def select_checkpoint(
    run_dir: Path,
    phase_dirs: tuple[Path, ...],
) -> tuple[Path, dict[str, Any]]:
    """Choose maximum validation accuracy, then invalid rate, then step."""
    evaluations: list[dict[str, Any]] = []
    for phase_dir in phase_dirs:
        for result in _step_evaluations(phase_dir):
            checkpoint_tag = str(result["checkpoint_tag"])
            checkpoint_step = int(result["step"])
            if checkpoint_tag.endswith("-final"):
                adapter_dir = phase_dir / "final-adapter"
            else:
                local_step = int(checkpoint_tag.rsplit("checkpoint-", 1)[1])
                adapter_dir = phase_dir / f"checkpoint-{local_step}"
            if not (adapter_dir / "adapter_config.json").is_file():
                continue
            metrics = result["metrics"]
            evaluations.append(
                {
                    "adapter_dir": adapter_dir,
                    "metrics": metrics,
                    "global_step": checkpoint_step,
                    "checkpoint_tag": checkpoint_tag,
                    "phase": result["phase"],
                }
            )
    if not evaluations:
        raise FileNotFoundError(f"No usable validation checkpoint exists under {run_dir}.")
    best = max(
        evaluations,
        key=lambda item: (
            float(item["metrics"]["solve_rate"]),
            -float(item["metrics"]["invalid_rate"]),
            -int(item["global_step"]),
        ),
    )
    return Path(best["adapter_dir"]), {
        key: value for key, value in best.items() if key != "adapter_dir"
    }


def _phase_callbacks(
    validation_problems: tuple[Problem, ...],
    tokenizer: Any,
    config: ProjectConfig,
    tracker: GpuBudget,
    run_dir: Path,
    experiment_id: str,
    phase_id: str,
    step_offset: int,
    model: Any,
    stop_at_global_step: int | None = None,
) -> list[Any]:
    return [
        ValidationCallback(
            validation_problems,
            tokenizer,
            config,
            tracker,
            run_dir,
            experiment_id,
            phase_id,
            global_step_offset=step_offset,
            model=model,
            stop_at_global_step=stop_at_global_step,
        )
    ]


def make_grpo_args(
    config: ProjectConfig,
    output_dir: Path,
    learning_rate: float,
    seed: int,
    steps: int,
    save_steps: int,
) -> Any:
    from trl import GRPOConfig

    values = config.grpo
    return GRPOConfig(
        output_dir=str(output_dir),
        run_name=output_dir.name,
        learning_rate=learning_rate,
        lr_scheduler_type=str(values["learning_rate_scheduler"]),
        warmup_steps=int(values["warmup_steps"]),
        weight_decay=float(values["weight_decay"]),
        max_grad_norm=float(values["max_grad_norm"]),
        optim=str(values["optimizer"]),
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=int(values["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
        generation_batch_size=int(values["generation_batch_size"]),
        per_device_eval_batch_size=int(values["num_generations"]),
        max_steps=steps,
        num_generations=int(values["num_generations"]),
        max_prompt_length=int(values["max_prompt_length"]),
        max_completion_length=int(values["max_completion_length"]),
        temperature=float(values["temperature"]),
        top_p=float(values["top_p"]),
        top_k=int(values["top_k"]),
        num_iterations=int(values["num_iterations"]),
        loss_type=str(values["loss_type"]),
        scale_rewards=str(values["scale_rewards"]),
        mask_truncated_completions=bool(values["mask_truncated_completions"]),
        beta=float(values["beta"]),
        epsilon=float(values["epsilon"]),
        logging_strategy="steps",
        logging_steps=int(values["logging_steps"]),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=values["save_total_limit"],
        save_only_model=False,
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=int(values["dataloader_num_workers"]),
        dataloader_drop_last=False,
        disable_dropout=bool(values["disable_dropout"]),
        report_to="none",
        seed=seed,
        data_seed=seed,
        use_vllm=False,
        log_completions=False,
        push_to_hub=False,
    )


def _lora_config(config: ProjectConfig) -> Any:
    from peft import LoraConfig, TaskType

    values = config.lora
    return LoraConfig(
        r=int(values["r"]),
        lora_alpha=int(values["lora_alpha"]),
        lora_dropout=float(values["lora_dropout"]),
        bias=str(values["bias"]),
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(values["target_modules"]),
    )


def train_grpo_experiment(
    config: ProjectConfig,
    bundle: DatasetBundle,
    tracker: GpuBudget,
    *,
    experiment_id: str,
    reward_kind: str,
    seed: int,
    steps: int,
    learning_rate: float,
    resume: bool = False,
    learning_rate_pilot: str | None = None,
    pilot: bool = False,
    stop_at_global_step: int | None = None,
    accept_probe_as_final: bool = False,
    checkpoint_save_steps: int | None = None,
) -> dict[str, Any]:
    """Train one direct-GRPO model in a fixed easy-to-mixed curriculum."""
    require_cuda()
    if steps <= 0:
        raise ValueError("A GRPO run requires a positive max-step budget.")

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOTrainer

    trainer_train = get_split(bundle, "train")
    validation = balanced_hash_subset(
        get_split(bundle, "validation"),
        int(config.grpo["curriculum_phase_evaluation_size"]),
        int(config.evaluation["bootstrap_seed"]),
    )
    phase_one_steps, phase_two_steps = _step_split(
        trainer_train,
        steps,
        float(config.grpo["curriculum_first_phase_fraction"]),
        int(config.grpo["curriculum_arity"]),
    )
    phase_sizes = (
        max(1, math.ceil(steps * float(config.grpo["curriculum_first_phase_fraction"]))),
        steps - max(1, math.ceil(steps * float(config.grpo["curriculum_first_phase_fraction"]))),
    )
    experiment_root = config.runs_dir / "experiments" / experiment_id
    existing_run = None
    if resume and experiment_root.exists():
        eligible = []
        for candidate in experiment_root.iterdir():
            manifest_path = candidate / "run_manifest.json"
            if not candidate.is_dir() or not manifest_path.is_file():
                continue
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved.get("experiment_id") == experiment_id and saved.get("status") != "completed":
                eligible.append(candidate)
        if eligible:
            existing_run = max(eligible, key=lambda path: path.name)
    run_id = existing_run.name if existing_run is not None else make_run_id()
    run_dir = experiment_root / run_id
    run_dir.mkdir(parents=True, exist_ok=resume)
    if existing_run is not None:
        saved_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if (
            saved_manifest.get("model_revision") != str(config.project["model_revision"])
            or saved_manifest.get("dataset_manifest") != _data_manifest(bundle)
            or saved_manifest.get("configuration") != config.values
            or saved_manifest.get("seed") != seed
            or saved_manifest.get("reward_kind") != reward_kind
        ):
            raise ValueError("Refusing to resume a GRPO run whose frozen model, dataset, seed, or config changed.")
        manifest = saved_manifest
    hardware = require_cuda()
    if existing_run is None:
        manifest = _run_metadata(
            config,
            bundle,
            experiment_id,
            "grpo",
            seed,
            steps,
            learning_rate,
            learning_rate_pilot,
            hardware,
            run_id,
        )
        manifest["reward_kind"] = reward_kind
        manifest["curriculum"] = {
            "initial_arity": int(config.grpo["curriculum_arity"]),
            "initial_phase_fraction": float(config.grpo["curriculum_first_phase_fraction"]),
        }
        manifest["pilot"] = pilot
    elif int(manifest.get("max_train_steps", 0)) != steps or float(manifest.get("learning_rate", 0.0)) != learning_rate:
        raise ValueError("Refusing to resume a GRPO run with a different step budget or learning rate.")
    write_json(run_dir / "run_manifest.json", manifest)
    rollout_writer = JsonlWriter(run_dir / "rollouts.jsonl")
    phase_dirs: list[Path] = []
    phase_status: list[dict[str, Any]] = []
    tokenizer = None
    try:
        for phase_index, (phase_steps, phase_problems) in enumerate(
            zip(phase_sizes, (phase_one_steps, phase_two_steps), strict=True),
            start=1,
        ):
            if phase_steps <= 0:
                continue
            tracker.ensure_available()
            phase_name = f"phase-{phase_index:02d}"
            phase_dir = run_dir / phase_name
            phase_dir.mkdir(parents=True, exist_ok=True)
            phase_dirs.append(phase_dir)
            phase_completed = phase_dir / "phase_complete.json"
            completed_record = _completed_phase_record(phase_completed, phase_steps) if resume else None
            if completed_record is not None:
                phase_status.append(completed_record)
                continue
            checkpoint = get_last_checkpoint(str(phase_dir)) if resume else None

            set_seed(seed)
            if phase_index == 1:
                model, tokenizer = load_base_model(config)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                peft_config = _lora_config(config)
            else:
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(config.project["model_id"]),
                        revision=str(config.project["tokenizer_revision"]),
                        trust_remote_code=False,
                        use_fast=True,
                    )
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token = tokenizer.eos_token
                base, _ = load_base_model(config)
                first_phase_model_path = run_dir / "phase-01" / "final-adapter"
                model = PeftModel.from_pretrained(
                    base,
                    str(first_phase_model_path),
                    is_trainable=True,
                )
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
                peft_config = None
            tokenizer.padding_side = "left"

            train_dataset = Dataset.from_list(_trainer_records(phase_problems, config))
            reward_function = make_reward_function(
                reward_kind,
                rollout_writer,
                phase_name,
            )
            args = make_grpo_args(
                config,
                phase_dir,
                learning_rate,
                seed,
                phase_steps,
                save_steps=(
                    50
                    if pilot
                    else checkpoint_save_steps or int(config.grpo["steps_per_phase"])
                ),
            )
            callbacks = _phase_callbacks(
                validation,
                tokenizer,
                config,
                tracker,
                run_dir,
                experiment_id,
                phase_name,
                sum(phase_sizes[: phase_index - 1]),
                model,
                stop_at_global_step,
            )
            trainer = GRPOTrainer(
                model=model,
                reward_funcs=reward_function,
                args=args,
                train_dataset=train_dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
                callbacks=callbacks,
            )
            for callback in callbacks:
                callback.model = trainer.model
            before_adapter_hash = adapter_state_sha256(trainer.model)
            before_base_hash = frozen_base_probe_sha256(trainer.model)
            total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
            trainable_parameters = sum(
                parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
            )
            if not trainable_parameters or trainable_parameters >= total_parameters:
                raise RuntimeError("GRPO must update a non-empty LoRA adapter while keeping the base frozen.")
            torch.cuda.reset_peak_memory_stats()
            train_started = time.perf_counter()
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            wall_seconds = time.perf_counter() - train_started
            after_adapter_hash = adapter_state_sha256(trainer.model)
            after_base_hash = frozen_base_probe_sha256(trainer.model)
            trainer.save_model(str(phase_dir / "final-adapter"))
            tokenizer.save_pretrained(phase_dir / "final-adapter")
            phase_record = {
                "phase": phase_name,
                "reward_kind": reward_kind,
                "training_problems": len(phase_problems),
                "requested_steps": phase_steps,
                "completed_steps": int(trainer.state.global_step),
                "start_global_step": sum(phase_sizes[: phase_index - 1]),
                "train_wall_seconds": wall_seconds,
                "train_samples_seen": train_result.metrics.get("train_samples_per_second"),
                "adapter_sha256_before": before_adapter_hash,
                "adapter_sha256_after": after_adapter_hash,
                "frozen_base_probe_sha256_before": before_base_hash,
                "frozen_base_probe_sha256_after": after_base_hash,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "trainable_parameter_fraction": trainable_parameters / total_parameters,
                "peak_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "peak_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
                "resumed_from_checkpoint": checkpoint,
                "phase_complete": int(trainer.state.global_step) >= phase_steps,
                "completed_at_utc": now_utc(),
            }
            phase_status.append(phase_record)
            if phase_record["phase_complete"]:
                write_json(phase_completed, phase_record)
                interrupted_marker = phase_dir / "phase_interrupted.json"
                if interrupted_marker.exists():
                    interrupted_marker.unlink()
            else:
                write_json(phase_dir / "phase_interrupted.json", phase_record)
            tracker.flush()

            if trainer.state.global_step == 0:
                raise RuntimeError(f"{phase_name} did not complete a single GRPO optimizer update.")
            del trainer, model, train_dataset
            torch.cuda.empty_cache()
            if tracker.should_stop:
                break

        if phase_status and len(phase_status) == 2:
            selected_dir, selection = select_checkpoint(run_dir, tuple(phase_dirs))
        else:
            candidates = [
                path / "final-adapter"
                for path in phase_dirs
                if (path / "final-adapter" / "adapter_config.json").exists()
            ]
            if not candidates:
                raise RuntimeError(f"GPU budget ended before {experiment_id} could save a usable adapter.")
            selected_dir = candidates[-1]
            selection = {"checkpoint_tag": "last_budgeted_adapter", "global_step": None}

        selected_destination = run_dir / "adapter_selected"
        if selected_destination.exists():
            shutil.rmtree(selected_destination)
        shutil.copytree(selected_dir, selected_destination)
        training_complete = len(phase_status) == 2 and all(
            int(phase.get("completed_steps", 0)) >= int(phase.get("requested_steps", 0))
            for phase in phase_status
        )
        is_probe = (
            stop_at_global_step is not None
            and not training_complete
            and not tracker.should_stop
            and not accept_probe_as_final
        )
        manifest.update(
            status="completed" if training_complete else "probe_complete" if is_probe else "budget_stopped",
            probe_accepted=bool(accept_probe_as_final and not training_complete),
            completed_at_utc=now_utc(),
            phase_runs=phase_status,
            selected_checkpoint={
                **selection,
                "adapter_path": selected_destination.relative_to(config.root).as_posix(),
            },
            budget=tracker.snapshot(),
            reward_group_statistics=rollout_group_statistics(run_dir / "rollouts.jsonl"),
        )
        manifest_path = write_json(run_dir / "run_manifest.json", manifest)
        if training_complete or not is_probe:
            _save_registry_record(
                config,
                experiment_id,
                run_id,
                manifest_path,
                selected_destination,
                "grpo",
                seed,
                selection.get("metrics", {}),
            )
        return manifest
    except BaseException as error:
        manifest.update(
            status="interrupted_or_failed",
            last_error=f"{type(error).__name__}: {error}",
            updated_at_utc=now_utc(),
            phase_runs=phase_status,
            budget=tracker.snapshot(),
        )
        write_json(run_dir / "run_manifest.json", manifest)
        raise
    finally:
        rollout_writer.close()


def finalize_grpo_probe(config: ProjectConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    """Accept a measured 200-step holdout run as the final seed when its preregistered gate fails."""
    if manifest.get("status") != "probe_complete":
        raise ValueError("Only an intentionally stopped holdout probe can be finalized this way.")
    run_dir = config.runs_dir / "experiments" / str(manifest["experiment_id"]) / str(manifest["run_id"])
    phase_dirs = tuple(sorted(run_dir.glob("phase-[0-9][0-9]")))
    selected_dir, selection = select_checkpoint(run_dir, phase_dirs)
    selected_destination = run_dir / "adapter_selected"
    if selected_destination.exists():
        shutil.rmtree(selected_destination)
    shutil.copytree(selected_dir, selected_destination)
    manifest.update(
        status="budget_stopped",
        probe_accepted=True,
        completed_at_utc=now_utc(),
        reward_group_statistics=rollout_group_statistics(run_dir / "rollouts.jsonl"),
        selected_checkpoint={
            **selection,
            "adapter_path": selected_destination.relative_to(config.root).as_posix(),
        },
    )
    manifest_path = write_json(run_dir / "run_manifest.json", manifest)
    _save_registry_record(
        config,
        str(manifest["experiment_id"]),
        str(manifest["run_id"]),
        manifest_path,
        selected_destination,
        "grpo",
        int(manifest["seed"]),
        selection.get("metrics", {}),
    )
    return manifest


def _sft_rows(
    train_problems: tuple[Problem, ...],
    reference_by_id: dict[str, ReferenceSolution],
    tokenizer: Any,
    config: ProjectConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_length = int(config.values["sft_control"]["max_length"])
    for problem in train_problems:
        witness = reference_by_id[problem.id].reference_expression
        encoded = render_completion(tokenizer, problem, config, witness)
        if len(encoded["input_ids"]) > max_length:
            raise ValueError(
                f"SFT witness {problem.id} exceeds the frozen {max_length}-token context; "
                "do not silently truncate the dataset."
            )
        rows.append(encoded)
    if not rows:
        raise ValueError("An SFT curriculum phase cannot have an empty supervised dataset.")
    return rows


def train_sft_experiment(
    config: ProjectConfig,
    bundle: DatasetBundle,
    tracker: GpuBudget,
    *,
    experiment_id: str,
    seed: int,
    steps: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Train the independent witness-supervised reference with prompt tokens masked."""
    require_cuda()
    if steps <= 0:
        raise ValueError("The SFT reference requires a positive maximum-step budget.")

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import DataCollatorForSeq2Seq, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    references = {item.id: item for item in bundle.witnesses}
    training_problems = get_split(bundle, "train")
    validation = balanced_hash_subset(
        get_split(bundle, "validation"),
        int(config.grpo["curriculum_phase_evaluation_size"]),
        int(config.evaluation["bootstrap_seed"]),
    )
    phase_one, phase_two = _step_split(
        training_problems,
        steps,
        float(config.grpo["curriculum_first_phase_fraction"]),
        int(config.grpo["curriculum_arity"]),
    )
    phase_sizes = (
        max(1, math.ceil(steps * float(config.grpo["curriculum_first_phase_fraction"]))),
        steps - max(1, math.ceil(steps * float(config.grpo["curriculum_first_phase_fraction"]))),
    )
    experiment_root = config.runs_dir / "experiments" / experiment_id
    existing_run = None
    if resume and experiment_root.exists():
        eligible = []
        for candidate in experiment_root.iterdir():
            manifest_path = candidate / "run_manifest.json"
            if not candidate.is_dir() or not manifest_path.is_file():
                continue
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved.get("experiment_id") == experiment_id and saved.get("status") != "completed":
                eligible.append(candidate)
        if eligible:
            existing_run = max(eligible, key=lambda path: path.name)
    run_id = existing_run.name if existing_run is not None else make_run_id()
    run_dir = experiment_root / run_id
    run_dir.mkdir(parents=True, exist_ok=resume)
    if existing_run is not None:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("model_revision") != str(config.project["model_revision"])
            or manifest.get("dataset_manifest") != _data_manifest(bundle)
            or manifest.get("configuration") != config.values
            or manifest.get("seed") != seed
        ):
            raise ValueError("Refusing to resume an SFT run whose frozen model, dataset, seed, or config changed.")
    hardware = require_cuda()
    if existing_run is None:
        manifest = _run_metadata(
            config,
            bundle,
            experiment_id,
            "sft",
            seed,
            steps,
            float(config.values["sft_control"]["learning_rate"]),
            None,
            hardware,
            run_id,
        )
        write_json(run_dir / "run_manifest.json", manifest)
    phase_dirs: list[Path] = []
    phase_status: list[dict[str, Any]] = []
    model = None
    tokenizer = None
    try:
        for phase_index, (phase_steps, phase_problems) in enumerate(
            zip(phase_sizes, (phase_one, phase_two), strict=True),
            start=1,
        ):
            if phase_steps <= 0:
                continue
            tracker.ensure_available()
            phase_name = f"phase-{phase_index:02d}"
            phase_dir = run_dir / phase_name
            phase_dir.mkdir(parents=True, exist_ok=True)
            phase_dirs.append(phase_dir)
            phase_completed = phase_dir / "phase_complete.json"
            completed_record = _completed_phase_record(phase_completed, phase_steps) if resume else None
            if completed_record is not None:
                phase_status.append(completed_record)
                continue
            resume_checkpoint = get_last_checkpoint(str(phase_dir)) if resume else None
            set_seed(seed)
            if phase_index == 1:
                model, tokenizer = load_base_model(config)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                peft_config = _lora_config(config)
            else:
                base, tokenizer = load_base_model(config)
                first_adapter = run_dir / "phase-01" / "final-adapter"
                model = PeftModel.from_pretrained(base, str(first_adapter), is_trainable=True)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
                peft_config = None
            tokenizer.padding_side = "right"
            sft_dataset = Dataset.from_list(
                _sft_rows(phase_problems, references, tokenizer, config)
            )
            values = config.values["sft_control"]
            args = SFTConfig(
                output_dir=str(phase_dir),
                run_name=phase_dir.name,
                learning_rate=float(values["learning_rate"]),
                lr_scheduler_type=str(values["learning_rate_scheduler"]),
                warmup_steps=0,
                weight_decay=0.0,
                optim="adamw_torch",
                bf16=True,
                fp16=False,
                tf32=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                per_device_train_batch_size=int(values["per_device_train_batch_size"]),
                gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
                max_steps=phase_steps,
                logging_strategy="steps",
                logging_steps=int(values["logging_steps"]),
                logging_first_step=True,
                save_strategy="steps",
                save_steps=int(config.grpo["steps_per_phase"]),
                save_total_limit=config.grpo["save_total_limit"],
                save_only_model=False,
                eval_strategy="no",
                remove_unused_columns=False,
                dataloader_num_workers=0,
                report_to="none",
                seed=seed,
                data_seed=seed,
                max_length=int(values["max_length"]),
                packing=bool(values["use_packing"]),
                completion_only_loss=False,
                dataset_num_proc=1,
                dataset_kwargs={"skip_prepare_dataset": True},
            )
            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                label_pad_token_id=-100,
                return_tensors="pt",
            )
            callbacks = _phase_callbacks(
                validation,
                tokenizer,
                config,
                tracker,
                run_dir,
                experiment_id,
                phase_name,
                sum(phase_sizes[: phase_index - 1]),
                model,
            )
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=sft_dataset,
                processing_class=tokenizer,
                data_collator=collator,
                peft_config=peft_config,
                callbacks=callbacks,
            )
            for callback in callbacks:
                callback.model = trainer.model
            total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
            trainable_parameters = sum(
                parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
            )
            if not trainable_parameters or trainable_parameters >= total_parameters:
                raise RuntimeError("SFT must update a non-empty LoRA adapter while keeping the base frozen.")
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            training_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
            wall_seconds = time.perf_counter() - started
            before_adapter_hash = adapter_state_sha256(trainer.model)
            trainer.save_model(str(phase_dir / "final-adapter"))
            tokenizer.save_pretrained(phase_dir / "final-adapter")
            phase_record = {
                "phase": phase_name,
                "requested_steps": phase_steps,
                "completed_steps": int(trainer.state.global_step),
                "start_global_step": sum(phase_sizes[: phase_index - 1]),
                "train_wall_seconds": wall_seconds,
                "train_samples_per_second": training_result.metrics.get("train_samples_per_second"),
                "adapter_sha256": before_adapter_hash,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "trainable_parameter_fraction": trainable_parameters / total_parameters,
                "peak_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "peak_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
                "completed_at_utc": now_utc(),
            }
            phase_status.append(phase_record)
            if int(trainer.state.global_step) >= phase_steps:
                write_json(phase_completed, phase_record)
                interrupted_marker = phase_dir / "phase_interrupted.json"
                if interrupted_marker.exists():
                    interrupted_marker.unlink()
            else:
                write_json(phase_dir / "phase_interrupted.json", phase_record)
            tracker.flush()
            if trainer.state.global_step == 0:
                raise RuntimeError(f"{phase_name} did not complete an SFT optimizer update.")
            del trainer, sft_dataset
            model = None
            tokenizer = None
            torch.cuda.empty_cache()
            if tracker.should_stop:
                break

        if len(phase_status) == 2:
            selected_dir, selection = select_checkpoint(run_dir, tuple(phase_dirs))
        else:
            selected_dir = next(
                (
                    phase_dir / "final-adapter"
                    for phase_dir in reversed(phase_dirs)
                    if (phase_dir / "final-adapter" / "adapter_config.json").exists()
                ),
                None,
            )
            if selected_dir is None:
                raise RuntimeError("GPU budget ended before the SFT reference saved a usable adapter.")
            selection = {"checkpoint_tag": "last_budgeted_adapter", "global_step": None}
        selected_destination = run_dir / "adapter_selected"
        if selected_destination.exists():
            shutil.rmtree(selected_destination)
        shutil.copytree(selected_dir, selected_destination)
        training_complete = len(phase_status) == 2 and all(
            int(phase.get("completed_steps", 0)) >= int(phase.get("requested_steps", 0))
            for phase in phase_status
        )
        manifest.update(
            status="completed" if training_complete else "budget_stopped",
            completed_at_utc=now_utc(),
            phase_runs=phase_status,
            selected_checkpoint={
                **selection,
                "adapter_path": selected_destination.relative_to(config.root).as_posix(),
            },
            budget=tracker.snapshot(),
            reference_witness_count=len(training_problems),
            prompt_loss_masking="prompt token labels set to -100; completion and EOS labels train",
        )
        manifest_path = write_json(run_dir / "run_manifest.json", manifest)
        _save_registry_record(
            config,
            experiment_id,
            run_id,
            manifest_path,
            selected_destination,
            "sft",
            seed,
            selection.get("metrics", {}),
        )
        return manifest
    except BaseException as error:
        manifest.update(
            status="interrupted_or_failed",
            last_error=f"{type(error).__name__}: {error}",
            updated_at_utc=now_utc(),
            phase_runs=phase_status,
            budget=tracker.snapshot(),
        )
        write_json(run_dir / "run_manifest.json", manifest)
        raise


def planned_training_steps(config: ProjectConfig, profile: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Choose the largest preregistered budget fitting the planned final run set."""
    grpo_seconds_per_step = float(profile["pilot_grpo_seconds_per_step"])
    sft_seconds_per_step = float(profile["sft_seconds_per_step"])
    model_load_seconds = float(profile["model_load_seconds_per_run"])
    evaluation_seconds_per_checkpoint = float(profile["validation_seconds_per_checkpoint"])
    candidates = [int(step) for step in config.grpo["final_step_candidates"]]
    final_training_seconds = float(config.budget["final_training_hours"]) * 3600
    margin = float(config.budget["final_training_margin_fraction"])
    def projected(steps: int, grpo_runs: int, include_sft: bool) -> dict[str, float]:
        first_phase = max(
            1,
            math.ceil(steps * float(config.grpo["curriculum_first_phase_fraction"])),
        )
        second_phase = steps - first_phase
        save_interval = int(config.grpo["steps_per_phase"])
        checkpoints_per_run = math.ceil(first_phase / save_interval) + math.ceil(second_phase / save_interval)
        per_grpo_run = (
            steps * grpo_seconds_per_step
            + model_load_seconds
            + checkpoints_per_run * evaluation_seconds_per_checkpoint
        )
        grpo_seconds = grpo_runs * per_grpo_run
        sft_seconds = 0.0
        if include_sft:
            sft_seconds = (
                steps * sft_seconds_per_step
                + model_load_seconds
                + checkpoints_per_run * evaluation_seconds_per_checkpoint
            )
        return {
            "grpo_seconds": grpo_seconds,
            "sft_seconds": sft_seconds,
            "total_seconds": grpo_seconds + sft_seconds,
            "grpo_runs": float(grpo_runs),
            "sft_runs": float(int(include_sft)),
            "validation_checkpoints_per_run": float(checkpoints_per_run),
        }

    selected: dict[str, Any] | None = None
    for steps in candidates:
        forecast = projected(steps, grpo_runs=4, include_sft=True)
        if forecast["total_seconds"] * (1.0 + margin) <= final_training_seconds:
            selected = {
                "steps": steps,
                "forecast": forecast,
                "seeds": [42, 43, 44],
                "run_ablation": True,
                "run_sft": True,
                "single_grpo_seed_only": False,
            }
            break
    if selected is None:
        for steps in candidates:
            forecast = projected(steps, grpo_runs=2, include_sft=True)
            if forecast["total_seconds"] * (1.0 + margin) <= final_training_seconds:
                selected = {
                    "steps": steps,
                    "forecast": forecast,
                    "seeds": [42],
                    "run_ablation": True,
                    "run_sft": True,
                    "single_grpo_seed_only": True,
                }
                break
    if selected is None:
        for steps in candidates:
            forecast = projected(steps, grpo_runs=1, include_sft=False)
            if forecast["total_seconds"] * (1.0 + margin) <= final_training_seconds:
                selected = {
                    "steps": steps,
                    "forecast": forecast,
                    "seeds": [42],
                    "run_ablation": False,
                    "run_sft": False,
                    "single_grpo_seed_only": True,
                }
                break
    if selected is None:
        raise RuntimeError("The measured pilots exceed the prescribed GPU-hour training reserve.")
    record = {
        "selected_steps_per_experiment": selected["steps"],
        "selected_grpo_seeds": selected["seeds"],
        "run_validity_ablation": selected["run_ablation"],
        "run_sft_control": selected["run_sft"],
        "forecast": selected["forecast"],
        "reserve_seconds": final_training_seconds,
        "margin_fraction": margin,
        "single_grpo_seed_only": bool(selected.get("single_grpo_seed_only", False)),
        "selected_at_utc": now_utc(),
    }
    return int(selected["steps"]), record

