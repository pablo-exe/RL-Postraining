from __future__ import annotations

import json
from types import SimpleNamespace

from rl_countdown.core import Problem
from rl_countdown.evaluation import _evaluate_file


class _Tracker:
    def __init__(self) -> None:
        self._flushed = False

    @property
    def should_stop(self) -> bool:
        return self._flushed

    def flush(self) -> None:
        self._flushed = True

    def ensure_available(self) -> None:
        pass


def test_sampled_progress_counts_every_saved_completion(tmp_path, monkeypatch) -> None:
    import rl_countdown.evaluation as evaluation

    def fake_generate(_model, _tokenizer, problem, _config, **_kwargs):
        return {
            "problem_id": problem.id,
            "arity": 3,
            "correct": True,
            "valid": True,
            "error_code": None,
            "finish_reason": "eos",
            "completion_token_count": 4,
            "latency_seconds": 0.1,
        }

    monkeypatch.setattr(evaluation, "generate_for_problem", fake_generate)
    config = SimpleNamespace(root=tmp_path, results_dir=tmp_path / "results", evaluation={})
    problems = tuple(Problem(f"problem-{index}", "test_id", (1, 2, 3), 6) for index in range(3))

    metrics = _evaluate_file(
        object(),
        object(),
        problems,
        config,
        _Tracker(),
        experiment_id="test",
        split="test_id",
        sampled=True,
        samples_per_problem=4,
        seed=7,
    )

    progress_path = tmp_path / "results" / "evaluations" / "test" / "progress_test_id_sampled_k4.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["completed_calls"] == 8
    assert progress["partial_metrics"]["count"] == 2
    assert metrics["count"] == 2
    assert not metrics["complete"]
    assert metrics["pass_at_k"] == 1.0
