from __future__ import annotations

import json

from rl_countdown.config import load_config
from rl_countdown.core import Problem
from rl_countdown.prompting import render_completion
from rl_countdown.training import JsonlWriter, make_reward_function


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert messages[0]["role"] == "system"
        return [11, 12]

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        assert text.endswith(self.eos_token)
        return {"input_ids": [21, 22]}


def test_sft_encoding_masks_prompt_and_trains_completion() -> None:
    config = load_config()
    encoded = render_completion(
        FakeTokenizer(),
        Problem("p1", "train", (2, 3, 7), 20),
        config,
        witness="(7 + 3) * 2",
    )
    assert encoded["input_ids"] == [11, 12, 21, 22]
    assert encoded["attention_mask"] == [1, 1, 1, 1]
    assert encoded["labels"] == [-100, -100, 21, 22]


def test_reward_function_shaping_distinguishes_exact_solution_from_valid_format(tmp_path) -> None:
    problem = Problem("p1", "train", (2, 3, 7), 20)
    exact_writer = JsonlWriter(tmp_path / "exact.jsonl")
    exact = make_reward_function("exact_plus_validity", exact_writer, "phase-01")
    rewards = exact(
        completions=[
            [{"role": "assistant", "content": "(7 + 3) * 2"}],
            [{"role": "assistant", "content": "7 + 3 + 2"}],
        ],
        problem_id=[problem.id, problem.id],
        nums=[list(problem.nums), list(problem.nums)],
        target=[problem.target, problem.target],
        arity=[problem.arity, problem.arity],
        trainer_state=type("State", (), {"global_step": 4})(),
    )
    exact_writer.close()
    assert rewards == [1.0, 0.2]

    valid_writer = JsonlWriter(tmp_path / "validity.jsonl")
    validity = make_reward_function("expression_validity", valid_writer, "phase-01")
    validity_rewards = validity(
        completions=[
            [{"role": "assistant", "content": "(7 + 3) * 2"}],
            [{"role": "assistant", "content": "7 + 3 + 2"}],
            [{"role": "assistant", "content": "try this: 7+3+2"}],
        ],
        problem_id=[problem.id] * 3,
        nums=[list(problem.nums)] * 3,
        target=[problem.target] * 3,
        arity=[problem.arity] * 3,
    )
    valid_writer.close()
    assert validity_rewards == [1.0, 1.0, 0.0]

    rows = [json.loads(line) for line in (tmp_path / "exact.jsonl").read_text().splitlines()]
    assert rows[0]["correct"] is True
    assert rows[1]["valid"] is True
    assert rows[1]["correct"] is False
    assert rows[1]["reward"] == 0.2
