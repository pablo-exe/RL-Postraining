"""The frozen, shared model prompt and rendered dataset helpers."""

from __future__ import annotations

from typing import Any

from rl_countdown.config import ProjectConfig
from rl_countdown.core import Problem


def chat_messages(problem: Problem, config: ProjectConfig) -> list[dict[str, str]]:
    system_prompt = str(config.values["prompt"]["system"]).strip()
    numbers = ", ".join(str(number) for number in problem.nums)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Numbers: [{numbers}]\nTarget: {problem.target}"},
    ]


def render_completion(
    tokenizer: Any,
    problem: Problem,
    config: ProjectConfig,
    witness: str | None = None,
) -> dict[str, Any]:
    prompt_ids = tokenizer.apply_chat_template(
        chat_messages(problem, config),
        tokenize=True,
        add_generation_prompt=True,
    )
    response_ids: list[int] = []
    if witness is not None:
        response_ids = tokenizer(
            witness + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
    input_ids = list(prompt_ids) + list(response_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": ([-100] * len(prompt_ids)) + list(response_ids),
    }

