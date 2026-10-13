from __future__ import annotations

import pytest

from rl_countdown.core import Problem, verify_expression


def test_problem_normalizes_numbers_and_reports_arity() -> None:
    problem = Problem("example", "validation", [2, 3, 7], 20)
    assert problem.nums == (2, 3, 7)
    assert problem.arity == 3


def test_exact_fraction_verifier_accepts_a_solution_and_valid_wrong_answer() -> None:
    correct = verify_expression("(7 + 3) * 2", [2, 3, 7], 20)
    wrong = verify_expression("7 + 3 + 2", [2, 3, 7], 20)
    assert correct.valid and correct.correct
    assert (correct.value_numerator, correct.value_denominator) == (20, 1)
    assert wrong.valid and not wrong.correct
    assert (wrong.value_numerator, wrong.value_denominator) == (12, 1)


@pytest.mark.parametrize(
    ("expression", "error_code"),
    [
        ("", "empty"),
        ("(7 + 3) * 2 because arithmetic", "invalid_characters"),
        ("7 + 3", "wrong_numbers"),
        ("(7 + 3) ** 2", "unsupported_node"),
        ("7 + 3 + True", "invalid_characters"),
    ],
)
def test_verifier_rejects_unsafe_or_invalid_responses(expression: str, error_code: str) -> None:
    result = verify_expression(expression, [2, 3, 7], 20)
    assert not result.valid
    assert not result.correct
    assert result.error_code == error_code


def test_verifier_uses_exact_rational_arithmetic() -> None:
    result = verify_expression("7 - 3 / 2", [2, 3, 7], 5)
    assert result.valid
    assert not result.correct
    assert (result.value_numerator, result.value_denominator) == (11, 2)


def test_verifier_rejects_division_by_zero_after_number_multiset_check() -> None:
    result = verify_expression("4 / (1 + 2 - 3)", [1, 2, 3, 4], 20)
    assert not result.valid
    assert result.error_code == "division_by_zero"

