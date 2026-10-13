"""Exact Countdown expression parsing and verification.

Generated model output is treated as untrusted text.  This module never
executes it: the expression is parsed into Python's AST, checked against a
small allowlist, and evaluated recursively with :class:`fractions.Fraction`.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

SPLITS = frozenset({"train", "validation", "test_id", "test_ood"})
ERROR_CODES = frozenset(
    {
        "empty",
        "too_long",
        "invalid_characters",
        "invalid_syntax",
        "unsupported_node",
        "wrong_numbers",
        "division_by_zero",
    }
)
_ALLOWED_CHARACTERS = frozenset("0123456789 +-*/()")
_ALLOWED_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


@dataclass(frozen=True, slots=True)
class Problem:
    """One immutable Countdown prompt, including its dataset partition."""

    id: str
    split: str
    nums: tuple[int, ...]
    target: int

    def __post_init__(self) -> None:
        try:
            normalized_nums = tuple(self.nums)
        except TypeError as exc:
            raise TypeError("nums must be an iterable of positive integers") from exc
        object.__setattr__(self, "nums", normalized_nums)

        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.split, str) or self.split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}")
        if len(normalized_nums) not in (3, 4):
            raise ValueError("Countdown problems must contain three or four numbers")
        if any(type(number) is not int or number <= 0 for number in normalized_nums):
            raise ValueError("nums must contain positive integers")
        if type(self.target) is not int:
            raise TypeError("target must be an integer")

    @property
    def arity(self) -> int:
        return len(self.nums)


@dataclass(frozen=True, slots=True)
class Verification:
    """Result of checking one expression against a Countdown problem."""

    valid: bool
    correct: bool
    error_code: str | None
    value_numerator: int | None
    value_denominator: int | None

    def __post_init__(self) -> None:
        if type(self.valid) is not bool or type(self.correct) is not bool:
            raise TypeError("valid and correct must be booleans")
        if self.error_code is not None and self.error_code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {self.error_code}")
        if self.correct and not self.valid:
            raise ValueError("an invalid expression cannot be correct")
        if self.valid:
            if self.error_code is not None:
                raise ValueError("a valid expression cannot have an error code")
            if type(self.value_numerator) is not int or type(self.value_denominator) is not int:
                raise ValueError("a valid expression must include its exact rational value")
            if self.value_denominator <= 0:
                raise ValueError("value_denominator must be positive")
        else:
            if self.correct:
                raise ValueError("an invalid expression cannot be correct")
            if self.error_code is None:
                raise ValueError("an invalid expression must have an error code")


def _invalid(error_code: str) -> Verification:
    return Verification(False, False, error_code, None, None)


def _collect_integer_leaves(node: ast.AST) -> list[int]:
    """Validate the expression AST and return its integer leaves."""
    if isinstance(node, ast.Expression):
        return _collect_integer_leaves(node.body)
    if isinstance(node, ast.Constant):
        if type(node.value) is not int:
            raise _UnsupportedExpression
        return [node.value]
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_OPERATORS):
            raise _UnsupportedExpression
        return _collect_integer_leaves(node.left) + _collect_integer_leaves(node.right)
    raise _UnsupportedExpression


class _UnsupportedExpression(Exception):
    pass


def _evaluate_fraction(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate_fraction(node.body)
    if isinstance(node, ast.Constant):
        # The AST was already validated by _collect_integer_leaves.
        return Fraction(node.value)
    if isinstance(node, ast.BinOp):
        left = _evaluate_fraction(node.left)
        right = _evaluate_fraction(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise _UnsupportedExpression


def verify_expression(
    text: str,
    nums: Iterable[int],
    target: int,
) -> Verification:
    """Validate and exactly evaluate a model answer.

    Accepted answers contain one arithmetic expression using every supplied
    positive integer exactly once, with only ``+``, ``-``, ``*``, ``/`` and
    parentheses.  Rational and negative intermediate values are supported.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        supplied_numbers = tuple(nums)
    except TypeError as exc:
        raise TypeError("nums must be an iterable of positive integers") from exc
    if any(type(number) is not int or number <= 0 for number in supplied_numbers):
        raise ValueError("nums must contain positive integers")
    if type(target) is not int:
        raise TypeError("target must be an integer")

    expression = text.strip()
    if not expression:
        return _invalid("empty")
    if len(expression) > 256:
        return _invalid("too_long")
    if any(character not in _ALLOWED_CHARACTERS for character in expression):
        return _invalid("invalid_characters")

    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return _invalid("invalid_syntax")

    try:
        leaves = _collect_integer_leaves(tree)
    except _UnsupportedExpression:
        return _invalid("unsupported_node")

    if Counter(leaves) != Counter(supplied_numbers):
        return _invalid("wrong_numbers")

    try:
        value = _evaluate_fraction(tree)
    except ZeroDivisionError:
        return _invalid("division_by_zero")
    except _UnsupportedExpression:
        # Defensive: the tree has already passed the same allowlist above.
        return _invalid("unsupported_node")

    return Verification(
        valid=True,
        correct=value == target,
        error_code=None,
        value_numerator=value.numerator,
        value_denominator=value.denominator,
    )

