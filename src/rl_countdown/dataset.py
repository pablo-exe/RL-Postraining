"""Deterministic Countdown data, exact witnesses, hashes, and persistence."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .core import Problem, verify_expression

DATASET_SEED = 20261013
DATASET_SCHEMA_VERSION = 1
SOLVER_VERSION = "fraction-subset-dp-v1"
SPLIT_SIZES: dict[str, int] = {
    "train": 4096,
    "validation": 512,
    "test_id": 1024,
    "test_ood": 512,
}
SPLIT_ORDER = tuple(SPLIT_SIZES)
ARITIES = (3, 4)


@dataclass(frozen=True, slots=True)
class ReferenceSolution:
    """A separate exact witness for one problem; never needed by GRPO."""

    id: str
    reference_expression: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.reference_expression, str) or not self.reference_expression:
            raise ValueError("reference_expression must be a non-empty string")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable, serializable record of frozen dataset content hashes."""

    seed: int
    split_counts: tuple[tuple[str, int], ...]
    problem_sha256: tuple[tuple[str, str], ...]
    witness_sha256: tuple[tuple[str, str], ...]
    dataset_sha256: str
    solver_version: str = SOLVER_VERSION
    schema_version: int = DATASET_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "solver_version": self.solver_version,
            "seed": self.seed,
            "split_counts": dict(self.split_counts),
            "problem_sha256": dict(self.problem_sha256),
            "witness_sha256": dict(self.witness_sha256),
            "dataset_sha256": self.dataset_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetManifest:
        return cls(
            seed=int(value["seed"]),
            split_counts=tuple((key, int(value["split_counts"][key])) for key in SPLIT_ORDER),
            problem_sha256=tuple((key, str(value["problem_sha256"][key])) for key in SPLIT_ORDER),
            witness_sha256=tuple((key, str(value["witness_sha256"][key])) for key in SPLIT_ORDER),
            dataset_sha256=str(value["dataset_sha256"]),
            solver_version=str(value["solver_version"]),
            schema_version=int(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Problems, their separate witnesses, and the content manifest."""

    problems: tuple[Problem, ...]
    witnesses: tuple[ReferenceSolution, ...]
    manifest: DatasetManifest

    def __post_init__(self) -> None:
        object.__setattr__(self, "problems", tuple(self.problems))
        object.__setattr__(self, "witnesses", tuple(self.witnesses))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _problem_record(problem: Problem) -> dict[str, Any]:
    return {
        "id": problem.id,
        "split": problem.split,
        "nums": list(problem.nums),
        "target": problem.target,
        "arity": problem.arity,
    }


def _witness_record(witness: ReferenceSolution) -> dict[str, str]:
    return {"id": witness.id, "reference_expression": witness.reference_expression}


def _serialize_problems(problems: Iterable[Problem]) -> bytes:
    return "".join(_json_line(_problem_record(problem)) for problem in problems).encode("utf-8")


def _serialize_witnesses(witnesses: Iterable[ReferenceSolution]) -> bytes:
    return "".join(_json_line(_witness_record(witness)) for witness in witnesses).encode("utf-8")


def _make_manifest(
    seed: int,
    problems: tuple[Problem, ...],
    witnesses: tuple[ReferenceSolution, ...],
) -> DatasetManifest:
    witness_by_id = {witness.id: witness for witness in witnesses}
    problem_hashes: list[tuple[str, str]] = []
    witness_hashes: list[tuple[str, str]] = []
    counts: list[tuple[str, int]] = []
    for split in SPLIT_ORDER:
        split_problems = tuple(problem for problem in problems if problem.split == split)
        split_witnesses = tuple(witness_by_id[problem.id] for problem in split_problems)
        counts.append((split, len(split_problems)))
        problem_hashes.append((split, _sha256(_serialize_problems(split_problems))))
        witness_hashes.append((split, _sha256(_serialize_witnesses(split_witnesses))))

    problem_bytes = _serialize_problems(problems)
    witness_bytes = _serialize_witnesses(witnesses)
    return DatasetManifest(
        seed=seed,
        split_counts=tuple(counts),
        problem_sha256=tuple(problem_hashes),
        witness_sha256=tuple(witness_hashes),
        dataset_sha256=_sha256(problem_bytes + b"\0" + witness_bytes),
    )


def _keep_best(targets: dict[Fraction, str], value: Fraction, expression: str) -> None:
    current = targets.get(value)
    if current is None or (len(expression), expression) < (len(current), current):
        targets[value] = expression


def solve_reachable(nums: Iterable[int]) -> dict[int, str]:
    """Return canonical witnesses for integer targets reachable from ``nums``.

    The dynamic program considers every ordered split of every subset, so it
    covers all binary expression trees and both operand orders for subtraction
    and division.  Fractions make all intermediate calculations exact.
    """
    numbers = tuple(nums)
    if len(numbers) not in ARITIES:
        raise ValueError("Countdown supports exactly three or four numbers")
    if any(type(number) is not int or number <= 0 for number in numbers):
        raise ValueError("nums must contain positive integers")

    subset_count = 1 << len(numbers)
    states: list[dict[Fraction, str]] = [{} for _ in range(subset_count)]
    for index, number in enumerate(numbers):
        states[1 << index][Fraction(number)] = str(number)

    for mask in range(1, subset_count):
        if mask & (mask - 1) == 0:
            continue
        submask = (mask - 1) & mask
        while submask:
            other = mask ^ submask
            if other:
                left_states = states[submask]
                right_states = states[other]
                for left_value, left_expression in left_states.items():
                    for right_value, right_expression in right_states.items():
                        candidates = (
                            (left_value + right_value, f"({left_expression}+{right_expression})"),
                            (left_value - right_value, f"({left_expression}-{right_expression})"),
                            (left_value * right_value, f"({left_expression}*{right_expression})"),
                        )
                        for value, expression in candidates:
                            _keep_best(states[mask], value, expression)
                        if right_value != 0:
                            _keep_best(
                                states[mask],
                                left_value / right_value,
                                f"({left_expression}/{right_expression})",
                            )
            submask = (submask - 1) & mask

    reachable: dict[int, str] = {}
    for value, expression in states[-1].items():
        if value.denominator == 1 and 10 <= value.numerator <= 100:
            reachable[value.numerator] = expression
    return reachable


def solve_countdown(nums: Iterable[int], target: int) -> str | None:
    """Return the canonical witness for ``target`` or ``None`` if unreachable."""
    if type(target) is not int:
        raise TypeError("target must be an integer")
    return solve_reachable(nums).get(target)


def _accepted_problems(
    *,
    low: int,
    high: int,
    arity: int,
    count: int,
    seed: int,
) -> list[tuple[tuple[int, ...], int, str]]:
    candidates = list(itertools.combinations_with_replacement(range(low, high + 1), arity))
    random.Random(seed + arity).shuffle(candidates)
    target_rng = random.Random(seed + arity + low * 1009)
    accepted: list[tuple[tuple[int, ...], int, str]] = []
    for numbers in candidates:
        reachable = solve_reachable(numbers)
        targets = sorted(reachable)
        if not targets:
            continue
        target = target_rng.choice(targets)
        accepted.append((numbers, target, reachable[target]))
        if len(accepted) == count:
            return accepted
    raise RuntimeError(
        f"Only generated {len(accepted)} solvable {arity}-number problems for "
        f"range {low}..{high}; {count} required"
    )


def _per_arity_split_counts() -> dict[str, dict[int, int]]:
    return {
        "train": {3: SPLIT_SIZES["train"] // 2, 4: SPLIT_SIZES["train"] // 2},
        "validation": {
            3: SPLIT_SIZES["validation"] // 2,
            4: SPLIT_SIZES["validation"] // 2,
        },
        "test_id": {3: SPLIT_SIZES["test_id"] // 2, 4: SPLIT_SIZES["test_id"] // 2},
        "test_ood": {3: SPLIT_SIZES["test_ood"] // 2, 4: SPLIT_SIZES["test_ood"] // 2},
    }


def _id_for(split: str, arity: int, index: int) -> str:
    return f"countdown-{split}-{arity}-{index:05d}"


def _display_order(numbers: tuple[int, ...], problem_id: str) -> tuple[int, ...]:
    """Derive a repeatable prompt order from the public problem identifier."""
    seed = int.from_bytes(hashlib.sha256(problem_id.encode("utf-8")).digest(), "big")
    shuffled = list(numbers)
    random.Random(seed).shuffle(shuffled)
    return tuple(shuffled)


def generate_dataset(seed: int = DATASET_SEED) -> DatasetBundle:
    """Generate the frozen train, validation, ID-test, and OOD-test splits."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    counts = _per_arity_split_counts()
    rows_by_split: dict[str, dict[int, list[tuple[Problem, ReferenceSolution]]]] = {
        split: {arity: [] for arity in ARITIES} for split in SPLIT_ORDER
    }

    id_total_by_arity = {
        arity: sum(counts[split][arity] for split in ("train", "validation", "test_id"))
        for arity in ARITIES
    }
    for arity in ARITIES:
        accepted = _accepted_problems(
            low=1,
            high=30,
            arity=arity,
            count=id_total_by_arity[arity],
            seed=seed,
        )
        offset = 0
        for split in ("train", "validation", "test_id"):
            split_count = counts[split][arity]
            for index, (numbers, target, witness_expression) in enumerate(
                accepted[offset : offset + split_count], start=1
            ):
                problem_id = _id_for(split, arity, index)
                problem = Problem(problem_id, split, _display_order(numbers, problem_id), target)
                witness = ReferenceSolution(problem_id, witness_expression)
                rows_by_split[split][arity].append((problem, witness))
            offset += split_count

        ood = _accepted_problems(
            low=31,
            high=60,
            arity=arity,
            count=counts["test_ood"][arity],
            seed=seed,
        )
        for index, (numbers, target, witness_expression) in enumerate(ood, start=1):
            problem_id = _id_for("test_ood", arity, index)
            problem = Problem(problem_id, "test_ood", _display_order(numbers, problem_id), target)
            witness = ReferenceSolution(problem_id, witness_expression)
            rows_by_split["test_ood"][arity].append((problem, witness))

    problems: list[Problem] = []
    witnesses: list[ReferenceSolution] = []
    for split_index, split in enumerate(SPLIT_ORDER):
        interleaved = [
            pair
            for pair in itertools.chain.from_iterable(
                zip(rows_by_split[split][3], rows_by_split[split][4], strict=True)
            )
        ]
        random.Random(seed + 20_261_013 + split_index).shuffle(interleaved)
        for problem, witness in interleaved:
            problems.append(problem)
            witnesses.append(witness)

    problems_tuple = tuple(problems)
    witnesses_tuple = tuple(witnesses)
    manifest = _make_manifest(seed, problems_tuple, witnesses_tuple)
    bundle = DatasetBundle(problems_tuple, witnesses_tuple, manifest)
    validate_dataset(bundle)
    return bundle


def get_split(bundle: DatasetBundle, split: str) -> tuple[Problem, ...]:
    """Return an immutable, order-preserving view of one frozen partition."""
    if split not in SPLIT_ORDER:
        raise ValueError(f"split must be one of {SPLIT_ORDER}")
    return tuple(problem for problem in bundle.problems if problem.split == split)


def balanced_hash_subset(
    problems: tuple[Problem, ...],
    size: int,
    seed: int,
) -> tuple[Problem, ...]:
    """Select a deterministic, arity-balanced subset ranked by row-ID hash."""
    if size < 0 or size > len(problems):
        raise ValueError("subset size must be between zero and the available problem count")
    if size == 0:
        return ()
    by_arity: dict[int, list[Problem]] = {}
    for problem in problems:
        by_arity.setdefault(problem.arity, []).append(problem)
    arities = sorted(by_arity)
    if not arities:
        raise ValueError("cannot select a subset from an empty problem collection")
    quotas = {arity: size // len(arities) for arity in arities}
    for arity in arities[: size % len(arities)]:
        quotas[arity] += 1
    if any(quotas[arity] > len(by_arity[arity]) for arity in arities):
        raise ValueError("cannot satisfy a balanced subset with the available arity counts")

    selected: list[Problem] = []
    for arity in arities:
        ranked = sorted(
            by_arity[arity],
            key=lambda problem: hashlib.sha256(f"{seed}:{problem.id}".encode()).digest(),
        )
        selected.extend(ranked[: quotas[arity]])
    return tuple(sorted(selected, key=lambda problem: problem.id))


def prepare_data(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Generate and persist the frozen dataset from a small config mapping.

    Accepts either the project's full YAML mapping (with a ``data`` section)
    or that section itself. Protocol-defining values are checked against the
    frozen specification before data is generated. The returned dictionary is
    the same SHA256 manifest written to disk.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    data_config = config.get("data", config)
    if not isinstance(data_config, Mapping):
        raise TypeError("config['data'] must be a mapping")

    fixed_values: dict[str, Any] = {
        "numbers_id_min": 1,
        "numbers_id_max": 30,
        "numbers_ood_min": 31,
        "numbers_ood_max": 60,
        "target_min": 10,
        "target_max": 100,
        "arities": [3, 4],
        "train_size": SPLIT_SIZES["train"],
        "validation_size": SPLIT_SIZES["validation"],
        "test_id_size": SPLIT_SIZES["test_id"],
        "test_ood_size": SPLIT_SIZES["test_ood"],
        "generation_batch_size_per_arity": [2048, 2048],
    }
    for key, expected in fixed_values.items():
        if key in data_config and data_config[key] != expected:
            raise ValueError(f"data.{key} is frozen for this protocol: {expected}")
    if "split_sizes" in data_config and dict(data_config["split_sizes"]) != SPLIT_SIZES:
        raise ValueError(f"split_sizes are frozen for this protocol: {SPLIT_SIZES}")
    bundle = generate_dataset(data_config.get("seed", DATASET_SEED))
    save_dataset(bundle, output_dir)
    return bundle.manifest.to_dict()


def validate_dataset(
    bundle: DatasetBundle,
    expected_split_sizes: dict[str, int] | None = SPLIT_SIZES,
) -> None:
    """Check split sizes, isolation, witness correctness, and manifest hashes."""
    problems = bundle.problems
    witnesses = bundle.witnesses
    problems_by_id = {problem.id: problem for problem in problems}
    witnesses_by_id = {witness.id: witness for witness in witnesses}
    if len(problems_by_id) != len(problems):
        raise ValueError("problem IDs must be unique")
    if len(witnesses_by_id) != len(witnesses):
        raise ValueError("witness IDs must be unique")
    if set(problems_by_id) != set(witnesses_by_id):
        raise ValueError("every problem must have exactly one separate witness")

    counts = Counter(problem.split for problem in problems)
    if set(counts) - set(SPLIT_ORDER):
        raise ValueError("dataset contains an unknown split")
    if expected_split_sizes is not None and dict(counts) != expected_split_sizes:
        raise ValueError(f"unexpected split sizes: {dict(counts)}")

    for split in SPLIT_ORDER:
        split_problems = [problem for problem in problems if problem.split == split]
        arity_counts = Counter(problem.arity for problem in split_problems)
        if arity_counts != Counter({3: len(split_problems) // 2, 4: len(split_problems) // 2}):
            raise ValueError(f"split {split} must be balanced between arities three and four")

    id_problems = [problem for problem in problems if problem.split in {"train", "validation", "test_id"}]
    id_multisets = [tuple(sorted(problem.nums)) for problem in id_problems]
    if len(set(id_multisets)) != len(id_multisets):
        raise ValueError("ID multiset leakage found across train, validation, and test_id")
    if any(any(number < 1 or number > 30 for number in problem.nums) for problem in id_problems):
        raise ValueError("ID problems must use numbers in 1..30")
    ood_problems = [problem for problem in problems if problem.split == "test_ood"]
    if any(any(number < 31 or number > 60 for number in problem.nums) for problem in ood_problems):
        raise ValueError("OOD problems must use numbers in 31..60")

    for problem in problems:
        witness = witnesses_by_id[problem.id]
        result = verify_expression(witness.reference_expression, problem.nums, problem.target)
        if not result.valid or not result.correct:
            raise ValueError(f"invalid reference witness for {problem.id}: {result.error_code}")

    calculated_manifest = _make_manifest(bundle.manifest.seed, problems, witnesses)
    if calculated_manifest != bundle.manifest:
        raise ValueError("dataset content does not match its manifest hashes")


def save_dataset(bundle: DatasetBundle, directory: str | Path) -> Path:
    """Persist problems, separate witnesses, and SHA256 manifest as JSONL/JSON.

    Saving the same frozen dataset twice is idempotent. Existing different or
    partial dataset files are preserved and reported rather than overwritten.
    """
    validate_dataset(bundle)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    problem_path = destination / "problems.jsonl"
    witness_path = destination / "witnesses.jsonl"
    manifest_path = destination / "manifest.json"
    output = {
        problem_path: _serialize_problems(bundle.problems),
        witness_path: _serialize_witnesses(bundle.witnesses),
        manifest_path: (
            json.dumps(bundle.manifest.to_dict(), sort_keys=True, indent=2) + "\n"
        ).encode("utf-8"),
    }

    existing = [path.exists() for path in output]
    if any(existing):
        if all(existing):
            current = load_dataset(destination)
            if current.manifest == bundle.manifest:
                return destination
        raise FileExistsError(f"refusing to overwrite existing dataset files in {destination}")

    temporary_paths: list[Path] = []
    try:
        for final_path, contents in output.items():
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{final_path.name}.", suffix=".tmp", dir=destination, delete=False
            ) as handle:
                handle.write(contents)
                temporary_paths.append(Path(handle.name))
        for temporary_path, final_path in zip(temporary_paths, output, strict=True):
            temporary_path.replace(final_path)
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()
    return destination


def load_dataset(directory: str | Path) -> DatasetBundle:
    """Load and fully verify a persisted frozen dataset."""
    source = Path(directory)
    manifest_data = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest = DatasetManifest.from_dict(manifest_data)

    problems: list[Problem] = []
    with (source / "problems.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank problem row at line {line_number}")
            record = json.loads(line)
            if set(record) != {"id", "split", "nums", "target", "arity"}:
                raise ValueError(f"unexpected problem fields at line {line_number}")
            problem = Problem(record["id"], record["split"], tuple(record["nums"]), record["target"])
            if record["arity"] != problem.arity:
                raise ValueError(f"incorrect arity at line {line_number}")
            problems.append(problem)

    witnesses: list[ReferenceSolution] = []
    with (source / "witnesses.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank witness row at line {line_number}")
            record = json.loads(line)
            if set(record) != {"id", "reference_expression"}:
                raise ValueError(f"unexpected witness fields at line {line_number}")
            witnesses.append(ReferenceSolution(record["id"], record["reference_expression"]))

    bundle = DatasetBundle(tuple(problems), tuple(witnesses), manifest)
    validate_dataset(bundle)
    return bundle

