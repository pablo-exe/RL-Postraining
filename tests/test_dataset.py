from __future__ import annotations

import json

import pytest

from rl_countdown.dataset import (
    DatasetBundle,
    generate_dataset,
    get_split,
    load_dataset,
    save_dataset,
    validate_dataset,
)


@pytest.fixture(scope="module")
def frozen_bundle() -> DatasetBundle:
    return generate_dataset(seed=20261013)


def test_full_dataset_has_frozen_counts_and_exact_witnesses(frozen_bundle: DatasetBundle) -> None:
    manifest = frozen_bundle.manifest.to_dict()
    assert manifest["split_counts"] == {
        "train": 4096,
        "validation": 512,
        "test_id": 1024,
        "test_ood": 512,
    }
    assert manifest["dataset_sha256"] == "5b70aa24f0ce17b5da72c8031e39388126608e14cb82cd563a47983e52b57363"
    validate_dataset(frozen_bundle)
    assert len({problem.id for problem in frozen_bundle.problems}) == 6144
    assert len({witness.id for witness in frozen_bundle.witnesses}) == 6144
    assert all(problem.arity in (3, 4) for problem in frozen_bundle.problems)


def test_save_load_roundtrip_preserves_hashes(frozen_bundle: DatasetBundle, tmp_path) -> None:
    save_dataset(frozen_bundle, tmp_path)
    loaded = load_dataset(tmp_path)
    assert loaded.manifest == frozen_bundle.manifest
    assert loaded.problems == frozen_bundle.problems
    assert loaded.witnesses == frozen_bundle.witnesses
    assert save_dataset(frozen_bundle, tmp_path) == tmp_path
    (tmp_path / "witnesses.jsonl").unlink()
    with pytest.raises(FileExistsError):
        save_dataset(frozen_bundle, tmp_path)


def test_load_rejects_tampered_problem_records(frozen_bundle: DatasetBundle, tmp_path) -> None:
    save_dataset(frozen_bundle, tmp_path)
    problems_path = tmp_path / "problems.jsonl"
    rows = problems_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["target"] += 1
    rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    problems_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(tmp_path)


def test_get_split_rejects_unknown_names(frozen_bundle: DatasetBundle) -> None:
    assert len(get_split(frozen_bundle, "test_ood")) == 512
    with pytest.raises(ValueError):
        get_split(frozen_bundle, "validation-test")

