from __future__ import annotations

import sqlite3
import time

import pytest

from rl_countdown.budget import GpuBudget


def test_budget_accounts_wall_time_and_closes_a_session(tmp_path) -> None:
    tracker = GpuBudget(tmp_path, "unit-test", max_hours=72, heartbeat_seconds=0.05)
    with tracker:
        time.sleep(0.08)
        tracker.flush()
        snapshot = tracker.snapshot()
        assert snapshot["used_gpu_hours"] >= 0
        assert snapshot["remaining_gpu_hours"] <= 72
    assert tracker.snapshot()["used_gpu_hours"] >= 0.00002
    with sqlite3.connect(tracker.ledger) as connection:
        row = connection.execute("SELECT command, seconds, status FROM sessions").fetchone()
    assert row[0] == "unit-test"
    assert row[1] >= 0.05
    assert row[2] == "completed"


def test_only_one_live_gpu_budget_session_can_run(tmp_path) -> None:
    with GpuBudget(tmp_path, "outer", max_hours=72, heartbeat_seconds=0.05):
        with pytest.raises(RuntimeError, match="already open"):
            with GpuBudget(tmp_path, "inner", max_hours=72, heartbeat_seconds=0.05):
                pass
