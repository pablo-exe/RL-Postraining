"""Persistent, interruption-tolerant GPU wall-clock budget accounting."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class BudgetExceededError(RuntimeError):
    """Raised before starting GPU work when the project has exhausted its cap."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class GpuBudget(AbstractContextManager["GpuBudget"]):
    """Account actual GPU-process wall time in one SQLite-backed project ledger."""

    def __init__(
        self,
        runs_dir: Path,
        command: str,
        max_hours: float = 72.0,
        heartbeat_seconds: float = 20.0,
    ) -> None:
        self.ledger = runs_dir / "accounting" / "gpu_budget.sqlite3"
        self.command = command
        self.max_seconds = max_hours * 3600
        self.heartbeat_seconds = heartbeat_seconds
        self.session_id = uuid4().hex
        self._monotonic_start = 0.0
        self._seconds_committed = 0.0
        self._closing = threading.Event()
        self._flush_lock = threading.Lock()
        self._heartbeat_thread: threading.Thread | None = None
        self._entered = False

    def _connection(self) -> sqlite3.Connection:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.ledger, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                command TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                last_heartbeat_utc TEXT NOT NULL,
                seconds REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL
            )
            """
        )
        connection.commit()
        return connection

    def total_seconds(self) -> float:
        if not self.ledger.exists():
            return 0.0
        with self._connection() as connection:
            if self._entered:
                total = connection.execute(
                    "SELECT COALESCE(SUM(seconds), 0) FROM sessions WHERE id<>?",
                    (self.session_id,),
                ).fetchone()[0]
            else:
                total = connection.execute("SELECT COALESCE(SUM(seconds), 0) FROM sessions").fetchone()[0]
        return float(total)

    def __enter__(self) -> GpuBudget:
        self._monotonic_start = time.monotonic()
        now = utc_now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT id, command, last_heartbeat_utc, seconds FROM sessions "
                "WHERE status='running' LIMIT 1"
            ).fetchone()
            if existing:
                stale_period_seconds = max(self.heartbeat_seconds * 2, 60)
                last_heartbeat = datetime.fromisoformat(existing[2])
                heartbeat_age = max(0.0, (datetime.now(UTC) - last_heartbeat).total_seconds())
                if heartbeat_age < stale_period_seconds:
                    raise RuntimeError(
                        f"GPU budget session {existing[1]!r} is already open. "
                        "Wait for the process to finish before launching another experiment."
                    )
                recovery_charge = min(heartbeat_age, stale_period_seconds)
                connection.execute(
                    "UPDATE sessions SET seconds=seconds+?, status='abandoned', last_heartbeat_utc=? WHERE id=?",
                    (recovery_charge, utc_now(), existing[0]),
                )
            total = float(connection.execute("SELECT COALESCE(SUM(seconds), 0) FROM sessions").fetchone()[0])
            if total >= self.max_seconds:
                raise BudgetExceededError(f"The project has already used {total / 3600:.2f} of its GPU-hour budget.")
            connection.execute(
                "INSERT INTO sessions (id, command, started_at_utc, last_heartbeat_utc, seconds, status) "
                "VALUES (?, ?, ?, ?, 0, 'running')",
                (self.session_id, self.command, now, now),
            )
        self._entered = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"gpu-budget-{self.session_id[:8]}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        return self

    @property
    def elapsed_seconds(self) -> float:
        if not self._entered:
            return 0.0
        return self._seconds_committed + time.monotonic() - self._monotonic_start

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.total_seconds() - self.elapsed_seconds)

    @property
    def should_stop(self) -> bool:
        # Keep enough time to save the last adapter, write manifests, and run
        # one validation callback if the budget boundary falls mid-phase.
        return self.remaining_seconds <= 900.0

    def ensure_available(self, minimum_seconds: float = 1.0) -> None:
        remaining = self.remaining_seconds
        if remaining < minimum_seconds:
            raise BudgetExceededError(
                f"Only {remaining / 3600:.3f} GPU hours remain; "
                f"{minimum_seconds / 3600:.3f} hours were reserved for {self.command!r}."
            )

    def snapshot(self) -> dict[str, object]:
        total = self.total_seconds() + self.elapsed_seconds if self._entered else self.total_seconds()
        return {
            "max_gpu_hours": self.max_seconds / 3600,
            "used_gpu_hours": total / 3600,
            "remaining_gpu_hours": max(0.0, self.max_seconds - total) / 3600,
            "current_command": self.command,
            "current_session_elapsed_hours": self.elapsed_seconds / 3600,
            "ledger": self.ledger.relative_to(self.ledger.parents[1]).as_posix(),
        }

    def write_summary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")
        temp_path.replace(path)

    def _heartbeat_loop(self) -> None:
        while not self._closing.wait(self.heartbeat_seconds):
            self.flush(status="running")

    def flush(self, status: str = "running") -> None:
        if not self._entered:
            return
        with self._flush_lock:
            if not self._entered:
                return
            elapsed = max(0.0, time.monotonic() - self._monotonic_start)
            self._seconds_committed += elapsed
            self._monotonic_start = time.monotonic()
            with self._connection() as connection:
                connection.execute(
                    "UPDATE sessions SET seconds=seconds+?, last_heartbeat_utc=?, status=? WHERE id=?",
                    (elapsed, utc_now(), status, self.session_id),
                )

    def close(self, status: str = "completed") -> None:
        if not self._entered:
            return
        self._closing.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self.heartbeat_seconds + 1)
        self.flush(status=status)
        with self._flush_lock:
            self._entered = False

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> bool:
        self.close("failed" if exception_type is not None else "completed")
        return False

