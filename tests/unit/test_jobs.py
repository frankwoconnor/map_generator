"""Unit tests for map_core.jobs (background job runner)."""

import sys
import time

import pytest

from map_core.jobs import JobManager, TERMINAL_STATUSES


@pytest.fixture
def manager():
    return JobManager(venv_python=sys.executable)


def _wait_terminal(job, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.status in TERMINAL_STATUSES:
            return job.status
        time.sleep(0.05)
    raise AssertionError(f"job never reached terminal status (stuck at {job.status})")


def test_submit_runs_command_to_done(manager):
    job = manager.submit(
        [sys.executable, "-c", "import time; print('hello'); time.sleep(0.1); print('bye')"],
        cwd="/",
        title="test",
    )
    assert job.status == "queued" or job.status == "running"
    status = _wait_terminal(job)
    assert status == "done"
    assert job.returncode == 0
    assert "hello" in list(job.lines)


def test_submit_captures_lines_in_order(manager):
    job = manager.submit(
        [sys.executable, "-c", "print('a'); print('b'); print('c')"],
        cwd="/",
    )
    _wait_terminal(job)
    assert [l for l in job.lines if l] == ["a", "b", "c"]
    assert job.snapshot()["lines"][-3:] == ["a", "b", "c"]


def test_submit_error_status_on_bad_command():
    manager = JobManager(venv_python=sys.executable)
    job = manager.submit([sys.executable, "-c", "raise RuntimeError('boom')"], cwd="/")
    status = _wait_terminal(job)
    assert status == "error"
    assert job.error is not None


def test_cancel_terminates_running_job(manager):
    job = manager.submit(
        [sys.executable, "-c", "import time\nwhile True: time.sleep(0.1)"],
        cwd="/",
    )
    time.sleep(0.6)
    assert job.status == "running"
    assert manager.cancel(job.id) is True
    status = _wait_terminal(job, timeout=10.0)
    assert status == "cancelled"
    assert job.error == "Generation cancelled."


def test_cancel_unknown_job_returns_false(manager):
    assert manager.cancel("job_nope") is False


def test_snapshot_has_key_fields(manager):
    job = manager.submit([sys.executable, "-c", "print('x')"], cwd="/", title="t1")
    _wait_terminal(job)
    snap = job.snapshot()
    for key in ("id", "title", "status", "returncode", "error", "lines", "total_lines"):
        assert key in snap
    assert snap["title"] == "t1"
    assert snap["status"] == "done"


def test_manager_list_and_get(manager):
    job = manager.submit([sys.executable, "-c", "print('x')"], cwd="/", title="t1")
    _wait_terminal(job)
    assert manager.get(job.id) is job
    assert manager.get("job_nope") is None
    assert [s["id"] for s in manager.list()] == [job.id]


def test_generation_command_uses_venv_python():
    manager = JobManager(venv_python="/x/python3")
    assert manager.generation_command("map_20260101_000000") == [
        "/x/python3",
        "main.py",
        "--prefix",
        "map_20260101_000000",
    ]