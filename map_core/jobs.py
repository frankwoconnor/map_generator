"""Background job runner for map generation.

Grouping

    description: Run the generation CLI in a daemon thread (piped subprocess) so
    the web UI can stream progress over SSE instead of blocking the HTTP request.
    Jobs are in-memory and process-local; finished artifacts land on disk and are
    discoverable via ``map_core.runs``.

A job is a generic command runner (``JobManager.submit(cmd, ...)``), which keeps
it testable without a real generation. Generation itself is a thin wrapper that
builds the ``main.py`` command with the virtualenv interpreter.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from map_core import paths

TERMINAL_STATUSES = ("done", "error", "cancelled")

MAX_LOG_LINES = 1000


@dataclass
class Job:
    """One background run. Thread-safe via :attr:`_lock`."""

    id: str
    title: str
    status: str = "queued"  # queued | running | done | error | cancelled
    returncode: Optional[int] = None
    error: Optional[str] = None
    out_dir: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    lines: Deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    _total_lines: int = 0
    _events: "queue.Queue[Dict[str, Any]]" = field(default_factory=queue.Queue)
    _cancel: bool = False
    _proc: Optional[subprocess.Popen] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- data helpers ------------------------------------------------------

    def publish(self, text: str) -> None:
        """Append a log line and enqueue it for SSE consumers."""
        with self._lock:
            self._total_lines += 1
            seq = self._total_lines
            self.lines.append(text)
        self._events.put({"type": "line", "seq": seq, "text": text})

    def snapshot(self, max_lines: int = 200) -> Dict[str, Any]:
        """JSON-friendly summary (status + bounded log tail)."""
        with self._lock:
            lines = list(self.lines)
            total = self._total_lines
        if max_lines and len(lines) > max_lines:
            lines = lines[-max_lines:]
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "returncode": self.returncode,
            "error": self.error,
            "out_dir": self.out_dir,
            "total_lines": total,
            "lines": lines,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def next_event(self, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None


class JobManager:
    """In-memory registry that runs jobs in daemon threads."""

    def __init__(self, venv_python: Optional[str] = None) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self.venv_python = venv_python or self._default_python()

    @staticmethod
    def _default_python() -> str:
        venv = paths.REPO_ROOT / ".venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python3"
        )
        return str(venv) if venv.exists() else sys.executable

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            snapshots = [j.snapshot() for j in reversed(list(self._jobs.values()))]
        return snapshots[:limit] if limit else snapshots

    def submit(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        title: str = "",
        log_path: Optional[str] = None,
        out_dir: Optional[str] = None,
    ) -> Job:
        """Start a job running ``cmd`` and return it immediately."""
        job = Job(
            id=f"job_{uuid.uuid4().hex[:8]}",
            title=title or f" ".join(cmd[:2]),
            out_dir=out_dir,
        )
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(
            target=self._run,
            args=(job, cmd, cwd, log_path),
            name=f"job-{job.id}",
            daemon=True,
        )
        thread.start()
        return job

    def cancel(self, job_id: str) -> bool:
        """Request cancellation; SIGTERM the process group if still running."""
        job = self.get(job_id)
        if job is None or job.status in TERMINAL_STATUSES:
            return False
        job._cancel = True
        proc = job._proc
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
        return True

    def generation_command(self, prefix: str) -> List[str]:
        """The CLI command that runs one map generation into ``<prefix>_<ts>``."""
        return [self.venv_python, "main.py", "--prefix", prefix]

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _run(
        job: Job,
        cmd: List[str],
        cwd: Optional[str],
        log_path: Optional[str],
    ) -> None:
        job.status = "running"
        job.started_at = time.time()
        logf = None
        try:
            if log_path:
                logf = open(log_path, "a", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,  # own process group so cancel is simple
            )
            job._proc = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                job.publish(line.rstrip("\n") or "")
                if logf is not None:
                    logf.write(line)
                    logf.flush()
            proc.wait()
            job.returncode = proc.returncode
            if job._cancel or (proc.returncode is not None and proc.returncode < 0):
                job.status = "cancelled"
                job.error = "Generation cancelled."
            elif proc.returncode == 0:
                job.status = "done"
            else:
                job.status = "error"
                job.error = f"Map generation failed with exit code {proc.returncode}."
        except Exception as exc:  # noqa: BLE001 - surface anything to the client
            job.status = "error"
            job.error = str(exc)
        finally:
            if logf is not None:
                logf.close()
            job.finished_at = time.time()
            job._events.put(
                {
                    "type": "terminal",
                    "status": job.status,
                    "error": job.error,
                }
            )