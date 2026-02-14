"""Worker daemon lifecycle management: start, stop, status, health check."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time

from claude_mem_lite.config import Config


class WorkerLifecycle:
    """Manage worker daemon: start, stop, status, health check."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.pid_path = config.pid_path
        self.socket_path = config.socket_path

    def start(self, *, daemon: bool = True) -> int:
        """Start worker process. Returns PID.

        If a worker is already running, returns its PID (idempotent).
        If a stale PID file exists but the process is dead, cleans up
        and starts a new worker.
        """
        existing_pid = self.get_pid()
        if existing_pid is not None:
            if self._is_pid_alive(existing_pid):
                return existing_pid  # Already running -- idempotent
            # Stale PID file -- process is dead, clean up
            self._cleanup_stale_files()

        self.config.ensure_dirs()

        if daemon:
            proc = subprocess.Popen(
                [sys.executable, "-m", "claude_mem_lite.worker.server"],
                cwd=str(self.config.base_dir),
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._write_pid(proc.pid)
            self._wait_for_socket(timeout=10)
            return proc.pid

        # Foreground mode (for development/testing)
        self._write_pid(os.getpid())
        from claude_mem_lite.worker.server import run_worker

        run_worker(self.config)
        self._cleanup_stale_files()
        return os.getpid()

    def stop(self) -> bool:
        """Stop worker via SIGTERM. Returns True if stopped."""
        pid = self.get_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(50):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    break
            self._cleanup_stale_files()
        except OSError:
            self._cleanup_stale_files()
            return False
        else:
            return True

    def restart(self, *, daemon: bool = True) -> int:
        """Stop existing worker and start a new one."""
        self.stop()
        return self.start(daemon=daemon)

    def is_running(self) -> bool:
        """Check if worker process is alive."""
        pid = self.get_pid()
        if pid is None:
            return False
        if self._is_pid_alive(pid):
            return True
        self._cleanup_stale_files()
        return False

    def get_pid(self) -> int | None:
        """Read PID from file. Returns None if no PID file or invalid."""
        try:
            return int(self.pid_path.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None

    def _is_pid_alive(self, pid: int) -> bool:
        """Check if a process with given PID is alive."""
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    def _write_pid(self, pid: int) -> None:
        self.pid_path.write_text(str(pid))

    def _cleanup_stale_files(self) -> None:
        """Remove PID file and socket file if process is dead."""
        for path in (self.pid_path, self.socket_path):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    def _wait_for_socket(self, timeout: int = 10) -> None:
        """Wait for socket file to appear (worker is ready)."""
        for _ in range(timeout * 10):
            if self.socket_path.exists():
                return
            time.sleep(0.1)
        msg = f"Worker did not create socket within {timeout}s"
        raise TimeoutError(msg)


def cli_main() -> None:
    """CLI entry point for claude-mem-worker command."""
    config = Config()
    lifecycle = WorkerLifecycle(config)

    if len(sys.argv) < 2:
        print("Usage: claude-mem-worker {start|stop|status|restart}")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "start":
        daemon = "--foreground" not in sys.argv
        pid = lifecycle.start(daemon=daemon)
        print(f"Worker started (PID: {pid})")
    elif cmd == "stop":
        if lifecycle.stop():
            print("Worker stopped")
        else:
            print("Worker not running")
    elif cmd == "status":
        if lifecycle.is_running():
            print(f"Worker running (PID: {lifecycle.get_pid()})")
        else:
            print("Worker not running")
    elif cmd == "restart":
        pid = lifecycle.restart()
        print(f"Worker restarted (PID: {pid})")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
