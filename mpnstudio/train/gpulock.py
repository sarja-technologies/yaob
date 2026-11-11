"""Cooperation with the Spark's existing GPU etiquette.

Other work on this machine takes an advisory lock on one file: many LLM jobs may hold it shared,
a heavy media run takes it exclusively. A long training job that simply held the shared lock
would block media runs for hours, so a job here holds the lock only while it is computing and
gives it up as soon as an exclusive waiter appears, then takes it again and continues.

When no lock file exists (another machine, tests) the lock is a no-op.
"""
from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

DEFAULT_LOCK = Path.home() / "spark-reset" / "gpu.lock"


class GpuLock:
    def __init__(self, path: Path | None = None):
        self.path = Path(os.environ.get("MPN_GPU_LOCK", path or DEFAULT_LOCK))
        self.fd: int | None = None
        self.key: str | None = None
        self.waited_s = 0.0
        self.yields = 0

    @property
    def enabled(self) -> bool:
        return self.path.exists()

    def acquire(self) -> float:
        """Take the shared lock, waiting while an exclusive holder runs. Returns seconds waited."""
        if not self.enabled:
            return 0.0
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
            st = os.fstat(self.fd)
            self.key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
        t0 = time.monotonic()
        fcntl.flock(self.fd, fcntl.LOCK_SH)
        w = time.monotonic() - t0
        self.waited_s += w
        return w

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def exclusive_waiter(self) -> bool:
        """Is someone blocked waiting for the exclusive lock on this file?"""
        if not self.enabled or self.key is None:
            return False
        try:
            for line in Path("/proc/locks").read_text().splitlines():
                if "->" in line and "FLOCK" in line and "WRITE" in line and self.key in line:
                    return True
        except OSError:
            return False
        return False

    def yield_if_wanted(self) -> float:
        """If an exclusive waiter is queued, release, let it run, and re-acquire. Returns seconds spent yielding."""
        if not self.exclusive_waiter():
            return 0.0
        self.yields += 1
        self.release()
        time.sleep(2.0)                 # give the waiter time to take the lock
        return self.acquire() + 2.0

    def close(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.close()
