"""Mirror stdout/stderr to a UTF-8 text file (full console transcript for a run)."""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


class _TeeTextIO:
    __slots__ = ("_orig", "_file", "_lock")

    def __init__(self, original: TextIO, file_obj: TextIO, lock: threading.Lock) -> None:
        self._orig = original
        self._file = file_obj
        self._lock = lock

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            self._orig.write(s)
            self._orig.flush()
            self._file.write(s)
            self._file.flush()
        return len(s)

    def flush(self) -> None:
        with self._lock:
            self._orig.flush()
            self._file.flush()

    def isatty(self) -> bool:
        return self._orig.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._orig, "encoding", None) or "utf-8"

    def fileno(self) -> int:
        return self._orig.fileno()


@dataclass
class ConsoleCapture:
    """Tee stdout/stderr to ``log_run_dir / \"console.txt\"``."""

    path: Path
    _orig_out: TextIO
    _orig_err: TextIO
    _file: TextIO
    _tee_out: TextIO
    _tee_err: TextIO

    @classmethod
    def start(cls, log_run_dir: Path) -> ConsoleCapture:
        log_run_dir.mkdir(parents=True, exist_ok=True)
        path = log_run_dir / "console.txt"
        f = open(path, "w", encoding="utf-8", buffering=1)
        lock = threading.Lock()
        orig_out, orig_err = sys.stdout, sys.stderr
        tee_out = _TeeTextIO(orig_out, f, lock)
        tee_err = _TeeTextIO(orig_err, f, lock)
        sys.stdout = tee_out  # type: ignore[assignment]
        sys.stderr = tee_err  # type: ignore[assignment]
        for h in logging.root.handlers:
            if not isinstance(h, logging.StreamHandler):
                continue
            if h.stream is orig_err:
                h.setStream(tee_err)
            elif h.stream is orig_out:
                h.setStream(tee_out)
        return cls(
            path=path.resolve(),
            _orig_out=orig_out,
            _orig_err=orig_err,
            _file=f,
            _tee_out=tee_out,
            _tee_err=tee_err,
        )

    def stop(self) -> None:
        for h in logging.root.handlers:
            if not isinstance(h, logging.StreamHandler):
                continue
            if h.stream is self._tee_err:
                h.setStream(self._orig_err)
            elif h.stream is self._tee_out:
                h.setStream(self._orig_out)
        sys.stdout = self._orig_out
        sys.stderr = self._orig_err
        try:
            self._file.flush()
        except Exception:  # noqa: BLE001
            pass
        self._file.close()
