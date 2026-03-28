"""Capture Docker container lifecycle events for stress replicas (restarts, die, oom)."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DockerEventWatcher:
    """Background thread: record container events whose names match this run."""

    def __init__(self, suite_name: str, run_id: str):
        self.suite_name = str(suite_name or "run")
        self.run_id = run_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.events: list[dict[str, Any]] = []
        self._written = False

    def _matches(self, name: str) -> bool:
        if not name:
            return False
        prefix = f"stress_{self.suite_name}_{self.run_id}_"
        return name.startswith(prefix)

    def _loop(self) -> None:
        import docker

        client = docker.from_env()
        try:
            for ev in client.events(decode=True):
                if self._stop.is_set():
                    break
                if ev.get("Type") != "container":
                    continue
                act = ev.get("Actor") or {}
                attrs = act.get("Attributes") or {}
                name = attrs.get("name") or ""
                if not self._matches(name):
                    continue
                self.events.append(
                    {
                        "time_unix": ev.get("time"),
                        "time_iso": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ev.get("time") or 0))
                        ),
                        "action": ev.get("Action") or ev.get("status"),
                        "name": name,
                        "container_id": act.get("ID", ""),
                        "exit_code": attrs.get("exitCode"),
                        "oom_killed": attrs.get("oomKilled"),
                    }
                )
        except Exception as e:  # noqa: BLE001
            log.debug("docker events stream ended: %s", e)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="docker-events", daemon=True)
        self._thread.start()

    def stop_and_write(self, path: Path) -> None:
        if self._written:
            return
        self._written = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(e, default=str) for e in self.events),
            encoding="utf-8",
        )
        log.info("Wrote %s docker events to %s", len(self.events), path)
