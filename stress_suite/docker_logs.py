"""Save `docker logs` for vLLM replicas (post-mortem without keeping the VM)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from stress_suite.docker_mgr import InferenceFleet

log = logging.getLogger(__name__)


def capture_replica_docker_log(
    container: Any,
    out_path: Path,
    *,
    tail: int = 8000,
    max_bytes: int = 524_288,
) -> None:
    """Fetch stdout+stderr from a Docker container (blocking). Truncates to max_bytes from the end."""
    try:
        raw: bytes = container.logs(stdout=True, stderr=True, tail=tail, timestamps=True)
    except Exception as e:  # noqa: BLE001
        raw = f"<docker logs() failed: {e!r}>\n".encode()
    if len(raw) > max_bytes:
        note = b"[... head truncated by stress_suite docker_log_capture.max_bytes ...]\n\n"
        raw = note + raw[-max_bytes:]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)


def _docker_log_capture_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("docker_log_capture") or {}


async def capture_not_ready_after_cold_start(
    fleet: InferenceFleet,
    cold_rows: list[dict[str, Any]],
    log_run_dir: Path,
    cfg: dict[str, Any],
) -> None:
    cap = _docker_log_capture_cfg(cfg)
    if cap.get("on_ready_fail", True) is False:
        return
    bad = [int(r["container_id"]) for r in cold_rows if not r.get("ready_ok")]
    if not bad:
        return
    dest = log_run_dir / "docker_logs"
    tail = int(cap.get("tail_lines", 8000))
    max_b = int(cap.get("max_bytes", 524_288))
    log.info(
        "Capturing docker logs for %s not-ready replica(s) -> %s/ (tail=%s lines)",
        len(bad),
        dest.name,
        tail,
    )

    async def one(cid: int) -> None:
        if cid < 0 or cid >= len(fleet.containers):
            return
        path = dest / f"replica_{cid}_not_ready.log"
        await asyncio.to_thread(
            capture_replica_docker_log,
            fleet.containers[cid],
            path,
            tail=tail,
            max_bytes=max_b,
        )

    await asyncio.gather(*(one(cid) for cid in bad))


def capture_all_at_end_before_stop(
    fleet: InferenceFleet,
    log_run_dir: Path,
    cfg: dict[str, Any],
) -> None:
    """Blocking: call from `finally` before `fleet.stop_all()`."""
    cap = _docker_log_capture_cfg(cfg)
    if not cap.get("capture_all_at_end", False):
        return
    dest = log_run_dir / "docker_logs"
    tail = int(cap.get("tail_lines_end", cap.get("tail_lines", 2000)))
    max_b = int(cap.get("max_bytes_end", cap.get("max_bytes", 262_144)))
    log.info(
        "capture_all_at_end: docker logs for %s replica(s) -> %s/ (tail=%s)",
        len(fleet.containers),
        dest.name,
        tail,
    )
    for i in range(len(fleet.containers)):
        path = dest / f"replica_{i}_end_of_run.log"
        try:
            capture_replica_docker_log(
                fleet.containers[i], path, tail=tail, max_bytes=max_b
            )
        except Exception as e:  # noqa: BLE001
            log.warning("capture_all_at_end replica %s: %s", i, e)
