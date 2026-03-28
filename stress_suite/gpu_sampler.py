"""Background nvidia-smi CSV sampling."""

from __future__ import annotations

import csv
import io
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Tuple


def _sample_block() -> Optional[Tuple[float, list[list[str]]]]:
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode != 0:
        return None
    host_t = time.time()
    rows: list[list[str]] = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(next(csv.reader(io.StringIO(line))))
    return host_t, rows


def start_gpu_sampler(
    out_path: Path,
    interval_seconds: float,
    stop_event: threading.Event,
) -> threading.Thread:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "host_unix_ts",
        "timestamp",
        "index",
        "name",
        "util_gpu_pct",
        "util_mem_pct",
        "mem_used_mib",
        "mem_total_mib",
    ]
    first = True

    def _run():
        nonlocal first
        while not stop_event.is_set():
            block = _sample_block()
            if block:
                host_t, gpu_rows = block
                mode = "w" if first else "a"
                with out_path.open(mode, newline="") as f:
                    w = csv.writer(f)
                    if first:
                        w.writerow(header)
                        first = False
                    for row in gpu_rows:
                        w.writerow([host_t] + row)
            stop_event.wait(interval_seconds)

    t = threading.Thread(target=_run, name="gpu-sampler", daemon=True)
    t.start()
    return t
