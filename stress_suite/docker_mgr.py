"""Start/stop inference containers with per-replica GPU pinning."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import uuid
from typing import Any

import docker
import docker.types

log = logging.getLogger(__name__)


def detect_gpu_count() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        lines = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("GPU ")]
        return max(1, len(lines))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 1


class InferenceFleet:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.client = docker.from_env()
        self.run_id = uuid.uuid4().hex[:10]
        self.containers: list[docker.models.containers.Container] = []
        self._lock = threading.Lock()

    def build_image_if_needed(self) -> None:
        dockerfile = self.cfg.get("build_dockerfile")
        if not dockerfile:
            return
        tag = self.cfg["image"]
        ctx = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        df_path = dockerfile
        log.info("Building image %s from %s (context %s)", tag, df_path, ctx)
        self.client.images.build(path=ctx, dockerfile=df_path, tag=tag, rm=True)

    def _vllm_command(self) -> list[str]:
        m = self.cfg["model"]
        dtype = self.cfg.get("dtype") or "float16"
        q = self.cfg.get("quantization")
        extra = self.cfg.get("vllm_extra_args") or []
        served = self.cfg.get("served_model_name")
        cmd: list[str] = [
            "--model",
            m,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--dtype",
            str(dtype),
        ]
        if served:
            cmd.extend(["--served-model-name", str(served)])
        if q:
            cmd.extend(["--quantization", str(q)])
        cmd.extend([str(x) for x in extra])
        return cmd

    def _environment(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            v = os.environ.get(key)
            if v:
                env["HUGGING_FACE_HUB_TOKEN"] = v
                break
        return env

    def start_replica(self, index: int, host_port: int, gpu_id: int) -> docker.models.containers.Container:
        name = f"stress_{self.cfg.get('suite_name', 'run')}_{self.run_id}_{index}"
        image = self.cfg["image"]
        backend = (self.cfg.get("backend") or "vllm").lower()
        if backend != "vllm":
            raise ValueError(
                f"Unsupported backend {backend!r}: only vLLM (Docker) replicas are supported."
            )
        shm_gb = int(self.cfg.get("containers", {}).get("shm_size_gb") or 2)
        shm_b = max(1, shm_gb) * (1024**3)

        device_requests = [
            docker.types.DeviceRequest(device_ids=[str(gpu_id)], capabilities=[["gpu"]])
        ]

        kwargs: dict[str, Any] = {
            "image": image,
            "detach": True,
            "name": name,
            "ports": {8000: host_port},
            "device_requests": device_requests,
            "shm_size": shm_b,
            "environment": self._environment(),
        }

        kwargs["command"] = self._vllm_command()

        labels = {
            "stress.suite": str(self.cfg.get("suite_name", "run"))[:63],
            "stress.run_id": self.run_id,
            "stress.replica_index": str(index),
        }
        for k, v in (self.cfg.get("run_labels") or {}).items():
            key = f"stress.{str(k)[:40]}"
            labels[key] = str(v)[:200]
        kwargs["labels"] = labels

        with self._lock:
            c = self.client.containers.run(**kwargs)
            self.containers.append(c)
        log.info("Started %s on host port %s GPU %s", name, host_port, gpu_id)
        return c

    def stop_all(self) -> None:
        for c in reversed(self.containers):
            try:
                c.stop(timeout=15)
            except Exception as e:  # noqa: BLE001
                log.warning("stop %s: %s", c.name, e)
            try:
                c.remove()
            except Exception as e:  # noqa: BLE001
                log.warning("remove %s: %s", c.name, e)
        self.containers.clear()
