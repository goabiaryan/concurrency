"""CLI orchestrator: build/start replicas, run scenarios, write CSV/JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml

from stress_suite import loadgen, metrics, span_logging
from stress_suite.config_util import apply_cli_overrides
from stress_suite.docker_events import DockerEventWatcher
from stress_suite.docker_mgr import InferenceFleet, detect_gpu_count
from stress_suite.gpu_sampler import start_gpu_sampler

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("stress_suite.runner")


async def wait_ready(client: httpx.AsyncClient, base: str, timeout: float) -> bool:
    models = f"{base.rstrip('/')}/v1/models"
    health = f"{base.rstrip('/')}/health"
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        for url in (models, health):
            try:
                r = await client.get(url, timeout=15)
                if r.status_code < 500:
                    return True
            except Exception:  # noqa: BLE001
                pass
        await asyncio.sleep(2)
    return False


def _summarize_docker_events(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"n": 0}
    actions: Counter[str] = Counter()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        actions[str(e.get("action") or "unknown")] += 1
    return {"n": n, "by_action": dict(actions)}


def _start_replicas_parallel(
    fleet: InferenceFleet,
    n: int,
    ports: list[int],
    num_gpus: int,
) -> None:
    def one(i: int) -> None:
        fleet.start_replica(i, ports[i], i % num_gpus)

    with ThreadPoolExecutor(max_workers=min(32, max(1, n))) as ex:
        list(ex.map(one, range(n)))


async def run_suite(cfg: dict[str, Any], skip_build: bool) -> int:
    out = Path(cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_ts = span_logging.utc_slug()
    log_run_dir = out / "logs" / f"run_{run_ts}"
    fh: logging.FileHandler | None = None
    spans_written: list[str] = []
    prom_timeout = float(cfg.get("prometheus_scrape_timeout_seconds") or 30.0)
    global_chaos = cfg.get("network_chaos") or None
    record_docker = cfg.get("docker_events", True)

    gpu_inv = metrics.detect_gpu_inventory()
    meta = {
        "suite_name": cfg.get("suite_name"),
        "started_unix": time.time(),
        "logs_run": str(log_run_dir.relative_to(out)),
        "gpu_inventory": gpu_inv,
        "config": cfg,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    try:
        fh = span_logging.setup_run_file_log(log_run_dir)
    except OSError as e:
        log.warning("Could not create orchestrator.log: %s", e)
    log.info("Run logs -> %s (orchestrator.log + spans/)", log_run_dir)

    fleet = InferenceFleet(cfg)
    watcher: DockerEventWatcher | None = None
    if record_docker:
        watcher = DockerEventWatcher(str(cfg.get("suite_name", "run")), fleet.run_id)
        watcher.start()

    if not skip_build and cfg.get("build_dockerfile"):
        fleet.build_image_if_needed()

    n = int(cfg["containers"]["count"])
    base_port = int(cfg["containers"]["base_host_port"])
    num_gpus = cfg["containers"].get("num_gpus")
    if num_gpus is None:
        num_gpus = detect_gpu_count()
    num_gpus = int(num_gpus)
    log.info("Assigning %s replicas across %s GPU(s)", n, num_gpus)

    ports = [base_port + i for i in range(n)]
    endpoints = [f"http://127.0.0.1:{p}" for p in ports]

    t_launch0 = time.perf_counter()
    parallel = cfg.get("containers", {}).get("parallel_start", True)
    if parallel and n > 1:
        _start_replicas_parallel(fleet, n, ports, num_gpus)
    else:
        for i in range(n):
            fleet.start_replica(i, ports[i], i % num_gpus)
    t_launch1 = time.perf_counter()

    stop_gpu = threading.Event()
    gpu_thread = None
    iv = float(cfg.get("gpu_sample_interval_seconds") or 0)
    gpu_csv = out / "gpu_timeseries.csv"
    if iv > 0:
        gpu_thread = start_gpu_sampler(gpu_csv, iv, stop_gpu)

    req_timeout = float(cfg["timeouts"]["request_seconds"])
    ready_timeout = float(cfg["timeouts"]["container_ready_seconds"])
    model_name = str(cfg["served_model_name"])
    rng = random.Random(cfg.get("seed") if cfg.get("seed") is not None else int(time.time()))

    records: list[dict[str, Any]] = []
    exit_code = 0
    cold_rows: list[dict[str, Any]] = []
    id_to_ep: dict[int, str] = {}
    rec_mark = 0

    try:
        async with httpx.AsyncClient() as client:
            async def measure_ready(cid: int, ep: str) -> dict[str, Any]:
                t0 = time.perf_counter()
                ok = await wait_ready(client, ep, ready_timeout)
                return {
                    "container_id": cid,
                    "endpoint": ep,
                    "host_port": ports[cid],
                    "gpu_id": cid % num_gpus,
                    "ready_ok": ok,
                    "ready_wait_s": time.perf_counter() - t0,
                }

            cold_rows = list(
                await asyncio.gather(*[measure_ready(i, endpoints[i]) for i in range(n)])
            )
            for row in cold_rows:
                row["batch_launch_window_s"] = t_launch1 - t_launch0
            pd.DataFrame(cold_rows).to_csv(out / "cold_start.csv", index=False)

            ready_pairs = [(r["container_id"], r["endpoint"]) for r in cold_rows if r["ready_ok"]]
            id_to_ep = {cid: ep for cid, ep in ready_pairs}
            if not ready_pairs:
                log.error("No replicas became ready in time.")
                return 2
            if len(ready_pairs) < n:
                log.warning(
                    "Continuing with %s/%s healthy replicas",
                    len(ready_pairs),
                    n,
                )

            p = await span_logging.flush_cold_start_span(
                log_run_dir,
                client,
                ready_pairs,
                cold_rows,
                prom_timeout,
            )
            spans_written.append(str(p))

            scen = cfg["scenarios"]
            wu = scen["warmup"]
            for cid, ep in ready_pairs:
                wrows = await loadgen.warmup_container(
                    client,
                    ep,
                    model_name,
                    int(wu["requests_per_container"]),
                    int(wu["max_output_tokens"]),
                    float(wu.get("gap_between_requests_ms") or 0),
                    req_timeout,
                    rng,
                    phase_cfg=wu,
                    global_chaos=global_chaos,
                )
                for r in wrows:
                    r["base_url"] = ep
                    r["container_id"] = cid
                records.extend(wrows)

            p = await span_logging.flush_span(
                log_run_dir,
                "warmup",
                client,
                ready_pairs,
                records[rec_mark:],
                prom_timeout,
            )
            spans_written.append(str(p))
            rec_mark = len(records)

            waves = scen["waves"]
            n_waves = int(waves["count"])
            for w in range(n_waves):
                if w > 0:
                    await asyncio.sleep(float(waves["gap_seconds"]))
                tasks = [
                    loadgen.wave_for_container(
                        client,
                        ep,
                        model_name,
                        int(waves["concurrent_per_container"]),
                        int(waves["requests_per_container_per_wave"]),
                        (int(waves["jitter_ms"][0]), int(waves["jitter_ms"][1])),
                        (
                            int(waves["prompt_token_est_range"][0]),
                            int(waves["prompt_token_est_range"][1]),
                        ),
                        (
                            int(waves["output_token_range"][0]),
                            int(waves["output_token_range"][1]),
                        ),
                        req_timeout,
                        rng,
                        w,
                        cid,
                        phase_cfg=waves,
                        global_chaos=global_chaos,
                    )
                    for cid, ep in ready_pairs
                ]
                wave_chunks = await asyncio.gather(*tasks)
                for chunk in wave_chunks:
                    for r in chunk:
                        r["base_url"] = id_to_ep.get(int(r.get("container_id", -1)), "")
                    records.extend(chunk)
                p = await span_logging.flush_span(
                    log_run_dir,
                    f"wave_{w}",
                    client,
                    ready_pairs,
                    records[rec_mark:],
                    prom_timeout,
                )
                spans_written.append(str(p))
                rec_mark = len(records)

            sus = scen.get("sustained") or {}
            if sus.get("enabled"):
                dur = float(sus["duration_seconds"])
                rps = float(sus["rps_per_container"])
                pr = (
                    int(waves["prompt_token_est_range"][0]),
                    int(waves["prompt_token_est_range"][1]),
                )
                ort = (
                    int(waves["output_token_range"][0]),
                    int(waves["output_token_range"][1]),
                )
                sus_phase = {**waves, **{k: v for k, v in sus.items() if k != "enabled"}}
                stasks = [
                    loadgen.sustained_rps(
                        client,
                        ep,
                        model_name,
                        dur,
                        rps,
                        pr,
                        ort,
                        req_timeout,
                        rng,
                        cid,
                        phase_cfg=sus_phase,
                        global_chaos=global_chaos,
                    )
                    for cid, ep in ready_pairs
                ]
                sustained_chunks = await asyncio.gather(*stasks)
                for chunk in sustained_chunks:
                    for r in chunk:
                        r["base_url"] = id_to_ep.get(r.get("container_id"), "")
                    records.extend(chunk)
                p = await span_logging.flush_span(
                    log_run_dir,
                    "sustained",
                    client,
                    ready_pairs,
                    records[rec_mark:],
                    prom_timeout,
                )
                spans_written.append(str(p))
                rec_mark = len(records)

            p = await span_logging.flush_span(
                log_run_dir,
                "final",
                client,
                ready_pairs,
                records,
                prom_timeout,
                write_chunk_csv=False,
            )
            spans_written.append(str(p))

        df = pd.DataFrame.from_records(records)
        df.to_csv(out / "requests.csv", index=False)

        cold_df = pd.DataFrame(cold_rows) if cold_rows else pd.DataFrame()
        events_path = out / "docker_events.jsonl"
        if watcher is not None:
            watcher.stop_and_write(events_path)
        docker_ev = (
            _summarize_docker_events(events_path)
            if record_docker
            else {"n": 0, "disabled": True}
        )

        summary = {
            "suite_name": cfg.get("suite_name"),
            "finished_unix": time.time(),
            "run_labels": cfg.get("run_labels") or {},
            "requests": metrics.summarize_requests(df),
            "cold_start": {
                "mean_ready_wait_s": float(cold_df["ready_wait_s"].mean())
                if not cold_df.empty and "ready_wait_s" in cold_df.columns
                else float("nan"),
                "all_ready": bool(all(r["ready_ok"] for r in cold_rows))
                if cold_rows
                else False,
                "p99_ready_wait_s": float(cold_df["ready_wait_s"].quantile(0.99))
                if not cold_df.empty and "ready_wait_s" in cold_df.columns
                else float("nan"),
            },
            "gpu_inventory": gpu_inv,
            "gpu_peaks": metrics.gpu_peaks_from_csv(gpu_csv),
            "docker_events": docker_ev,
        }
        metrics.write_summary(out / "summary.json", summary)
        shutil.copy2(out / "summary.json", log_run_dir / "summary.json")
        meta["finished_unix"] = summary["finished_unix"]
        meta["summary_paths"] = {"summary_json": "summary.json", "requests_csv": "requests.csv"}
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        log.info("Wrote %s", out.resolve())
        print(json.dumps(summary, indent=2, default=str))
    except Exception as e:  # noqa: BLE001
        log.exception("Run failed: %s", e)
        exit_code = 1
    finally:
        if watcher is not None:
            try:
                watcher.stop_and_write(out / "docker_events.jsonl")
            except Exception:  # noqa: BLE001
                pass
        try:
            span_logging.write_manifest(log_run_dir, spans_written)
        except Exception:  # noqa: BLE001
            pass
        if fh is not None:
            span_logging.teardown_run_file_log(fh)
        stop_gpu.set()
        if gpu_thread is not None:
            gpu_thread.join(timeout=5)
        fleet.stop_all()

    return exit_code


def main() -> None:
    ap = argparse.ArgumentParser(description="TinyLLaMA / vLLM stress harness")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip docker build even if build_dockerfile is set",
    )
    ap.add_argument("--containers", type=int, help="Override containers.count")
    ap.add_argument("--base-host-port", type=int, help="Override containers.base_host_port")
    ap.add_argument("--waves", type=int, help="Override scenarios.waves.count")
    ap.add_argument(
        "--concurrent",
        type=int,
        help="Override scenarios.waves.concurrent_per_container",
    )
    ap.add_argument("--output-dir", type=str, help="Override output_dir")
    ap.add_argument("--suite-name", type=str, help="Override suite_name (labels + naming)")
    ap.add_argument("--seed", type=int, help="Override seed")
    ap.add_argument(
        "--quantization",
        type=str,
        help="Override quantization (use empty string for null)",
    )
    ap.add_argument(
        "--gpu-sku",
        type=str,
        help="Tag results (stored in run_labels.gpu_sku and Docker labels)",
    )
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    apply_cli_overrides(cfg, args)
    code = asyncio.run(run_suite(cfg, skip_build=args.skip_build))
    sys.exit(code)


if __name__ == "__main__":
    main()
