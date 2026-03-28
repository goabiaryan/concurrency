"""Per-span disk + console logging: custom harness metrics and vLLM Prometheus scrape."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd

from stress_suite import metrics

log = logging.getLogger(__name__)

_PROM_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+eE0-9.]+|NaN|nan|\+Inf|-Inf)$"
)


def utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def setup_run_file_log(log_run_dir: Path) -> logging.FileHandler:
    log_run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_run_dir / "orchestrator.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(fh)
    return fh


def teardown_run_file_log(fh: logging.FileHandler) -> None:
    root = logging.getLogger()
    root.removeHandler(fh)
    fh.close()


def _parse_prom_counters_gauges(prom_text: str) -> dict[str, float]:
    """Best-effort parse of scalar exposition lines (skips HELP/TYPE/comments, histogram buckets)."""
    out: dict[str, float] = {}
    for raw in prom_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name, val_s = m.group(1), m.group(2)
        if "vllm:" not in name and not name.startswith("vllm_"):
            continue
        if name.endswith("_bucket"):
            continue
        try:
            out[name] = float(val_s)
        except ValueError:
            continue
    return out


async def scrape_prometheus(
    client: httpx.AsyncClient,
    base_url: str,
    timeout: float = 30.0,
) -> tuple[str | None, str | None]:
    url = f"{base_url.rstrip('/')}/metrics"
    try:
        r = await client.get(url, timeout=timeout)
        if r.status_code >= 400:
            return None, f"http_{r.status_code}:{r.text[:300]!r}"
        return r.text, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


async def scrape_all_replicas(
    client: httpx.AsyncClient,
    ready_pairs: list[tuple[int, str]],
    timeout: float = 30.0,
) -> dict[int, tuple[str | None, str | None]]:
    out: dict[int, tuple[str | None, str | None]] = {}
    for cid, ep in ready_pairs:
        text, err = await scrape_prometheus(client, ep, timeout=timeout)
        out[cid] = (text, err)
    return out


def summarize_records_chunk(chunk: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(chunk)
    if not rows:
        return {"n": 0, "note": "no requests in this span"}
    return metrics.summarize_requests(pd.DataFrame.from_records(rows))


def emit_span_console(
    span_name: str,
    custom_summary: dict[str, Any],
    prom_by_cid: dict[int, tuple[str | None, str | None]],
) -> None:
    log.info("---------- span=%s ----------", span_name)
    log.info("custom (harness): %s", json.dumps(custom_summary, indent=2, default=str))
    for cid, (text, err) in prom_by_cid.items():
        if err:
            log.info("vLLM replica %s /metrics: ERROR %s", cid, err)
            continue
        if not text:
            log.info("vLLM replica %s /metrics: empty", cid)
            continue
        parsed = _parse_prom_counters_gauges(text)
        # Prefer a few high-signal keys if present (names vary by vLLM version)
        preferred = [
            k
            for k in sorted(parsed)
            if any(
                x in k
                for x in (
                    "num_requests_running",
                    "num_requests_waiting",
                    "generation_tokens",
                    "prompt_tokens",
                    "request_success",
                    "request_failure",
                    "time_to_first_token",
                    "time_per_output_token",
                )
            )
        ][:24]
        snap = {k: parsed[k] for k in preferred}
        if not snap and parsed:
            snap = dict(list(parsed.items())[:20])
        log.info(
            "vLLM replica %s /metrics: %s scalar vllm_* keys (showing up to 20)",
            cid,
            len(parsed),
        )
        log.info("vLLM replica %s snapshot: %s", cid, json.dumps(snap, indent=2, default=str))


async def flush_span(
    logs_run_dir: Path,
    span_name: str,
    client: httpx.AsyncClient,
    ready_pairs: list[tuple[int, str]],
    records_chunk: list[dict[str, Any]],
    prom_timeout: float,
    write_chunk_csv: bool = True,
) -> Path:
    """
    Write under logs_run_dir/spans/<span>_<utc>/:
      - custom_summary.json, requests_chunk.csv (if any)
      - vllm/replica_<id>.prom (raw), vllm/replica_<id>.json (parsed scalars)
      - console_snapshot.txt (duplicate of key info for humans)
    """
    span_ts = utc_slug()
    span_dir = logs_run_dir / "spans" / f"{span_name}_{span_ts}"
    span_dir.mkdir(parents=True, exist_ok=True)
    vdir = span_dir / "vllm"
    vdir.mkdir(exist_ok=True)

    custom_summary = summarize_records_chunk(records_chunk)
    (span_dir / "custom_summary.json").write_text(
        json.dumps(custom_summary, indent=2, default=str),
        encoding="utf-8",
    )
    if records_chunk and write_chunk_csv:
        pd.DataFrame.from_records(records_chunk).to_csv(
            span_dir / "requests_chunk.csv", index=False
        )

    prom_by_cid = await scrape_all_replicas(client, ready_pairs, timeout=prom_timeout)
    emit_span_console(span_name, custom_summary, prom_by_cid)

    snapshot_lines = [
        f"span={span_name}",
        f"utc={span_ts}",
        "",
        "=== custom_summary.json ===",
        json.dumps(custom_summary, indent=2, default=str),
        "",
    ]

    for cid, (text, err) in prom_by_cid.items():
        if text:
            (vdir / f"replica_{cid}.prom").write_text(text, encoding="utf-8")
            parsed = _parse_prom_counters_gauges(text)
            (vdir / f"replica_{cid}.json").write_text(
                json.dumps(parsed, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            (vdir / f"replica_{cid}.error.txt").write_text(err or "empty", encoding="utf-8")
        snapshot_lines.append(f"=== vLLM replica {cid} ===")
        if err:
            snapshot_lines.append(f"error: {err}")
        elif text:
            # keep console snapshot smaller: vllm lines only
            vl = [
                ln
                for ln in text.splitlines()
                if not ln.startswith("#")
                and ("vllm:" in ln or ln.startswith("vllm_"))
            ]
            snapshot_lines.extend(vl[:400])
            if len(vl) > 400:
                snapshot_lines.append(f"... ({len(vl) - 400} more vllm lines in replica_{cid}.prom)")
        snapshot_lines.append("")

    (span_dir / "console_snapshot.txt").write_text(
        "\n".join(snapshot_lines), encoding="utf-8"
    )
    log.info("Wrote span logs to %s", span_dir.resolve())
    return span_dir


async def flush_cold_start_span(
    logs_run_dir: Path,
    client: httpx.AsyncClient,
    ready_pairs: list[tuple[int, str]],
    cold_rows: list[dict[str, Any]],
    prom_timeout: float,
) -> Path:
    """Replicas are up; scrape /metrics + save cold_start table summary."""
    span_name = "cold_start"
    span_ts = utc_slug()
    span_dir = logs_run_dir / "spans" / f"{span_name}_{span_ts}"
    span_dir.mkdir(parents=True, exist_ok=True)
    vdir = span_dir / "vllm"
    vdir.mkdir(exist_ok=True)

    cold_summary = {
        "n_replicas": len(cold_rows),
        "ready_ok": [r for r in cold_rows if r.get("ready_ok")],
        "rows": cold_rows,
    }
    (span_dir / "cold_start.json").write_text(
        json.dumps(cold_summary, indent=2, default=str),
        encoding="utf-8",
    )
    if cold_rows:
        pd.DataFrame(cold_rows).to_csv(span_dir / "cold_start.csv", index=False)

    prom_by_cid = await scrape_all_replicas(client, ready_pairs, timeout=prom_timeout)
    emit_span_console(
        span_name,
        {
            "cold_start_replicas": len(cold_rows),
            "ready_ok_count": sum(1 for r in cold_rows if r.get("ready_ok")),
        },
        prom_by_cid,
    )

    snapshot_lines = [f"span={span_name}", f"utc={span_ts}", "", "=== cold_start ===", ""]
    for r in cold_rows:
        snapshot_lines.append(json.dumps(r, default=str))

    for cid, (text, err) in prom_by_cid.items():
        if text:
            (vdir / f"replica_{cid}.prom").write_text(text, encoding="utf-8")
            parsed = _parse_prom_counters_gauges(text)
            (vdir / f"replica_{cid}.json").write_text(
                json.dumps(parsed, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            (vdir / f"replica_{cid}.error.txt").write_text(err or "empty", encoding="utf-8")
        snapshot_lines.append(f"=== vLLM replica {cid} (post-ready) ===")
        if text:
            vl = [
                ln
                for ln in text.splitlines()
                if ("vllm:" in ln or ln.startswith("vllm_")) and not ln.startswith("#")
            ]
            snapshot_lines.extend(vl[:200])

    (span_dir / "console_snapshot.txt").write_text(
        "\n".join(snapshot_lines), encoding="utf-8"
    )
    log.info("Wrote cold_start span logs to %s", span_dir.resolve())
    return span_dir


def write_manifest(logs_run_dir: Path, spans_written: list[str]) -> None:
    (logs_run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "spans": spans_written,
                "written_unix": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
