"""Aggregate request logs and write summary JSON."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _pct(series: pd.Series, q: float) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(np.percentile(s.to_numpy(dtype=float), q))


def gpu_peaks_from_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"note": "no gpu_timeseries.csv"}
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return {"note": "unreadable"}
    if df.empty:
        return {"note": "empty"}
    out: dict[str, Any] = {}
    idx_col = "index" if "index" in df.columns else None
    if idx_col:
        for gpu_idx, g in df.groupby(idx_col):
            util = pd.to_numeric(g.get("util_gpu_pct", pd.Series(dtype=float)), errors="coerce")
            memu = pd.to_numeric(
                g.get("mem_used_mib", pd.Series(dtype=float)), errors="coerce"
            )
            out[f"gpu_{gpu_idx}"] = {
                "util_gpu_pct_max": float(util.max()) if not util.empty else float("nan"),
                "util_gpu_pct_mean": float(util.mean()) if not util.empty else float("nan"),
                "mem_used_mib_max": float(memu.max()) if not memu.empty else float("nan"),
                "mem_used_mib_mean": float(memu.mean()) if not memu.empty else float("nan"),
            }
    else:
        util = pd.to_numeric(df.get("util_gpu_pct", pd.Series(dtype=float)), errors="coerce")
        memu = pd.to_numeric(df.get("mem_used_mib", pd.Series(dtype=float)), errors="coerce")
        out["aggregate"] = {
            "util_gpu_pct_max": float(util.max()) if not util.empty else float("nan"),
            "mem_used_mib_max": float(memu.max()) if not memu.empty else float("nan"),
        }
    return out


def detect_gpu_inventory() -> dict[str, Any]:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if r.returncode != 0:
            return {"error": r.stderr[:200]}
        rows = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                rows.append(
                    {"index": parts[0], "name": parts[1], "memory_total_mib": parts[2]}
                )
        return {"gpus": rows}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def summarize_requests(
    df: pd.DataFrame,
    *,
    include_phase_breakdown: bool = True,
    include_tier_breakdown: bool = True,
) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    if "ok" in df.columns:
        ok_mask = df["ok"].fillna(False).astype(bool)
    else:
        ok_mask = pd.Series(True, index=df.index)
    ok = df[ok_mask]
    err = df[~ok_mask]
    out: dict[str, Any] = {
        "n_total": int(len(df)),
        "n_ok": int(len(ok)),
        "n_error": int(len(err)),
        "error_rate": float(len(err) / max(1, len(df))),
    }
    if "error_class" in df.columns:
        out["error_classes"] = (
            df["error_class"].fillna("unknown").value_counts().to_dict()
        )
    if ok.empty:
        return out
    tok_col = "output_tokens_effective" if "output_tokens_effective" in ok.columns else "output_tokens_est"
    if tok_col in ok.columns and "total_s" in ok.columns:
        t = pd.to_numeric(ok[tok_col], errors="coerce")
        lat = pd.to_numeric(ok["total_s"], errors="coerce")
        ratio = (t / lat).replace([np.inf, -np.inf], np.nan).dropna()
        out["throughput"] = {
            "output_tokens_per_s_mean_per_request": float(ratio.mean())
            if not ratio.empty
            else float("nan"),
            "output_tokens_per_s_p99_per_request": float(np.percentile(ratio, 99))
            if len(ratio) > 0
            else float("nan"),
            "sum_output_tokens_div_sum_latency_s": float(t.sum() / max(1e-9, lat.sum())),
        }
        if "container_id" in ok.columns:
            by_c: dict[str, Any] = {}
            for cid, g in ok.groupby("container_id"):
                t2 = pd.to_numeric(g[tok_col], errors="coerce")
                l2 = pd.to_numeric(g["total_s"], errors="coerce")
                by_c[str(cid)] = float(t2.sum() / max(1e-9, l2.sum()))
            out["throughput"]["by_container_id"] = by_c
    if "prefill_proxy_s" in ok.columns and "decode_phase_s" in ok.columns:
        pre = pd.to_numeric(ok["prefill_proxy_s"], errors="coerce")
        dec = pd.to_numeric(ok["decode_phase_s"], errors="coerce")
        out["prefill_decode_proxy"] = {
            "prefill_proxy_s_p50": _pct(pre, 50),
            "prefill_proxy_s_p99": _pct(pre, 99),
            "decode_phase_s_p50": _pct(dec, 50),
            "decode_phase_s_p99": _pct(dec, 99),
            "decode_to_prefill_ratio_p50": float((dec / pre.replace(0, np.nan)).median())
            if not pre.empty
            else float("nan"),
        }
    if "usage_prompt_tokens" in ok.columns:
        up = pd.to_numeric(ok["usage_prompt_tokens"], errors="coerce").dropna()
        if not up.empty:
            out["usage_prompt_tokens_from_server"] = {
                "mean": float(up.mean()),
                "p99": float(np.percentile(up, 99)),
            }
    if "usage_completion_tokens" in ok.columns:
        uc = pd.to_numeric(ok["usage_completion_tokens"], errors="coerce").dropna()
        if not uc.empty:
            out["usage_completion_tokens_from_server"] = {
                "mean": float(uc.mean()),
                "p99": float(np.percentile(uc, 99)),
            }
    for col, label in [
        ("ttft_s", "ttft_s"),
        ("total_s", "total_s"),
        ("tpot_s_per_tok", "tpot_s_per_tok"),
        ("output_tokens_est", "output_tokens_est"),
    ]:
        if col not in ok.columns:
            continue
        s = ok[col].astype(float)
        out[label] = {
            "mean": float(s.mean()),
            "p50": _pct(s, 50),
            "p90": _pct(s, 90),
            "p99": _pct(s, 99),
            "max": float(s.max()),
        }
    if include_phase_breakdown and "phase" in df.columns:
        by_phase: dict[str, Any] = {}
        for ph, g in df.groupby("phase"):
            by_phase[str(ph)] = summarize_requests(
                g,
                include_phase_breakdown=False,
                include_tier_breakdown=include_tier_breakdown,
            )
        out["by_phase"] = by_phase
    if include_tier_breakdown and "prompt_tier" in ok.columns:
        by_tier: dict[str, Any] = {}
        for tier, g in ok.groupby("prompt_tier"):
            by_tier[str(tier)] = summarize_requests(
                g,
                include_phase_breakdown=False,
                include_tier_breakdown=False,
            )
        out["by_prompt_tier"] = by_tier
    return out


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
