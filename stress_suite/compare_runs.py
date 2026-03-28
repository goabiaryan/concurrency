"""Compare two summary.json files (baseline vs stressed run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _dig(d: dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare stress suite summary.json files")
    ap.add_argument("baseline", type=Path, help="First summary (e.g. single-container baseline)")
    ap.add_argument("stressed", type=Path, help="Second summary (e.g. max stress)")
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON diff instead of text",
    )
    args = ap.parse_args()
    a = json.loads(args.baseline.read_text(encoding="utf-8"))
    b = json.loads(args.stressed.read_text(encoding="utf-8"))

    keys = [
        ("error_rate", ("requests", "error_rate")),
        ("p99_ttft_s", ("requests", "ttft_s", "p99")),
        ("p99_total_s", ("requests", "total_s", "p99")),
        ("p99_tpot", ("requests", "tpot_s_per_tok", "p99")),
        ("mean_tok_per_s", ("requests", "throughput", "output_tokens_per_s_mean_per_request")),
    ]
    diff: dict[str, Any] = {"baseline_path": str(args.baseline), "stressed_path": str(args.stressed)}
    lines = []
    for name, path in keys:
        va = _dig(a, *path)
        vb = _dig(b, *path)
        diff[name] = {"baseline": va, "stressed": vb}
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va not in (0, None):
            try:
                delta_pct = 100.0 * (float(vb) - float(va)) / float(va)
            except (TypeError, ValueError, ZeroDivisionError):
                delta_pct = None
            diff[name]["delta_pct_vs_baseline"] = delta_pct
            lines.append(f"{name}: baseline={va} stressed={vb} ({delta_pct:+.1f}% vs baseline)")
        else:
            lines.append(f"{name}: baseline={va} stressed={vb}")

    if args.json:
        print(json.dumps(diff, indent=2, default=str))
    else:
        print("\n".join(lines))
        print(
            "\nTip: use requests.by_phase in each summary JSON for burst vs sustained breakdown."
        )


if __name__ == "__main__":
    main()
