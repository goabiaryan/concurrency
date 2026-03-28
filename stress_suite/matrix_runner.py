"""Run multiple merged configs sequentially (e.g. quantization / GPU SKU matrix)."""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

import yaml

from stress_suite.config_util import deep_merge
from stress_suite.runner import run_suite


def main() -> None:
    ap = argparse.ArgumentParser(description="Run stress suite matrix from YAML")
    ap.add_argument("--matrix", required=True, help="Matrix YAML (variants + base_config)")
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="Pass through to each run (skip docker build)",
    )
    args = ap.parse_args()
    matrix_path = Path(args.matrix).resolve()
    doc = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    base_rel = doc.get("base_config")
    if not base_rel:
        sys.exit("matrix file must contain base_config: path/to.yaml")
    base_path = (matrix_path.parent / base_rel).resolve()
    if not base_path.exists():
        base_path = Path(base_rel).resolve()
    if not base_path.exists():
        sys.exit(f"base_config not found: {base_rel}")
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    variants = doc.get("variants") or []
    if not variants:
        sys.exit("matrix file must contain non-empty variants: list")

    exit_max = 0
    for i, var in enumerate(variants):
        label = var.get("label", f"variant_{i}")
        overrides = var.get("overrides") or {}
        merged = deep_merge(base_cfg, overrides)
        if var.get("label"):
            merged["suite_name"] = str(var["label"])
        tmp = Path(tempfile.mkdtemp(prefix="stress_matrix_")) / "merged.yaml"
        tmp.write_text(yaml.safe_dump(merged, default_flow_style=False, sort_keys=False), encoding="utf-8")
        print(f"=== matrix {i + 1}/{len(variants)}: {label} -> {merged.get('output_dir')} ===")
        code = asyncio.run(run_suite(merged, skip_build=args.skip_build))
        exit_max = max(exit_max, code)
    sys.exit(exit_max)


if __name__ == "__main__":
    main()
