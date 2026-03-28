"""Deep-merge YAML overrides and CLI patches."""

from __future__ import annotations

import copy
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_cli_overrides(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    """Mutates cfg in place from argparse Namespace."""
    if getattr(args, "containers", None) is not None:
        cfg.setdefault("containers", {})["count"] = int(args.containers)
    if getattr(args, "base_host_port", None) is not None:
        cfg.setdefault("containers", {})["base_host_port"] = int(args.base_host_port)
    if getattr(args, "waves", None) is not None:
        cfg.setdefault("scenarios", {}).setdefault("waves", {})["count"] = int(args.waves)
    if getattr(args, "concurrent", None) is not None:
        cfg.setdefault("scenarios", {}).setdefault("waves", {})[
            "concurrent_per_container"
        ] = int(args.concurrent)
    if getattr(args, "output_dir", None):
        cfg["output_dir"] = str(args.output_dir)
    if getattr(args, "suite_name", None):
        cfg["suite_name"] = str(args.suite_name)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = int(args.seed)
    if getattr(args, "quantization", None) is not None:
        q = args.quantization
        cfg["quantization"] = None if q in ("", "null", "none") else str(q)
    if getattr(args, "gpu_sku", None):
        cfg.setdefault("run_labels", {})["gpu_sku"] = str(args.gpu_sku)
    return cfg
