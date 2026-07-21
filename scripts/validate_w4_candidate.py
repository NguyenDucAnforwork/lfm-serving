#!/usr/bin/env python3
"""Validate a W4A16 GPTQ/compressed-tensors artifact and run directory.

This is a guardrail for the decode-cost experiments. It checks the constraints
from the goal before a W4 run is considered a real candidate:

  * compressed-tensors quantization, not bitsandbytes;
  * W4 int weights, symmetric, expected group_size;
  * lm_head ignored/unquantized;
  * no fp8_per_tensor / KV FP8 mixed into the same W4 run config;
  * server log confirms vLLM selected Marlin or Machete for
    CompressedTensorsWNA16.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def require(ok: bool, msg: str, failures: list[str]) -> None:
    if not ok:
        failures.append(msg)


def find_weight_config(qcfg: dict[str, Any]) -> dict[str, Any]:
    groups = qcfg.get("config_groups") or {}
    if not isinstance(groups, dict) or not groups:
        return {}
    first = next(iter(groups.values()))
    return first.get("weights") if isinstance(first, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", required=True, help="Compressed model directory")
    ap.add_argument("--run-dir", default=None, help="Optional results/<run> directory")
    ap.add_argument("--config", default=None, help="Optional config.env path; defaults to run-dir/config.env")
    ap.add_argument("--server-log", default=None, help="Optional server.log path; defaults to run-dir/server.log")
    ap.add_argument("--expected-group-size", type=int, default=128, help="Expected W4 group size")
    args = ap.parse_args()

    artifact = Path(args.artifact)
    config_json = artifact / "config.json"
    failures: list[str] = []
    require(config_json.exists(), f"missing artifact config: {config_json}", failures)

    qcfg: dict[str, Any] = {}
    if config_json.exists():
        config = json.loads(config_json.read_text())
        qcfg = config.get("quantization_config") or {}
        weights = find_weight_config(qcfg)
        ignore = qcfg.get("ignore") or []
        require(qcfg.get("quant_method") == "compressed-tensors", "quant_method is not compressed-tensors", failures)
        require(qcfg.get("format") == "pack-quantized", "quantization format is not pack-quantized", failures)
        require(qcfg.get("quantization_status") == "compressed", "quantization_status is not compressed", failures)
        require(weights.get("num_bits") == 4, "weight num_bits is not 4", failures)
        require(weights.get("type") == "int", "weight type is not int", failures)
        require(weights.get("symmetric") is True, "weight symmetric is not true", failures)
        require(weights.get("strategy") == "group", "weight strategy is not group", failures)
        require(
            weights.get("group_size") == args.expected_group_size,
            f"weight group_size is not {args.expected_group_size}",
            failures,
        )
        require("lm_head" in ignore, "lm_head is not listed in quantization_config.ignore", failures)
        require("bitsandbytes" not in json.dumps(qcfg).lower(), "artifact mentions bitsandbytes", failures)

    run_dir = Path(args.run_dir) if args.run_dir else None
    config_path = Path(args.config) if args.config else (run_dir / "config.env" if run_dir else None)
    log_path = Path(args.server_log) if args.server_log else (run_dir / "server.log" if run_dir else None)

    env = read_env(config_path) if config_path else {}
    if config_path:
        require(config_path.exists(), f"missing run config: {config_path}", failures)
        require(env.get("QUANTIZATION") == "compressed-tensors", "run QUANTIZATION is not compressed-tensors", failures)
        require(env.get("KV_CACHE_DTYPE", "") in ("", "auto"), "KV_CACHE_DTYPE must not enable FP8 for W4 runs", failures)
        require(env.get("LINEAR_BACKEND") in ("machete", "marlin"), "LINEAR_BACKEND is not forced to machete or marlin", failures)
        joined = "\n".join(f"{k}={v}" for k, v in env.items()).lower()
        require("bitsandbytes" not in joined, "run config mentions bitsandbytes", failures)
        require("fp8_per_tensor" not in joined and "quantization=fp8" not in joined, "run config combines W4 with FP8 quantization", failures)

    backend = None
    if log_path:
        require(log_path.exists(), f"missing server log: {log_path}", failures)
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            match = re.search(r"Using\s+([A-Za-z0-9_:.]+)\s+for\s+CompressedTensorsWNA16", text)
            if match:
                backend = match.group(1)
            require(backend is not None, "server log does not confirm CompressedTensorsWNA16 backend selection", failures)
            if backend:
                require(re.search(r"(machete|marlin)", backend, re.IGNORECASE) is not None, f"backend is not Machete/Marlin: {backend}", failures)
            require("bitsandbytes" not in text.lower(), "server log mentions bitsandbytes", failures)

    result = {
        "artifact": str(artifact),
        "run_dir": str(run_dir) if run_dir else None,
        "backend": backend,
        "valid": not failures,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
