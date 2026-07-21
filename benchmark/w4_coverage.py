#!/usr/bin/env python3
"""Report W4-vs-BF16 Linear parameter/GEMM coverage for an exported artifact."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open


def module_from_key(key: str, suffix: str) -> str:
    if not key.endswith(suffix):
        raise ValueError(key)
    return key[: -len(suffix)]


def module_bucket(name: str) -> str:
    if name == "lm_head":
        return "lm_head"
    if ".conv." in name:
        return "short_conv_linear"
    if ".self_attn." in name:
        return "attention"
    if ".feed_forward." in name:
        return "mlp"
    return "other_linear"


def layer_index(name: str) -> int | None:
    m = re.search(r"\bmodel\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def shape_numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    artifact = Path(args.artifact)
    st_path = artifact / "model.safetensors"
    cfg_path = artifact / "config.json"
    if not st_path.exists():
        raise SystemExit(f"missing {st_path}")
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")

    config = json.loads(cfg_path.read_text())
    qcfg = config.get("quantization_config") or {}
    ignore = set(qcfg.get("ignore") or [])

    modules: dict[str, dict[str, Any]] = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        for key in keys:
            if key.endswith(".weight_shape"):
                mod = module_from_key(key, ".weight_shape")
                shape = [int(x) for x in f.get_tensor(key).tolist()]
                modules[mod] = {
                    "module": mod,
                    "dtype": "w4",
                    "shape": shape,
                    "params": shape_numel(shape),
                    "bucket": module_bucket(mod),
                    "layer": layer_index(mod),
                    "ignored": mod in ignore,
                }

        for key in keys:
            if not key.endswith(".weight"):
                continue
            mod = module_from_key(key, ".weight")
            if mod in modules:
                continue
            if mod.endswith("embed_tokens") or ".embed_tokens" in mod:
                continue
            tensor = f.get_tensor(key)
            shape = [int(x) for x in tensor.shape]
            # Count matrix weights as Linear GEMMs. Skip vector norms and Conv1d
            # kernels; conv.in_proj/out_proj remain 2D and are reported.
            if len(shape) != 2:
                continue
            modules[mod] = {
                "module": mod,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": shape,
                "params": shape_numel(shape),
                "bucket": module_bucket(mod),
                "layer": layer_index(mod),
                "ignored": mod in ignore,
            }

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        total_modules = len(selected)
        total_params = sum(m["params"] for m in selected)
        q_modules = [m for m in selected if m["dtype"] == "w4"]
        q_params = sum(m["params"] for m in q_modules)
        by_bucket: dict[str, Any] = {}
        for bucket in sorted({m["bucket"] for m in selected}):
            rows = [m for m in selected if m["bucket"] == bucket]
            q_rows = [m for m in rows if m["dtype"] == "w4"]
            params = sum(m["params"] for m in rows)
            q_params_bucket = sum(m["params"] for m in q_rows)
            by_bucket[bucket] = {
                "modules": len(rows),
                "quantized_modules": len(q_rows),
                "module_coverage": len(q_rows) / len(rows) if rows else 0.0,
                "params": params,
                "quantized_params": q_params_bucket,
                "param_coverage": q_params_bucket / params if params else 0.0,
            }
        return {
            "modules": total_modules,
            "quantized_modules": len(q_modules),
            "module_coverage": len(q_modules) / total_modules if total_modules else 0.0,
            "params": total_params,
            "quantized_params": q_params,
            "param_coverage": q_params / total_params if total_params else 0.0,
            "by_bucket": by_bucket,
        }

    all_modules = sorted(modules.values(), key=lambda x: x["module"])
    core_modules = [
        m
        for m in all_modules
        if m["bucket"] in {"attention", "mlp"}
    ]
    result = {
        "artifact": str(artifact),
        "quantization": {
            "method": qcfg.get("quant_method"),
            "format": qcfg.get("format"),
            "ignore": sorted(ignore),
        },
        "all_linear_2d": summarize(all_modules),
        "core_attention_mlp": summarize(core_modules),
        "bf16_modules": [m for m in all_modules if m["dtype"] != "w4"],
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n")


if __name__ == "__main__":
    main()
