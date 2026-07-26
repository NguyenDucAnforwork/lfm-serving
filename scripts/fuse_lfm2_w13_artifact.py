#!/usr/bin/env python3
"""Fuse LFM2 feed-forward w1/w3 tensors into vLLM's runtime w13 layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def rewrite_quantization_config(
    config: dict, keep_lm_head_quantized: bool, w4_targets: str
) -> None:
    quant_config = config.get("quantization_config")
    if not quant_config:
        return

    groups = quant_config.get("config_groups", {})
    if w4_targets == "w13-only":
        if "group_0" in groups:
            groups["group_0"]["targets"] = [
                r"re:model\.layers\.[0-9]\.feed_forward\.w13$"
            ]
        if "group_1" in groups:
            groups["group_1"]["targets"] = [
                r"re:model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$",
                r"re:model\.layers\.[0-9]\.feed_forward\.w2$",
                r"re:model\.layers\.(1[0-5])\.feed_forward\.(w13|w2)$",
            ]
            if keep_lm_head_quantized:
                groups["group_1"]["targets"].append("lm_head")
        return

    for group in groups.values():
        targets = group.get("targets", [])
        rewritten = []
        for target in targets:
            if not keep_lm_head_quantized and target == "lm_head":
                continue
            if w4_targets == "w13-only":
                target = target.replace(
                    r"model\\.layers\\.[0-9]\\.feed_forward\\.(w1|w2|w3)$",
                    r"model\\.layers\\.[0-9]\\.feed_forward\\.w13$",
                )
                target = target.replace(
                    r"model\\.layers\\.\\d+\\.feed_forward\\.(w1|w2|w3)$",
                    r"model\\.layers\\.\\d+\\.feed_forward\\.w2$",
                )
                target = target.replace(
                    r"model\\.layers\\.(1[0-5])\\.feed_forward\\.(w1|w2|w3)$",
                    r"model\\.layers\\.(1[0-5])\\.feed_forward\\.(w13|w2)$",
                )
            else:
                target = target.replace(
                    r"feed_forward\.(w1|w2|w3)$", r"feed_forward\.(w13|w2)$"
                )
            rewritten.append(target)
        group["targets"] = rewritten

    if w4_targets == "w13-only":
        groups.get("group_1", {}).setdefault("targets", []).append(
            r"re:model\.layers\.[0-9]\.feed_forward\.w2$"
        )


def pack_signed_int4_to_uint4b8(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.int8:
        raise TypeError(f"expected int8 unpacked int4 tensor, got {weight.dtype}")
    if weight.shape[-1] % 8 != 0:
        raise ValueError(f"last dimension must be divisible by 8, got {tuple(weight.shape)}")

    unsigned = (weight.to(torch.int32) + 8) & 0xF
    unsigned = unsigned.reshape(*weight.shape[:-1], weight.shape[-1] // 8, 8)
    packed = torch.zeros(unsigned.shape[:-1], dtype=torch.int32)
    for idx in range(8):
        packed |= unsigned[..., idx] << (4 * idx)
    return packed.contiguous()


def quantize_fp8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 448.0
    qweight = (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return qweight.contiguous(), scale.to(torch.bfloat16).contiguous()


def fuse_tensors(
    src: Path,
    dst: Path,
    bf16_lm_head_from: Path | None = None,
    early_w2_fp8_from: Path | None = None,
    w4_targets: str = "w13-w2",
) -> None:
    if dst.exists():
        raise FileExistsError(f"output already exists: {dst}")
    dst.mkdir(parents=True)

    for path in src.iterdir():
        if path.name == "model.safetensors":
            continue
        if path.name == "config.json":
            continue
        target = dst / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)

    tensors: dict[str, torch.Tensor] = {}
    metadata = None
    fused_layers: set[int] = set()

    with safe_open(src / "model.safetensors", framework="pt", device="cpu") as f:
        metadata = f.metadata()
        keys = list(f.keys())

        skip: set[str] = set()
        for key in keys:
            if ".feed_forward.w1." not in key:
                continue

            w1_key = key
            w3_key = key.replace(".feed_forward.w1.", ".feed_forward.w3.", 1)
            if w3_key not in keys:
                raise KeyError(f"missing matching w3 tensor for {w1_key}: expected {w3_key}")

            w1 = f.get_tensor(w1_key)
            w3 = f.get_tensor(w3_key)
            if w1.ndim == 0 or w3.ndim == 0:
                raise ValueError(f"cannot infer output dimension for scalar tensors: {w1_key}, {w3_key}")
            if w1.shape[1:] != w3.shape[1:]:
                raise ValueError(f"non-concat dimensions differ: {w1_key} {tuple(w1.shape)} vs {tuple(w3.shape)}")

            w13_key = w1_key.replace(".feed_forward.w1.", ".feed_forward.w13.", 1)
            tensors[w13_key] = torch.cat([w1, w3], dim=0).contiguous()
            skip.add(w1_key)
            skip.add(w3_key)

            try:
                layer = int(w1_key.split(".feed_forward.", 1)[0].rsplit(".", 1)[1])
                fused_layers.add(layer)
            except (IndexError, ValueError):
                pass

        for key in keys:
            if key in skip:
                continue
            tensors[key] = f.get_tensor(key)

    config = json.loads((src / "config.json").read_text())
    rewrite_quantization_config(
        config,
        keep_lm_head_quantized=bf16_lm_head_from is None,
        w4_targets=w4_targets,
    )

    # vLLM's CompressedTensorsW4A8Fp8 scheme registers W4 parameters as
    # weight_packed + weight_scale + weight_shape. llm-compressor serializes
    # the packed payload under weight, so adapt only the INT4 group targets.
    w4_layer_prefixes = {
        f"model.layers.{idx}.feed_forward.{proj}"
        for idx in range(10)
        for proj in (("w13",) if w4_targets == "w13-only" else ("w13", "w2"))
    }
    for prefix in sorted(w4_layer_prefixes):
        weight_key = f"{prefix}.weight"
        if weight_key not in tensors:
            continue
        weight = tensors.pop(weight_key)
        tensors[f"{prefix}.weight_packed"] = pack_signed_int4_to_uint4b8(weight)
        tensors[f"{prefix}.weight_shape"] = torch.tensor(
            list(weight.shape), dtype=torch.int64
        )

    if bf16_lm_head_from is not None:
        tensors.pop("lm_head.weight_scale", None)
        with safe_open(bf16_lm_head_from, framework="pt", device="cpu") as f:
            lm_head_key = (
                "lm_head.weight" if "lm_head.weight" in f.keys() else "model.embed_tokens.weight"
            )
            tensors["lm_head.weight"] = f.get_tensor(lm_head_key)

    if early_w2_fp8_from is not None:
        with safe_open(early_w2_fp8_from, framework="pt", device="cpu") as f:
            for idx in range(10):
                prefix = f"model.layers.{idx}.feed_forward.w2"
                for suffix in ("weight_packed", "weight_shape", "weight_zero_point", "weight_g_idx"):
                    tensors.pop(f"{prefix}.{suffix}", None)
                weight = f.get_tensor(f"{prefix}.weight")
                qweight, scale = quantize_fp8_per_channel(weight)
                tensors[f"{prefix}.weight"] = qweight
                tensors[f"{prefix}.weight_scale"] = scale

    save_file(tensors, dst / "model.safetensors", metadata=metadata)

    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"wrote {dst}")
    print(f"fused layers: {sorted(fused_layers)}")
    print(f"tensors: {len(tensors)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--bf16-lm-head-from", type=Path)
    parser.add_argument("--early-w2-fp8-from", type=Path)
    parser.add_argument("--w4-targets", choices=["w13-w2", "w13-only"], default="w13-w2")
    args = parser.parse_args()

    fuse_tensors(
        args.src,
        args.dst,
        bf16_lm_head_from=args.bf16_lm_head_from,
        early_w2_fp8_from=args.early_w2_fp8_from,
        w4_targets=args.w4_targets,
    )


if __name__ == "__main__":
    main()
