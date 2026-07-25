#!/usr/bin/env python3
"""Backport vLLM PR #48917 (merged 2026-07-21): thread quant_config through
LFM2's ShortConv projections.

Bug (present in vllm/vllm-openai:v0.22.1 AND :v0.25.1, verified against both
source trees): Lfm2ShortConvDecoderLayer builds ShortConv() without passing
quant_config, and ShortConv.__init__ does not even accept a quant_config
parameter -- so ShortConv.in_proj / ShortConv.out_proj are always built with
the default (unquantized) linear method, regardless of --quantization. In
LFM2.5-1.2B (hidden_size=2048, 10 ShortConv layers) this silently leaves
~168M params (~14% of the model) in BF16 under --quantization=fp8_per_tensor,
even though Lfm2MLP and the attention projections right next to it ARE
quantized (they DO receive quant_config).

This script edits the installed site-packages files directly (same approach
as apply_vllm_decode_metadata_fastpath.py) so the base image's compiled CUDA
extensions/kernels are untouched -- only Python wiring changes. Idempotent:
safe to re-run / safe under Docker layer caching.
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"missing expected source block for {label}")
    return text.replace(old, new, 1)


def patch_short_conv(purelib: Path) -> None:
    target = purelib / "vllm" / "model_executor" / "layers" / "mamba" / "short_conv.py"
    if not target.exists():
        raise SystemExit(f"target file not found: {target}")

    text = target.read_text()
    if "quant_config: QuantizationConfig | None = None" in text:
        print(f"ShortConv quant_config already applied: {target}")
        return

    text = replace_once(
        text,
        "from vllm.model_executor.layers.mamba.abstract import MambaBase\n",
        "from vllm.model_executor.layers.mamba.abstract import MambaBase\n"
        "from vllm.model_executor.layers.quantization import QuantizationConfig\n",
        "QuantizationConfig import",
    )

    text = replace_once(
        text,
        """    def __init__(
        self,
        config,
        dim: int,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):""",
        """    def __init__(
        self,
        config,
        dim: int,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):""",
        "ShortConv.__init__ signature",
    )

    text = replace_once(
        text,
        """        self.in_proj = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim] * 3,
            bias=self.bias,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.out_proj",
        )""",
        """        self.in_proj = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim] * 3,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )""",
        "in_proj/out_proj quant_config",
    )

    target.write_text(text)
    print(f"applied ShortConv quant_config wiring: {target}")


def patch_lfm2_model(purelib: Path, relpath: str, label: str, dim_expr: str) -> None:
    target = purelib / relpath
    if not target.exists():
        print(f"skip (not present in this image): {target}")
        return

    text = target.read_text()
    if (
        "self.short_conv = ShortConv(\n"
        "            config=config,\n"
        f"            dim={dim_expr},\n"
        "            layer_idx=layer_idx,\n"
        "            model_config=model_config,\n"
        "            cache_config=cache_config,\n"
        "            quant_config=quant_config,\n"
    ) in text:
        print(f"{label} quant_config already applied: {target}")
        return

    # dim= varies by model (lfm2.py: config.conv_dim; lfm2_moe.py:
    # config.hidden_size) -- match on the common prefix/suffix instead.
    old_block = (
        "        self.short_conv = ShortConv(\n"
        "            config=config,\n"
        f"            dim={dim_expr},\n"
        "            layer_idx=layer_idx,\n"
        "            model_config=model_config,\n"
        "            cache_config=cache_config,\n"
        '            prefix=f"{prefix}.conv",\n'
        "        )"
    )
    new_block = (
        "        self.short_conv = ShortConv(\n"
        "            config=config,\n"
        f"            dim={dim_expr},\n"
        "            layer_idx=layer_idx,\n"
        "            model_config=model_config,\n"
        "            cache_config=cache_config,\n"
        "            quant_config=quant_config,\n"
        '            prefix=f"{prefix}.conv",\n'
        "        )"
    )
    if old_block not in text:
        print(f"WARNING: {label} ShortConv(...) call site not found in expected form, skipping: {target}")
        return
    text = text.replace(old_block, new_block, 1)

    target.write_text(text)
    print(f"applied {label} -> ShortConv quant_config forwarding: {target}")


def main() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    patch_short_conv(purelib)
    patch_lfm2_model(purelib, "vllm/model_executor/models/lfm2.py", "lfm2.py", "config.conv_dim")
    # lfm2_moe.py has the same ShortConv(...) call-site pattern in some vLLM
    # versions (dim=config.hidden_size there); patch it too if present so MoE
    # variants aren't silently left unpatched. Not required for
    # LFM2.5-1.2B-Instruct (dense model).
    patch_lfm2_model(purelib, "vllm/model_executor/models/lfm2_moe.py", "lfm2_moe.py", "config.hidden_size")


if __name__ == "__main__":
    main()
