#!/usr/bin/env python3
"""Patch vLLM LFM2 to use a runtime FP8 copy for the tied LM head.

LFM2 ties `lm_head.weight` to `model.embed_tokens.weight` in vLLM. That keeps
embedding lookup precise, but it also means the logits projection uses the BF16
embedding weight even when the dense model is running online FP8. This patch
unties only at runtime after checkpoint loading:

* `model.embed_tokens.weight` remains BF16 and keeps serving input embedding.
* `lm_head.weight` becomes a separate FP8 per-tensor copy with dynamic FP8
  activations, using vLLM's existing FP8 linear kernel path.

The patch is gated by `LFM2_FP8_LM_HEAD=1` so the image can be A/B tested without
rebuilding.
"""

from __future__ import annotations

import sysconfig
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"missing expected source block for {label}")
    return text.replace(old, new, 1)


def main() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    target = purelib / "vllm" / "model_executor" / "models" / "lfm2.py"
    if not target.exists():
        raise SystemExit(f"target file not found: {target}")

    text = target.read_text()
    if "class RuntimeFp8LMHead" in text:
        print(f"LFM2 FP8 lm_head patch already applied: {target}")
        return

    text = replace_once(
        text,
        """from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn as nn
from transformers import Lfm2Config
""",
        """from collections.abc import Iterable
from itertools import islice
import os

import torch
import torch.nn as nn
from transformers import Lfm2Config
""",
        "os import",
    )

    text = replace_once(
        text,
        """from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
""",
        """from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
""",
        "fp8 lm_head imports",
    )

    text = replace_once(
        text,
        """
class Lfm2ForCausalLM(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid, SupportsQuant
):
""",
        """
class RuntimeFp8LMHeadMethod:
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1]).contiguous()
        x_q, x_scale = ops.scaled_fp8_quant(x_2d, scale=None)
        logits = torch._scaled_mm(
            x_q,
            layer.weight.t(),
            scale_a=x_scale,
            scale_b=layer.weight_scale,
            out_dtype=x.dtype,
        )
        if bias is not None:
            logits = logits + bias
        return logits.view(*x.shape[:-1], layer.weight.shape[0])


class RuntimeFp8LMHead(nn.Module):
    def __init__(self, embed_tokens: VocabParallelEmbedding):
        super().__init__()
        self.tp_size = embed_tokens.tp_size
        self.shard_indices = embed_tokens.shard_indices
        self.embedding_dim = embed_tokens.embedding_dim
        self.num_embeddings = embed_tokens.num_embeddings
        self.org_vocab_size = embed_tokens.org_vocab_size
        self.org_vocab_size_padded = embed_tokens.org_vocab_size_padded
        self.num_embeddings_padded = embed_tokens.num_embeddings_padded
        self.num_added_embeddings = embed_tokens.num_added_embeddings
        self.num_embeddings_per_partition = (
            embed_tokens.num_embeddings_per_partition
        )
        self.num_org_embeddings_per_partition = (
            embed_tokens.num_org_embeddings_per_partition
        )
        self.num_added_embeddings_per_partition = (
            embed_tokens.num_added_embeddings_per_partition
        )
        self.weight_block_size = None
        self.bias = None

        # Keep input embeddings tied to the BF16 checkpoint tensor; only logits
        # use this cloned weight, which is immediately converted to FP8.
        qweight, weight_scale = ops.scaled_fp8_quant(
            embed_tokens.weight.detach().contiguous(), scale=None
        )
        self.weight = nn.Parameter(qweight.contiguous(), requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        self.input_scale = None
        self.input_scale_ub = None
        self.quant_method = RuntimeFp8LMHeadMethod()
        set_weight_attrs(self.weight, {"input_dim": 1, "output_dim": 0})


class Lfm2ForCausalLM(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid, SupportsQuant
):
""",
        "runtime fp8 lm_head class",
    )

    text = replace_once(
        text,
        """    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
""",
        """    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        loaded = loader.load_weights(weights)
        if (
            os.environ.get("LFM2_FP8_LM_HEAD", "0") == "1"
            and get_pp_group().is_last_rank
            and isinstance(self.model.embed_tokens, VocabParallelEmbedding)
            and not isinstance(self.lm_head, RuntimeFp8LMHead)
        ):
            self.lm_head = RuntimeFp8LMHead(self.model.embed_tokens)
            print(
                "LFM2_FP8_LM_HEAD enabled: "
                f"embed_dtype={self.model.embed_tokens.weight.dtype} "
                f"lm_head_type={type(self.lm_head).__name__} "
                f"lm_head_weight_dtype={self.lm_head.weight.dtype} "
                f"quant_method={type(self.lm_head.quant_method).__name__} "
                "tied_weight_object="
                f"{self.lm_head.weight is self.model.embed_tokens.weight}",
                flush=True,
            )
        return loaded
""",
        "load_weights fp8 lm_head hook",
    )

    target.write_text(text)
    print(f"applied LFM2 FP8 lm_head patch: {target}")


if __name__ == "__main__":
    main()
