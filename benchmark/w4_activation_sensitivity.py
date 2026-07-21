#!/usr/bin/env python3
"""Rank BF16-vs-W4 module activation differences on chat calibration prompts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.quantize_w4a16_gptq import build_calibration_dataset


def tensor_out(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0].detach()
    return None


def target_modules(model: torch.nn.Module) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name == "lm_head" or ".conv." in name:
            continue
        if any(name.endswith(s) for s in ("q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3")):
            names.append(name)
    return names


def capture_outputs(
    model: torch.nn.Module, module_names: list[str], input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    handles = []
    modules = dict(model.named_modules())

    for name in module_names:
        def hook(_module, _inp, out, *, key=name):
            t = tensor_out(out)
            if t is not None:
                outputs[key] = t.float().cpu()

        handles.append(modules[name].register_forward_hook(hook))
    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    for handle in handles:
        handle.remove()
    return outputs


def update_stats(stats: dict[str, dict[str, float]], base: dict[str, torch.Tensor], cand: dict[str, torch.Tensor]) -> None:
    for name, b in base.items():
        c = cand.get(name)
        if c is None or c.shape != b.shape:
            continue
        diff = b - c
        s = stats[name]
        s["samples"] += 1
        s["elements"] += b.numel()
        s["sqerr"] += float(torch.sum(diff * diff).item())
        s["base_sq"] += float(torch.sum(b * b).item())
        s["cand_sq"] += float(torch.sum(c * c).item())
        s["dot"] += float(torch.sum(b * c).item())
        s["max_abs"] = max(s["max_abs"], float(torch.max(torch.abs(diff)).item()))


def layer_of(name: str) -> int:
    parts = name.split(".")
    try:
        return int(parts[2])
    except Exception:
        return -1


def kind_of(name: str) -> str:
    if ".self_attn." in name:
        return "attn." + name.rsplit(".", 1)[-1]
    if ".feed_forward." in name:
        return "ffn." + name.rsplit(".", 1)[-1]
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--candidate-model", required=True)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json-out", required=True, type=Path)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.candidate_model, local_files_only=True)
    dataset = build_calibration_dataset(tokenizer, args.num_samples, args.max_seq_length)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.baseline_model, local_files_only=True, torch_dtype=dtype, trust_remote_code=False
    ).to(args.device)
    cand = AutoModelForCausalLM.from_pretrained(
        args.candidate_model, local_files_only=True, torch_dtype=dtype, trust_remote_code=False
    ).to(args.device)
    base.eval()
    cand.eval()

    names = target_modules(base)
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"samples": 0, "elements": 0, "sqerr": 0.0, "base_sq": 0.0, "cand_sq": 0.0, "dot": 0.0, "max_abs": 0.0}
    )

    for i, row in enumerate(dataset):
        toks = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_length,
            add_special_tokens=False,
        )
        input_ids = toks["input_ids"].to(args.device)
        attention_mask = toks["attention_mask"].to(args.device)
        b = capture_outputs(base, names, input_ids, attention_mask)
        c = capture_outputs(cand, names, input_ids, attention_mask)
        update_stats(stats, b, c)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[{i + 1}/{len(dataset)}] captured {len(b)} modules", flush=True)

    rows = []
    for name, s in stats.items():
        mse = s["sqerr"] / max(1.0, s["elements"])
        rel_mse = s["sqerr"] / max(1e-12, s["base_sq"])
        cos = s["dot"] / max(1e-12, math.sqrt(s["base_sq"] * s["cand_sq"]))
        rows.append(
            {
                "module": name,
                "layer": layer_of(name),
                "kind": kind_of(name),
                "samples": int(s["samples"]),
                "elements": int(s["elements"]),
                "mse": mse,
                "relative_mse": rel_mse,
                "cosine": cos,
                "max_abs": s["max_abs"],
            }
        )
    rows.sort(key=lambda r: (r["relative_mse"], r["mse"]), reverse=True)
    summary = {
        "baseline_model": args.baseline_model,
        "candidate_model": args.candidate_model,
        "num_samples": args.num_samples,
        "max_seq_length": args.max_seq_length,
        "ranked_modules": rows,
        "by_kind": [],
        "by_layer": [],
    }
    for key_name, key_func in (("by_kind", lambda r: r["kind"]), ("by_layer", lambda r: str(r["layer"]))):
        grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"sqerr": 0.0, "base_sq": 0.0, "modules": 0})
        for r in rows:
            # Reconstruct enough aggregate from normalized fields for ranking.
            g = grouped[key_func(r)]
            g["sqerr"] += r["relative_mse"]
            g["base_sq"] += 1.0
            g["modules"] += 1
        summary[key_name] = [
            {"group": k, "mean_relative_mse": v["sqerr"] / max(1.0, v["base_sq"]), "modules": int(v["modules"])}
            for k, v in sorted(grouped.items(), key=lambda kv: kv[1]["sqerr"] / max(1.0, kv[1]["base_sq"]), reverse=True)
        ]

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.json_out}")
    print("Top modules:")
    for r in rows[:20]:
        print(f"  {r['module']:45s} rel_mse={r['relative_mse']:.6g} cos={r['cosine']:.6f}")


if __name__ == "__main__":
    main()
