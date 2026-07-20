#!/usr/bin/env python3
"""Validate submission docker-compose flags for the known candidate types.

This is intentionally dependency-free: it scans the compose text for command
tokens instead of requiring PyYAML in the submission environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def require(ok: bool, msg: str, failures: list[str]) -> None:
    if not ok:
        failures.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("compose")
    ap.add_argument("--candidate", choices=["fp8", "w4a16-gptq"], required=True)
    ap.add_argument("--allow-placeholder-image", action="store_true")
    args = ap.parse_args()

    text = Path(args.compose).read_text()
    failures: list[str] = []

    common_required = [
        "vllm.entrypoints.openai.api_server",
        "--model=/model",
        "--served-model-name=LFM2.5-1.2B-Instruct",
        "--host=0.0.0.0",
        "--port=8000",
        "--dtype=bfloat16",
        "--max-model-len=5120",
        "--gpu-memory-utilization=0.70",
        "--tensor-parallel-size=1",
        "--max-num-seqs=8",
        "--max-num-batched-tokens=512",
        "--enable-prefix-caching",
        "--prefix-caching-hash-algo=xxhash",
        "--disable-log-stats",
        "--disable-uvicorn-access-log",
    ]
    for token in common_required:
        require(token in text, f"missing required token: {token}", failures)

    require("bitsandbytes" not in text.lower(), "compose mentions BitsAndBytes", failures)
    if not args.allow_placeholder_image:
        require("<DOCKERHUB_USER>" not in text, "compose still has placeholder image tag", failures)

    if args.candidate == "fp8":
        require("--quantization=fp8_per_tensor" in text, "FP8 compose missing fp8_per_tensor quantization", failures)
        require("--quantization=fp8\n" not in text, "FP8 compose uses legacy fp8 quantization", failures)
        require("--linear-backend=" not in text, "FP8 compose should not force W4 linear backend", failures)
    else:
        require("--quantization=compressed-tensors" in text, "W4 compose missing compressed-tensors quantization", failures)
        require("--linear-backend=machete" in text or "--linear-backend=marlin" in text, "W4 compose missing Machete/Marlin linear backend", failures)
        require("fp8_per_tensor" not in text, "W4 compose mixes fp8_per_tensor with W4", failures)
        require("--kv-cache-dtype=fp8" not in text and "--kv-cache-dtype=fp8_e5m2" not in text, "W4 compose enables FP8 KV cache", failures)

    if failures:
        print("INVALID")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("VALID")


if __name__ == "__main__":
    main()
