#!/usr/bin/env python3
"""Check CUDA GPU compute capability for hardware-specific experiments."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-major", type=int, required=True)
    ap.add_argument("--min-minor", type=int, default=0)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    result = {
        "ok": False,
        "required": {"major": args.min_major, "minor": args.min_minor},
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "error": None,
    }

    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        result["device_count"] = int(torch.cuda.device_count()) if result["cuda_available"] else 0
        for idx in range(result["device_count"]):
            major, minor = torch.cuda.get_device_capability(idx)
            result["devices"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "major": major,
                    "minor": minor,
                }
            )
        result["ok"] = any(
            (d["major"], d["minor"]) >= (args.min_major, args.min_minor)
            for d in result["devices"]
        )
    except Exception as exc:  # pragma: no cover - diagnostic script
        result["error"] = repr(exc)

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")
    print(text)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
