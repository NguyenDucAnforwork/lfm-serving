#!/usr/bin/env python3
"""Append one row to EXPERIMENTS.md's summary table from a results dir."""
import argparse
import json
from pathlib import Path


def read_env(path: Path) -> dict:
    d = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        d[k] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--decision", default="")
    ap.add_argument("--experiments-md", default="/workspace/lfm-serving/EXPERIMENTS.md")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = read_env(run_dir / "config.env")
    summary = json.loads((run_dir / "summary.json").read_text())
    vram_peak_path = run_dir / "vram_peak_mib.txt"
    vram_peak = vram_peak_path.read_text().strip() if vram_peak_path.exists() else "?"

    def fmt(x, nd=1):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"

    row = "| {run} | {maxlen} | {util} | {seqs} | {batchtok} | {pfx} | {chunk} | {vram} | {ers} | {ttft_m}/{ttft_p} | {tpot_m}/{tpot_p} | {err} | {decision} |".format(
        run=run_dir.name,
        maxlen=cfg.get("MAX_MODEL_LEN", "?"),
        util=cfg.get("GPU_MEM_UTIL", "?"),
        seqs=cfg.get("MAX_NUM_SEQS", "?"),
        batchtok=cfg.get("MAX_NUM_BATCHED_TOKENS") or "default",
        pfx="on" if cfg.get("ENABLE_PREFIX_CACHING") == "1" else "off",
        chunk={"1": "on", "0": "off", "": "default"}.get(cfg.get("ENABLE_CHUNKED_PREFILL", ""), "default"),
        vram=vram_peak,
        ers=fmt(summary.get("ers_mean_pct"), 2),
        ttft_m=fmt(summary.get("ttft_ms_mean")),
        ttft_p=fmt(summary.get("ttft_ms_p95")),
        tpot_m=fmt(summary.get("tpot_ms_mean"), 2),
        tpot_p=fmt(summary.get("tpot_ms_p95"), 2),
        err=fmt((summary.get("error_rate") or 0) * 100, 1) + "%",
        decision=args.decision,
    )

    md_path = Path(args.experiments_md)
    text = md_path.read_text()
    marker = "<!-- rows appended below by scripts/run_experiment.sh workflow -->"
    text = text.replace(marker, row + "\n" + marker)
    md_path.write_text(text)
    print(row)


if __name__ == "__main__":
    main()
