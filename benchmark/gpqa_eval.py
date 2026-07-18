#!/usr/bin/env python3
"""Real(ish) GPQA-diamond accuracy check against a running vLLM server.

The official GPQA-diamond dataset (Idavidrein/gpqa) is HF-gated and requires
manually accepting terms on the dataset page -- not something a token alone
can bypass (see EXPERIMENTS.md). This script instead uses
`hendrydong/gpqa_diamond_mc`, an ungated re-upload of the same 198 diamond
questions already formatted as 4-way multiple choice with the answer as
\\boxed{LETTER} -- as close to the real gate as this box can get without the
official dataset's access being approved.

Sends each question as a single chat turn (zero-shot, matching
gpqa_diamond_zeroshot's setup), parses the model's \\boxed{X} answer, and
reports accuracy vs the same metric run against another server (e.g. BF16
vs FP8) for a direct accuracy-delta comparison.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time

import aiohttp


def extract_boxed_letter(text: str) -> str | None:
    m = re.search(r"\\boxed\{([A-D])\}", text)
    if m:
        return m.group(1)
    # fallback: look for a bare "answer is X" / lone letter near the end
    m = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", text.strip()[-40:])
    return m.group(1) if m else None


async def ask_one(session, base_url, model, problem, sem, timeout_s):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": 3072,
        "temperature": 0.0,
    }
    async with sem:
        try:
            async with session.post(
                f"{base_url}/v1/chat/completions", json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                if resp.status != 200:
                    return {"status": f"http_{resp.status}", "text": None}
                data = await resp.json()
                text = data["choices"][0]["message"]["content"]
                return {"status": "ok", "text": text}
        except Exception as e:
            return {"status": f"error:{e}", "text": None}


async def run(args):
    from datasets import load_dataset

    ds = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    sem = asyncio.Semaphore(args.num_concurrent)
    results = []
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            ask_one(session, args.base_url, args.model, row["problem"], sem, args.timeout)
            for row in ds
        ]
        n_done = 0
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            n_done += 1
            if args.verbose and n_done % 20 == 0:
                print(f"  [{n_done}/{len(ds)}]", file=sys.stderr)

    # re-associate isn't needed for aggregate accuracy since we only need counts,
    # but to compute correctness we must map back to expected answers in order.
    # asyncio.as_completed loses order, so redo with gather for correctness mapping.
    return ds, results


async def run_ordered(args):
    from datasets import load_dataset

    ds = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    sem = asyncio.Semaphore(args.num_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            ask_one(session, args.base_url, args.model, row["problem"], sem, args.timeout)
            for row in ds
        ]
        results = []
        n_done = 0
        for i in range(0, len(tasks), args.num_concurrent):
            batch = tasks[i:i + args.num_concurrent]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            n_done += len(batch_results)
            if args.verbose:
                print(f"  [{n_done}/{len(ds)}]", file=sys.stderr)
    return ds, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="0 = full 198")
    ap.add_argument("--num-concurrent", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--output", default=None, help="write per-question JSONL here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ds, results = asyncio.run(run_ordered(args))
    duration = time.time() - t0

    n_correct = 0
    n_error = 0
    n_unparsed = 0
    records = []
    for row, r in zip(ds, results):
        expected = extract_boxed_letter(row["solution"])
        got = extract_boxed_letter(r["text"]) if r["text"] else None
        correct = (expected is not None and got == expected)
        if r["status"] != "ok":
            n_error += 1
        elif got is None:
            n_unparsed += 1
        if correct:
            n_correct += 1
        records.append({
            "domain": row["domain"], "expected": expected, "got": got,
            "correct": correct, "status": r["status"],
        })

    n = len(ds)
    acc = n_correct / n if n else 0.0
    print(f"GPQA-diamond-mc (ungated mirror, n={n}): accuracy = {acc:.4f} ({n_correct}/{n})")
    print(f"  errors: {n_error}, unparsed answers: {n_unparsed}, duration: {duration:.1f}s")

    if args.output:
        with open(args.output, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Per-question detail written to {args.output}")


if __name__ == "__main__":
    main()
