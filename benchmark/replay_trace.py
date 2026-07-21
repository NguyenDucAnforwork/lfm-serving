#!/usr/bin/env python3
"""Replay trace_grading_public.jsonl against a running vLLM OpenAI server.

Follows conv_id / turn_idx / timestamp_ms / think_ms exactly as scheduled in
the trace:
  * turn_idx==0 of each conversation fires at (benchmark_start + timestamp_ms)
  * turn_idx>0 fires at (completion time of the previous turn in that
    conversation) + think_ms
  * conversations run fully concurrently (independent asyncio tasks); no
    client-side concurrency cap is imposed -- realistic overlap comes purely
    from Poisson arrivals + think_ms, and the server's own queuing shows up
    as added latency/TTFT, which is exactly what should be measured.

Sends streaming Chat Completions requests, measuring TTFT / TPOT / latency /
errors per request, and writes per-request JSONL + CSV results plus a summary
(with ERS via compute_ers.py). Warmup requests (in_warmup==true) are sent
(to warm the server/prefix cache) but excluded from the final ERS.

Synthetic prompt content is generated from trace token-length metadata only
(see trace_utils.py) -- the trace itself carries no real prompt text.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent))
from trace_utils import PromptBuilder, group_by_conv, load_trace  # noqa: E402
import compute_ers  # noqa: E402


def build_messages(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


async def send_turn(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    ignore_eos: bool,
    timeout_s: float,
    meta: dict,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    t_send = time.perf_counter()
    ttft = None
    token_times: list[float] = []
    content_parts: list[str] = []
    usage = None
    status = "ok"
    error = None

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            if resp.status != 200:
                status = "error"
                body = await resp.text()
                error = f"http_{resp.status}:{body[:200]}"
            else:
                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {}) or {}
                        piece = delta.get("content") or ""
                        if piece:
                            now = time.perf_counter()
                            if ttft is None:
                                ttft = now - t_send
                            token_times.append(now)
                            content_parts.append(piece)
                    if obj.get("usage"):
                        usage = obj["usage"]
    except asyncio.TimeoutError:
        status = "timeout"
        error = f"timeout after {timeout_s}s"
    except aiohttp.ClientError as e:
        status = "error"
        error = f"{type(e).__name__}: {e}"

    t_end = time.perf_counter()
    latency_ms = (t_end - t_send) * 1000.0
    ttft_ms = ttft * 1000.0 if ttft is not None else None

    n_output_tokens = None
    if usage:
        n_output_tokens = usage.get("completion_tokens")
    if n_output_tokens is None:
        n_output_tokens = len(token_times)  # fallback: 1 SSE content-delta ~= 1 token

    tpot_mean_ms = None
    tpot_p95_ms = None
    if len(token_times) >= 2:
        gaps_ms = [(b - a) * 1000.0 for a, b in zip(token_times, token_times[1:])]
        tpot_mean_ms = statistics.mean(gaps_ms)
        tpot_p95_ms = compute_ers.percentile(gaps_ms, 0.95)

    if status == "ok" and n_output_tokens == 0:
        status = "empty_output"

    record = {
        **meta,
        "status": status,
        "error": error,
        "ttft_ms": ttft_ms,
        "tpot_mean_ms": tpot_mean_ms,
        "tpot_p95_ms": tpot_p95_ms,
        "latency_ms": latency_ms,
        "output_tokens": n_output_tokens,
        "prompt_tokens_actual": usage.get("prompt_tokens") if usage else None,
        "output_text": "".join(content_parts),
        "output_text_len": len("".join(content_parts)),
        "t_send_epoch": time.time() - (time.perf_counter() - t_send),
    }
    return record


async def run_conversation(
    conv_id: int,
    rows: list,
    trace_start: float,
    prompt_builder: PromptBuilder,
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    ignore_eos: bool,
    timeout_s: float,
    results: list,
    keep_output_text: bool,
    workload: str = "legacy",
    progress_cb=None,
):
    turn0 = rows[0]
    target_send_time = trace_start + turn0.timestamp_ms / 1000.0
    now = time.perf_counter()
    if target_send_time > now:
        await asyncio.sleep(target_send_time - now)

    last_turn_idx = rows[-1].turn_idx
    # In spec mode we carry a growing message history across the conversation:
    # shared system prefix + per-conv prefix (turn 0) + each turn's fresh user
    # block, with the model's own replies threaded back as assistant turns.
    history: list[dict] = []
    if workload == "spec":
        history.append({"role": "system", "content": prompt_builder.build_system_content()})

    for row in rows:
        if workload == "spec":
            if row.turn_idx == 0:
                user = (
                    prompt_builder.build_conv_prefix(conv_id)
                    + "\n\n"
                    + prompt_builder.build_user_turn(conv_id, 0)
                )
            else:
                user = prompt_builder.build_user_turn(conv_id, row.turn_idx)
            history.append({"role": "user", "content": user})
            messages = list(history)
        else:
            content = prompt_builder.build_turn_content(conv_id, row.turn_idx, row.in_tokens_est)
            messages = [{"role": "user", "content": content}]

        meta = {
            "request_id": f"{conv_id}-{row.turn_idx}",
            "conv_id": conv_id,
            "turn_idx": row.turn_idx,
            "in_warmup": row.in_warmup,
            "target_in_tokens": row.in_tokens_est,
            "out_tokens_max": row.out_tokens_max,
            "scheduled_offset_ms": (time.perf_counter() - trace_start) * 1000.0,
        }
        record = await send_turn(
            session, base_url, model, messages, row.out_tokens_max, ignore_eos, timeout_s, meta
        )
        results.append(record)
        if workload == "spec":
            # Thread the reply back so the next turn is a real extension.
            history.append({"role": "assistant", "content": record.get("output_text") or ""})
        if not keep_output_text:
            # Keep default result files lean; opt in to captured text when
            # exact changed-sample or malformed/truncated-output analysis is
            # required for a promising candidate.
            record.pop("output_text", None)
        if progress_cb:
            progress_cb(record)
        if row.turn_idx != last_turn_idx:
            await asyncio.sleep(row.think_ms / 1000.0)


async def run_benchmark(args) -> list[dict]:
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HOME", args.hf_home)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    shared_prefix_tokens = None if args.prefix_overlap == "strong" else args.weak_prefix_tokens
    prompt_builder = PromptBuilder(tokenizer, seed=args.seed, shared_prefix_tokens=shared_prefix_tokens)

    rows = load_trace(args.trace)
    by_conv = group_by_conv(rows)

    results: list[dict] = []
    n_total = len(rows)
    n_done = 0
    t_start_wall = time.time()

    def progress_cb(record):
        nonlocal n_done
        n_done += 1
        if args.verbose and (n_done % 25 == 0 or n_done == n_total):
            print(f"  [{n_done}/{n_total}] last={record['request_id']} status={record['status']}", file=sys.stderr)

    # force_close avoids a keep-alive reuse race (client picks a pooled
    # connection just as the server side times it out -> ServerDisconnectedError)
    # seen intermittently under this trace's bursty concurrent arrivals.
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        trace_start = time.perf_counter()
        tasks = [
            run_conversation(
                conv_id, conv_rows, trace_start, prompt_builder, session,
                args.base_url, args.model, args.ignore_eos, args.timeout, results,
                args.keep_output_text, args.workload, progress_cb,
            )
            for conv_id, conv_rows in by_conv.items()
        ]
        await asyncio.gather(*tasks)

    results.sort(key=lambda r: (r["conv_id"], r["turn_idx"]))
    return results


def write_outputs(results: list[dict], out_jsonl: Path, out_csv: Path):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    fieldnames = list(results[0].keys()) if results else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="/workspace/lfm-serving/trace_grading_public.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer repo/path. Defaults to --model; set separately when "
             "--model is a served alias such as LFM2.5-1.2B-Instruct.",
    )
    ap.add_argument("--hf-home", default="/workspace/.hf_home")
    ap.add_argument("--out-dir", default=None, help="Directory for results; default results/<timestamp>")
    ap.add_argument("--name", default=None, help="Experiment name; used in default out-dir and EXPERIMENTS.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--workload", choices=["legacy", "spec"], default="legacy",
        help="legacy: one single-user-message prompt per turn (old trace_grading_public "
             "assumptions). spec: official updated workload (shared system prefix + "
             "per-conv prefix + growing multi-turn history with replies threaded back). "
             "Use with trace_grading_spec.jsonl.",
    )
    ap.add_argument(
        "--prefix-overlap", choices=["strong", "weak"], default="strong",
        help="strong (default): every turn shares the whole body with earlier turns of the "
             "same conversation (near-best-case prefix-cache hit rate, matches all prior "
             "EXPERIMENTS.md results). weak: only --weak-prefix-tokens repeat; the rest of "
             "each turn's body is fresh, turn-unique filler (pessimistic cache-hit rate, used "
             "to sanity-check configs aren't overfit to the strong mode's cache behavior).",
    )
    ap.add_argument(
        "--weak-prefix-tokens", type=int, default=256,
        help="Size of the fixed shared anchor in --prefix-overlap=weak mode (default 256).",
    )
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (s)")
    ap.add_argument(
        "--ignore-eos", action="store_true", default=True,
        help="Force generation to run to out_tokens_max instead of stopping at EOS "
             "(default on: gives stable, comparable TTFT/TPOT across configs).",
    )
    ap.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--keep-output-text",
        action="store_true",
        help="Persist generated text in results.jsonl/results.csv for exact "
             "changed-sample and malformed/truncated-output analysis. Default "
             "off to keep benchmark artifacts small.",
    )
    args = ap.parse_args()
    if args.tokenizer_model is None:
        args.tokenizer_model = args.model

    ts = time.strftime("%Y%m%d-%H%M%S")
    name = args.name or ts
    out_dir = Path(args.out_dir) if args.out_dir else Path("/workspace/lfm-serving/results") / name
    out_jsonl = out_dir / "results.jsonl"
    out_csv = out_dir / "results.csv"

    print(
        f"Running trace replay: trace={args.trace} model={args.model} "
        f"tokenizer_model={args.tokenizer_model} base_url={args.base_url}"
    )
    print(f"  prefix_overlap={args.prefix_overlap}" + (f" (shared_prefix_tokens={args.weak_prefix_tokens})" if args.prefix_overlap == "weak" else " (full body shared per conv_id)"))
    t0 = time.perf_counter()
    results = asyncio.run(run_benchmark(args))
    duration_s = time.perf_counter() - t0
    write_outputs(results, out_jsonl, out_csv)
    print(f"Wrote {len(results)} request records to {out_jsonl} and {out_csv}")

    summary = compute_ers.summarize(results, exclude_warmup=True)
    summary["benchmark_duration_s"] = duration_s
    summary["prefix_overlap"] = args.prefix_overlap
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (scored requests, warmup excluded) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
