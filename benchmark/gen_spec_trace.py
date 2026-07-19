#!/usr/bin/env python3
"""Generate a trace file matching the OFFICIAL updated workload spec (18/07/2026).

Spec (grading_workflow_spec.jsonl):
    num_conversations           = 70
    user_turns_per_conversation = 6            (total 420 requests)
    shared_system_prefix_tokens = 1000         (identical across ALL conversations)
    per_conversation_prefix_tokens = 1000      (unique per conv, added to turn 0)
    new_user_tokens_per_turn    = 150          (each turn's fresh user text)
    output_tokens_per_turn_pinned = 300        (assistant reply, every turn)
    arrival                     = Poisson, seed 42

This differs from the older trace_grading_public.jsonl (constant ~4000 in / 200
out, no growing context). Here the context GROWS turn over turn:

    in_tokens(turn t) ~= shared_system(1000) + per_conv(1000)
                         + new_user(150)
                         + t * (output_pinned(300) + new_user(150))
                       = 2150 + 450 * t     (t = 0..5)  -> 2150..4400

We preserve the seed-42 arrival timing (turn-0 timestamp_ms), think_ms, and the
warmup flags from the existing public trace (already Poisson seed 42), and only
rewrite the token-length structure (in_tokens_est growing) and out_tokens_max=300
so downstream replay drives the real multi-turn growing-context shape.
"""
from __future__ import annotations

import json
from pathlib import Path

SHARED_SYSTEM_PREFIX_TOKENS = 1000
PER_CONVERSATION_PREFIX_TOKENS = 1000
NEW_USER_TOKENS_PER_TURN = 150
OUTPUT_TOKENS_PER_TURN_PINNED = 300

ROOT = Path(__file__).resolve().parent.parent
SRC_TRACE = ROOT / "trace_grading_public.jsonl"
OUT_TRACE = ROOT / "trace_grading_spec.jsonl"


def expected_in_tokens(turn_idx: int) -> int:
    """Approx input tokens the server sees at this turn (chat-template overhead
    excluded; it is small and added at replay time by the real tokenizer)."""
    base = SHARED_SYSTEM_PREFIX_TOKENS + PER_CONVERSATION_PREFIX_TOKENS + NEW_USER_TOKENS_PER_TURN
    grown = turn_idx * (OUTPUT_TOKENS_PER_TURN_PINNED + NEW_USER_TOKENS_PER_TURN)
    return base + grown


def main() -> None:
    src = [json.loads(l) for l in SRC_TRACE.read_text().splitlines() if l.strip()]
    out_rows = []
    for r in src:
        row = dict(r)
        turn = row["turn_idx"]
        row["in_tokens_est"] = expected_in_tokens(turn)
        # in_chars is only informational; keep a rough char estimate (~3 chars/token)
        row["in_chars"] = row["in_tokens_est"] * 3
        row["out_tokens_max"] = OUTPUT_TOKENS_PER_TURN_PINNED
        out_rows.append(row)

    with open(OUT_TRACE, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    convs = sorted({r["conv_id"] for r in out_rows})
    print(f"Wrote {len(out_rows)} rows ({len(convs)} conversations) -> {OUT_TRACE}")
    sample = [r for r in out_rows if r["conv_id"] == convs[0]]
    print("Sample conv token profile (turn_idx, in_tokens_est, out_tokens_max):")
    for r in sample:
        print(f"  {r['turn_idx']}: {r['in_tokens_est']} in / {r['out_tokens_max']} out")


if __name__ == "__main__":
    main()
