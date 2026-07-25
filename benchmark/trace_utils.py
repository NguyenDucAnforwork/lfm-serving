"""Shared helpers for loading the trace and generating synthetic prompts.

The public trace (trace_grading_public.jsonl) carries only timing and
token-length metadata, not real prompt text:

    {"conv_id", "turn_idx", "in_warmup", "timestamp_ms", "think_ms",
     "in_chars", "in_tokens_est", "out_tokens_max"}

Empirically (see EXPERIMENTS.md):
  * every conversation has exactly 6 turns
  * in_tokens_est is ~constant (~4000) across all turns of a conversation,
    not growing turn over turn -> each turn re-sends a large shared
    context rather than an ever-growing chat history
  * only turn_idx==0 carries a meaningful timestamp_ms (the conversation's
    Poisson arrival offset from trace start); turns 1..5 fire at
    (completion time of the previous turn) + think_ms
  * think_ms is constant (3000ms) and out_tokens_max is constant (200)
    in the public trace, but the code does not assume that generally

Synthetic prompt design: for each conv_id we generate one long filler
"context" once (deterministic given --seed), then for each turn we take a
token-exact prefix of that same context (so consecutive turns of a
conversation share a real, cacheable token prefix -- this is what makes
prefix-caching experiments meaningful) and append a short turn-specific
suffix question. This is not real data and is not meant to be: it exists
only to drive the server with the right token lengths and arrival pattern.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Chat-template + suffix overhead varies a little by tokenizer; measured
# empirically for LFM2.5's template (see README). We slice the filler
# context short enough to leave room for it.
TEMPLATE_OVERHEAD_TOKENS = 24
SUFFIX_TEMPLATE = "\n\n[Turn {turn}] Based only on the passage above, answer in a few sentences: what is the single most important idea, and why?"

_FILLER_VOCAB = (
    "system architecture latency throughput scheduler cache token batch "
    "context window inference decode prefill request queue worker replica "
    "gpu memory bandwidth utilization concurrency stream response client "
    "server model weight tensor kernel attention layer embedding vocabulary "
    "policy retry timeout error rate percentile median deploy container "
    "network protocol endpoint router balancer shard partition metric log "
    "trace span session conversation dialogue turn history summary topic"
).split()


@dataclass
class TraceRow:
    conv_id: int
    turn_idx: int
    in_warmup: bool
    timestamp_ms: int
    think_ms: int
    in_chars: int
    in_tokens_est: int
    out_tokens_max: int


def load_trace(path: str | Path) -> list[TraceRow]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                TraceRow(
                    conv_id=d["conv_id"],
                    turn_idx=d["turn_idx"],
                    in_warmup=bool(d["in_warmup"]),
                    timestamp_ms=d["timestamp_ms"],
                    think_ms=d["think_ms"],
                    in_chars=d["in_chars"],
                    in_tokens_est=d["in_tokens_est"],
                    out_tokens_max=d["out_tokens_max"],
                )
            )
    return rows


def group_by_conv(rows: list[TraceRow]) -> dict[int, list[TraceRow]]:
    by_conv: dict[int, list[TraceRow]] = {}
    for r in rows:
        by_conv.setdefault(r.conv_id, []).append(r)
    for conv_id in by_conv:
        by_conv[conv_id].sort(key=lambda r: r.turn_idx)
    return by_conv


def _filler_text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_FILLER_VOCAB) for _ in range(n_words))


class PromptBuilder:
    """Builds token-length-matched synthetic prompts per conversation.

    Two overlap modes, controlling how much of a turn's prompt is a
    byte-for-byte repeat of earlier turns in the same conversation (i.e. how
    cacheable it is for vLLM's automatic prefix caching):

      * shared_prefix_tokens=None ("strong" / default): every turn is a
        token-exact prefix of the SAME per-conv_id filler sequence, so
        essentially the whole body (~all but the trailing suffix) is a
        repeat of turn 0's content. This is what all of EXPERIMENTS.md's
        results up through the batchtok sweep were measured with. It is
        also, deliberately, closer to a best-case cache-hit rate than most
        real chat traffic would produce -- we do NOT claim the hidden
        grading trace has this same near-perfect overlap; it's one
        plausible pattern consistent with the public trace's near-constant
        in_tokens_est (see EXPERIMENTS.md).
      * shared_prefix_tokens=N (int, "weak" mode): only the first N tokens
        are a repeat across turns of a conversation (a fixed-size shared
        "anchor", e.g. a system preamble or running summary); the rest of
        the body is freshly generated filler unique to that turn, so at
        most N tokens can ever be a prefix-cache hit no matter how long the
        prompt is. This is a deliberately pessimistic alternative used only
        to check whether the winning config is overfit to the strong mode's
        cache-hit rate (see EXPERIMENTS.md "Benchmark assumptions").

    Caches filler token sequences per conv_id (and, in weak mode, per
    (conv_id, turn_idx) for the non-shared portion), generated
    deterministically from --seed, so repeated benchmark runs are
    reproducible and comparable across experiments.
    """

    def __init__(self, tokenizer, seed: int = 0, shared_prefix_tokens: Optional[int] = None):
        self.tokenizer = tokenizer
        self.seed = seed
        self.shared_prefix_tokens = shared_prefix_tokens  # None = share everything ("strong")
        self._base_ids_by_conv: dict[int, list[int]] = {}
        self._fresh_ids_by_key: dict[tuple[int, int], list[int]] = {}

    def _gen_ids(self, rng: random.Random, min_len: int) -> list[int]:
        # Over-generate a bit; words ~1.3 tokens each for this tokenizer.
        text = _filler_text(rng, max(64, int(min_len * 1.6)))
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        while len(ids) < min_len:
            text += " " + _filler_text(rng, 64)
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids

    def _base_ids(self, conv_id: int, min_len: int) -> list[int]:
        ids = self._base_ids_by_conv.get(conv_id)
        if ids is not None and len(ids) >= min_len:
            return ids
        rng = random.Random(f"{self.seed}-{conv_id}")
        ids = self._gen_ids(rng, min_len)
        self._base_ids_by_conv[conv_id] = ids
        return ids

    def _fresh_ids(self, conv_id: int, turn_idx: int, min_len: int) -> list[int]:
        if min_len <= 0:
            return []
        key = (conv_id, turn_idx)
        ids = self._fresh_ids_by_key.get(key)
        if ids is not None and len(ids) >= min_len:
            return ids
        rng = random.Random(f"{self.seed}-{conv_id}-{turn_idx}-fresh")
        ids = self._gen_ids(rng, min_len)
        self._fresh_ids_by_key[key] = ids
        return ids

    def build_turn_content(self, conv_id: int, turn_idx: int, target_tokens: int) -> str:
        suffix = SUFFIX_TEMPLATE.format(turn=turn_idx)
        suffix_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
        body_len = max(target_tokens - len(suffix_ids) - TEMPLATE_OVERHEAD_TOKENS, 16)

        if self.shared_prefix_tokens is None:
            # Strong overlap: whole body is a prefix of the same per-conv sequence.
            base_ids = self._base_ids(conv_id, body_len)
            body_text = self.tokenizer.decode(base_ids[:body_len], skip_special_tokens=True)
        else:
            # Weak overlap: only a fixed-size shared anchor repeats; the
            # remainder is fresh (turn-unique) filler.
            shared_len = min(self.shared_prefix_tokens, body_len)
            fresh_len = body_len - shared_len
            base_ids = self._base_ids(conv_id, shared_len)[:shared_len]
            fresh_ids = self._fresh_ids(conv_id, turn_idx, fresh_len)[:fresh_len]
            body_text = (
                self.tokenizer.decode(base_ids, skip_special_tokens=True)
                + " "
                + self.tokenizer.decode(fresh_ids, skip_special_tokens=True)
            )
        return body_text + suffix

    def measure_prompt_tokens(self, content: str) -> int:
        messages = [{"role": "user", "content": content}]
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        return len(ids)
