#!/usr/bin/env python3
"""Apply the local vLLM decode metadata fast-path patch inside a Docker image.

The base image is pinned to vllm-openai:v0.22.1. This script edits the installed
site-packages file directly and is intentionally idempotent so Docker layer
caching/rebuilds are safe.
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


def main() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    target = purelib / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    if not target.exists():
        raise SystemExit(f"target file not found: {target}")

    text = target.read_text()
    if "decode_req_indices_gpu" in text and "_discard_request_mask_gpu_is_zero" in text:
        print(f"decode metadata fast-path already applied: {target}")
        return

    text = replace_once(
        text,
        """        self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        # Maps current batch position -> previous batch position (-1 for new reqs)
""",
        """        self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        # Fast-path constants for pure decode batches where every request
        # schedules exactly one token. Avoid re-uploading req_indices,
        # query_pos=zeros, and num_scheduled_tokens=ones every decode step.
        self.decode_req_indices_gpu = torch.arange(
            self.max_num_reqs, dtype=torch.int64, device=self.device
        )
        self.decode_query_pos_gpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=self.device
        )
        self.decode_num_scheduled_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Maps current batch position -> previous batch position (-1 for new reqs)
""",
        "decode device constants",
    )

    text = replace_once(
        text,
        """        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
""",
        """        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self._discard_request_mask_gpu_is_zero = True
        self.num_decode_draft_tokens = self._make_buffer(
""",
        "discard mask state flag",
    )

    text = replace_once(
        text,
        """        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
""",
        """        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        is_uniform_one_token_batch = (
            total_num_scheduled_tokens == num_reqs
            and np.all(num_scheduled_tokens == 1)
        )
""",
        "uniform one-token predicate",
    )

    text = replace_once(
        text,
        """        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returning
        self.discard_request_mask.np[:num_reqs] = (
            self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)
""",
        """        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returning
        is_pure_decode_batch = (
            is_uniform_one_token_batch
            and np.all(
                self.input_batch.num_computed_tokens_cpu[:num_reqs]
                >= self.input_batch.num_prompt_tokens[:num_reqs]
            )
        )
        if is_pure_decode_batch:
            self.discard_request_mask.np[:num_reqs] = False
            if not self._discard_request_mask_gpu_is_zero:
                self.discard_request_mask.gpu.zero_()
                self._discard_request_mask_gpu_is_zero = True
        else:
            self.discard_request_mask.np[:num_reqs] = (
                self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
            )
            self.discard_request_mask.copy_to_gpu(num_reqs)
            self._discard_request_mask_gpu_is_zero = not bool(
                self.discard_request_mask.np[:num_reqs].any()
            )
""",
        "discard mask fast path",
    )

    text = replace_once(
        text,
        """        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )
""",
        """        if is_uniform_one_token_batch:
            req_indices_gpu = self.decode_req_indices_gpu[:num_reqs]
            query_pos_gpu = self.decode_query_pos_gpu[:num_reqs]
            num_scheduled_tokens_gpu = self.decode_num_scheduled_tokens_gpu[:num_reqs]
            self.positions[:num_reqs] = self.num_computed_tokens[:num_reqs].to(
                torch.int64
            )
        else:
            self.req_indices.np[:total_num_scheduled_tokens] = req_indices
            self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
            req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

            self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
            query_pos_gpu = self.query_pos.gpu[:total_num_scheduled_tokens]
            self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
            self.num_scheduled_tokens.copy_to_gpu(num_reqs)
            num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
            self.positions[:total_num_scheduled_tokens] = (
                self.num_computed_tokens[req_indices_gpu].to(torch.int64)
                + query_pos_gpu
            )
""",
        "req/query/num scheduled fast path",
    )

    target.write_text(text)
    print(f"applied decode metadata fast-path: {target}")


if __name__ == "__main__":
    main()
