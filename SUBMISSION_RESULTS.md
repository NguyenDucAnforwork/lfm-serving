# Official Submission Results

This file records the official Viettel BTC portal results obtained so far for the `LFM2.5-1.2B-Instruct` serving submission. These measurements were produced on the organizer's grading infrastructure and should be treated as more authoritative than the local RTX 3090 replay results.

## Environment and submission format

- Base runtime: `vllm/vllm-openai:v0.22.1`
- Model: `LFM2.5-1.2B-Instruct`
- Docker image tags used:
  - `siconhoccode/lfm-serving:safe-bf16-v1`
  - `siconhoccode/lfm-serving:fp8-v1`
- Both tags point to the same Docker image; BF16 versus FP8 is selected entirely through the submitted Docker Compose command.
- Official trace size: 420 requests
- Accuracy gate result for every successful measured run: `accuracy_drop = 0`, `f_delta = 1`, `penalty = 1`

## Submission summary

| ID | Runtime configuration | Status | ERS / final score | TTFT p50 | TTFT p95 | TBT median | Failed requests | Accuracy drop | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| S0 | BF16 compose referencing nonexistent tag `siconhoccode/lfm-serving:safe-bf16` | Failed before grading | — | — | — | — | — | — | Packaging error; fixed by using the `-v1` tag |
| S1 | BF16, `max_num_batched_tokens=512` | Success | **49.69** | 49 ms | 83 ms | Not exposed | 7 | 0 | Valid safe baseline |
| S2 | FP8 per-tensor, `max_num_batched_tokens=512` | Success | **60.20** | 52 ms | 85 ms | 4 ms | 4 | 0 | **Current official best** |
| S3 | BF16, `max_num_batched_tokens=256` | Success | **46.44** | 59 ms | 140 ms | 6 ms | 7 | 0 | Rejected; smaller prefill budget regressed both TTFT and ERS |

## Detailed results

### S0 — Invalid Docker image tag

The first BF16 submission referenced:

```yaml
image: siconhoccode/lfm-serving:safe-bf16
```

Only the following tag had actually been pushed:

```text
siconhoccode/lfm-serving:safe-bf16-v1
```

The organizer therefore could not pull the requested image, and the submission failed before benchmarking. The Docker image itself was not the problem. The compose file was corrected to use the exact pushed tag.

### S1 — BF16 baseline, batch-token budget 512

Key command settings:

```text
--dtype=bfloat16
--max-model-len=5120
--gpu-memory-utilization=0.70
--tensor-parallel-size=1
--max-num-seqs=8
--max-num-batched-tokens=512
--enable-prefix-caching
--prefix-caching-hash-algo=xxhash
--disable-log-stats
--disable-uvicorn-access-log
```

Official metrics:

```text
ERS / final score: 49.69
Total requests:    420
TTFT p50:          49 ms
TTFT p95:          83 ms
Failed requests:   7
Accuracy drop:     0
f_delta:           1
Penalty:           1
Warmup count:      0
```

Interpretation:

- The submission was fully valid and passed the accuracy gate.
- TTFT was already reasonably low.
- The lower score relative to FP8 indicates that decode cost, rather than prefill latency alone, was the main performance bottleneck.
- Seven failed requests hurt the score, but they were not large enough to explain the full gap to the FP8 result.

### S2 — FP8 per-tensor, batch-token budget 512

The FP8 run used the same general serving shape as S1, with the additional flag:

```text
--quantization=fp8_per_tensor
```

The legacy `--quantization=fp8` path was deliberately not used because it had produced incoherent output during local testing.

Official metrics:

```text
ERS / final score: 60.20
Total requests:    420
TTFT p50:          52 ms
TTFT p95:          85 ms
TBT median:        4 ms
Failed requests:   4
Accuracy drop:     0
f_delta:           1
Penalty:           1
Warmup count:      0
```

Improvement over BF16-512:

```text
Absolute ERS gain:   +10.51 points
Relative ERS gain:   approximately +21.2%
Failed requests:     7 -> 4
TTFT p50:            49 -> 52 ms
TTFT p95:            83 -> 85 ms
Accuracy drop:       unchanged at 0
```

Interpretation:

- FP8 produced a large and clean official gain without measurable accuracy loss.
- TTFT stayed almost unchanged, so most of the gain came from faster token generation rather than faster prompt processing.
- The result validates weight quantization as the strongest mechanism tested so far.
- This is the current official best configuration and should remain the reference point for future experiments.

### S3 — BF16, batch-token budget 256

This run changed only:

```text
--max-num-batched-tokens=512 -> 256
```

Official metrics:

```text
ERS / final score: 46.44
Total requests:    420
TTFT p50:          59 ms
TTFT p95:          140 ms
TBT median:        6 ms
Failed requests:   7
Accuracy drop:     0
f_delta:           1
Penalty:           1
Warmup count:      0
```

Regression relative to BF16-512:

```text
ERS change:        -3.25 points
TTFT p50:          49 -> 59 ms
TTFT p95:          83 -> 140 ms
Failed requests:   unchanged at 7
```

Interpretation:

- The local RTX 3090 optimum at 256 did not transfer to the official H200 grading environment.
- A 256-token scheduler budget was too small for the official workload and caused long prefills to be split across more scheduler iterations.
- The added scheduling overhead worsened both median and tail TTFT, while TBT remained worse than the FP8 result.
- FP8-256 should not be prioritized solely on the basis of the previous local optimum.

## Main conclusions

1. **Packaging must use exact pushed Docker tags.** The initial failure was caused by referencing `safe-bf16` instead of `safe-bf16-v1`.
2. **FP8 per-tensor quantization is the only major official win found so far.** It improved ERS from 49.69 to 60.20 while preserving accuracy.
3. **The current official bottleneck is decode efficiency.** FP8 materially improved the score while leaving TTFT nearly unchanged.
4. **Local scheduler optima do not necessarily transfer to the official H200 environment.** Reducing `max_num_batched_tokens` from 512 to 256 caused a clear regression.
5. **Future work should be mechanism-driven rather than a blind parameter sweep.** The next high-value directions are profiling the FP8 decode step, evaluating a genuinely faster online INT4/TorchAO path, testing mixed precision if full INT4 is unstable, and considering FP8 KV cache or profile-guided CUDA graph/compile changes only when measurements identify corresponding bottlenecks.

## Current best submission

```text
Quantization:             fp8_per_tensor
max_model_len:            5120
max_num_seqs:             8
max_num_batched_tokens:   512
gpu_memory_utilization:   0.70
prefix caching:           enabled
prefix hash:              xxhash
Official ERS:             60.20
Accuracy drop:            0
```

## Recommended target

The immediate engineering target is not an arbitrary leaderboard score but a measurable reduction in decode cost:

```text
Current TBT median: 4 ms
Near-term target:   approximately 3.0-3.5 ms
```

Reaching that range without increasing TTFT tails, failure count, or accuracy drop would provide a technically justified path from the current 60.20 score toward the mid-to-high 60s.