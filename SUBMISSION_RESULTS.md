# Official Submission Results

This file records official Viettel BTC portal results for
`LFM2.5-1.2B-Instruct`. These measurements were produced on the organizer's
H200/MIG grading infrastructure and are more authoritative than local RTX 3090
replay results.

## Environment and submission format

- Base runtime: `vllm/vllm-openai:v0.22.1`
- Model alias: `LFM2.5-1.2B-Instruct`
- Official trace size: 420 requests
- Best quantization path so far: `--quantization=fp8_per_tensor`
- Legacy `--quantization=fp8` remains rejected due incoherent local output.
- W4 compressed-tensors configs must not pass
  `--quantization=compressed-tensors` explicitly; vLLM 0.22.1 must autodetect it
  from `/model/config.json`. The explicit flag caused a competition-container
  startup failure.

## Submission summary

| ID | Candidate | ERS | TTFT p50 | TTFT p95 | TBT median | Failed requests | Accuracy drop | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| S0 | BF16 but compose referenced wrong `safe-bf16` image tag | — | — | — | — | — | — | Packaging failure; not graded |
| S1 | BF16, `max_num_seqs=8`, `max_num_batched_tokens=512` | 49.69 | 49 ms | 83 ms | not exposed | 7 | 0 | Valid safe baseline |
| S2 | BF16, `max_num_batched_tokens=256` | 46.44 | 59 ms | 140 ms | 6 ms | 7 | 0 | Rejected; local optimum did not transfer |
| S3 | FP8 per-tensor, `max_num_batched_tokens=512` | 60.20 | 52 ms | 85 ms | 4 ms | 4 | 0 | Strong candidate |
| S4 | FP8 repeat, same config | 59.78 | 53 ms | 83 ms | 4 ms | 4 | 0 | Confirms official noise band |
| S5 | FP8 repeat, same config | **60.89** | 50 ms | 85 ms | 4 ms | 4 | 0 | Current best official result |
| S6 | W4A16 G64, MLP 10-15 BF16, Marlin | 49.71 | 49 ms | 83 ms | 6 ms | 8 | 0 | Rejected; W4 backend slower on H200 |
| S7 | Same W4 checkpoint, Machete | 49.13 | 54 ms | 84 ms | 6 ms | 7 | 0 | Rejected; backend switch did not help |

## Current best official configuration

```text
Image:                    siconhoccode/lfm-serving:fp8
Quantization:             fp8_per_tensor
max_model_len:            5120
max_num_seqs:             8
max_num_batched_tokens:   512
gpu_memory_utilization:   0.70
prefix caching:           enabled
prefix hash:              xxhash
Official ERS best:        60.89
Official ERS typical:     about 60.2 +/- 0.5
TBT median:               4 ms
Failures:                 4 / 420
Accuracy drop:            0
```

## Conclusions from official submissions

1. **FP8 is the reliable best path.** It improves BF16 from 49.69 ERS to about
   60.2 ERS, reduces failures from 7 to 4, and preserves `accuracy_drop=0`.
2. **The dominant remaining gap is decode.** Good and bad candidates have
   similar TTFT p50/p95, while ERS tracks TBT: FP8 at 4 ms scores around 60,
   W4/BF16 at 6 ms scores around 49.
3. **Stock W4 compressed-tensors is rejected on official H200.** Local RTX 3090
   ranked W4 faster than FP8, but official H200 reversed the ranking:
   FP8 median TBT 4 ms, W4 median TBT 6 ms. Both Marlin and Machete were poor,
   so the issue is not a simple backend selection mistake. Hidden accuracy is
   also not the cause because `accuracy_drop=0`.
4. **Local RTX 3090 is useful for correctness and failure filtering, not winner
   prediction.** Two major local-to-official inversions occurred:
   `max_num_batched_tokens=256` and W4 G64.
5. **The remaining easy point is likely failure/tail handling.** FP8 still has
   4 failures in all official repeats. If these are queue/deadline failures,
   `max_num_seqs=16` may recover some score without touching accuracy.

## Current submit/probe queue

Submit in this order:

1. `submission/docker-compose.fp8-seqs16.yml`
   - Image: `siconhoccode/lfm-serving:fp8`
   - Same validated FP8 config, only `max_num_seqs=8 -> 16`
   - Goal: test whether official 4 failures are queue/deadline starvation.
2. Build/push `siconhoccode/lfm-serving:fp8-metadata-fastpath`, then submit
   `submission/docker-compose.fp8-metadata-fastpath-seqs8.yml`
   - Same runtime flags as best FP8.
   - Bakes local vLLM decode metadata fast-path patch.
   - Goal: test whether the measured local launch/copy event reduction transfers.
3. Submit `submission/docker-compose.fp8-metadata-fastpath-seqs16.yml` only if
   either probe above shows useful signal.
4. Optional one-slot W4 probe:
   `submission/docker-compose.w4a16-g64-mlp10-15-attn10-12-14-bf16.machete.yml`
   after building `siconhoccode/lfm-serving:w4a16-g64-mlp10-15-attn10-12-14-bf16`.
   This is not expected to beat FP8; local trace passed but GSM8K regressed.

## Related image/config inventory

| Image | Status | Configs |
|---|---|---|
| `siconhoccode/lfm-serving:fp8` | Already pushed; current best official image | `docker-compose.fp8.yml`, `docker-compose.fp8-seqs16.yml` |
| `siconhoccode/lfm-serving:w4a16-g64-mlp10-15-bf16` | Already pushed; official rejected | W4 MLP10-15 Marlin/Machete |
| `siconhoccode/lfm-serving:fp8-metadata-fastpath` | Needs local build/push | `docker-compose.fp8-metadata-fastpath-seqs8.yml`, `docker-compose.fp8-metadata-fastpath-seqs16.yml` |
| `siconhoccode/lfm-serving:w4a16-g64-mlp10-15-attn10-12-14-bf16` | Needs local build/push | `docker-compose.w4a16-g64-mlp10-15-attn10-12-14-bf16.machete.yml` |

## Accuracy notes

- Official successful submissions so far report `accuracy_drop=0`.
- Local FP8 accuracy was previously cleared on ARC, GSM8K, and official
  `gpqa_diamond_zeroshot` with the gated dataset after access was granted.
- W4 mixed-precision candidates can preserve ARC but repeatedly regress GSM8K
  by more than the 2.5 pp preferred threshold. Treat W4 as latency/backend probe
  only unless a new checkpoint passes the full accuracy gate.
