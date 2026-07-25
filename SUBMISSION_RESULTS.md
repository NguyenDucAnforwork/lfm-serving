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
Image:                    siconhoccode/lfm-serving:fp8-v1
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

## VERDICT (2026-07-25, updated after 4 official H200 submissions): ShortConv fix REJECTED

Four official submissions now isolate every combination of {v0.22.1,
v0.25.1} x {patched, unpatched}:

| Image | Base | Patched? | ERS | TTFT p50/p95 (ms) | TBT median (ms) | Failed |
|---|---|---|---:|---|---:|---:|
| `fp8-v1` | v0.22.1 | no | 60.89 | 50/85 | 4 | 4 |
| `fp8-v0251` | v0.25.1 | no | 59.89 | 52/81 | 4 | 5 |
| `fp8-shortconv-quant-v0221` | v0.22.1 | **yes** | 60.15 | 59/100 | 4 | 6 |
| `fp8-shortconv-quant` | v0.25.1 | **yes** | 51.65 | 81/125 | 4 | 5 |

Conclusions:

1. **TBT is 4ms in all four runs, no exceptions.** The core hypothesis --
   quantizing ShortConv's ~168M previously-unquantized params would reduce
   decode weight-read bandwidth and lower TBT -- is **not confirmed by any
   real H200 data**, despite clear local evidence (RTX PRO 6000 Blackwell)
   of a model-load-memory reduction (-160 MiB, matching the math almost
   exactly) and a modest local TPOT improvement (-5.4%). Whatever the real
   H200/MIG decode bottleneck is, it evidently isn't ShortConv weight-read
   bandwidth in the way predicted. Local RTX-class hardware is confirmed
   (again) to not predict H200 win/loss size -- this project's third
   confirmed case after `max_num_batched_tokens=256` and W4-G64.
2. **v0.25.1 alone is a statistical wash** (59.89 vs 60.89 baseline, inside
   the established 59.78-60.89 noise band) -- ruling out "the base image
   bump caused the regression" as a general explanation.
3. **The patch alone (on v0.22.1) is also a statistical wash** (60.15,
   inside the same noise band) -- ruling out "the patch itself is broken"
   as a general explanation.
4. **Only the v0.25.1 + patch combination regresses hard** (51.65, TTFT
   +60-65% worse than baseline) -- a real, reproducible interaction effect
   between the two, not attributable to either alone. Not investigated
   further: with neither half showing a standalone win, there is no
   remaining upside to chase in this direction, and interaction-effect
   debugging without H200 profiler access is not a good use of further
   submission slots.

**Decision: do not pursue ShortConv quantization further.** Do not submit
`fp8-shortconv-quant-retention` (Q2) -- it is built on the same v0.25.1 +
patch combination confirmed to regress. Revert to `fp8-v1` as the base for
all future candidates. `patches/apply_vllm_shortconv_quant.py` and the
`fp8-shortconv-quant*` Dockerfiles are kept in the repo as a documented,
closed investigation (the underlying vLLM bug this backports -- PR #48917
-- is real and worth knowing about even though it didn't pay off for this
workload/hardware combination), not as active candidates.

## Historical: ShortConv quant_config fix investigation (2026-07-25, superseded by verdict above)

Verified directly against vLLM source (both the installed v0.22.1 and the
v0.25.1 tag on GitHub) that `Lfm2ShortConvDecoderLayer` builds `ShortConv()`
without passing `quant_config`, and `ShortConv.__init__` doesn't even accept
the parameter. Result: the 10 ShortConv layers' `in_proj`/`out_proj`
(~168M params, ~14% of LFM2.5-1.2B) silently stay BF16 under
`--quantization=fp8_per_tensor`, even though `Lfm2MLP` and attention right
next to them ARE quantized. This is the exact bug fixed by upstream vLLM
PR #48917 (merged 2026-07-21, "Fix LFM2 ShortConv Quantization
Configuration") -- not yet in v0.25.1, so backported as a site-packages
patch (`patches/apply_vllm_shortconv_quant.py`, same non-invasive approach
as the existing decode-metadata-fastpath patch).

Two new candidates, both on a v0.25.1 base (was v0.22.1) with the same
validated FP8 runtime flags as the current best (S5, ERS 60.89):

- **Q1** (`submission/Dockerfile.fp8-shortconv-quant-local` +
  `docker-compose.fp8-shortconv-quant-seqs8.yml`): the ShortConv fix alone,
  isolated from any scheduler change.
- **Q2** (`submission/Dockerfile.fp8-shortconv-quant-retention-local` +
  `docker-compose.fp8-shortconv-quant-retention-seqs8.yml`): Q1 plus
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` (vLLM PR #47782, already present
  in v0.25.1 -- no extra backport needed). Targets this workload's shared
  1000-token system prefix + growing per-conversation history shape.

Local validation (RTX PRO 6000 Blackwell, NOT bandwidth-bound like H200 --
useful for correctness/accuracy only, not win-size prediction; also fixed a
separate local-harness bug the same day, see below):

| | Q1 | Q2 | Baseline (unpatched FP8) |
|---|---:|---:|---:|
| Model load memory | 1.23 GiB | 1.26 GiB | 1.39 GiB |
| TPOT mean/p95 (ms) | 1.79/1.94 | not clean -- see note | 1.89/2.10 |
| Errors | 0/420 | 0/420 | 0/420 |
| GSM8K strict/flexible | 0.665/0.675 | 0.68/0.69 | 0.65/0.66 |
| ARC acc/acc_norm | 0.3874/0.4164 | 0.3899/0.4164 | 0.3839/0.4266 |
| GPQA ungated-mirror (198q) | 0.3333 (66/198) | 0.2929 (58/198) | 0.3131 (62/198) |

Model load dropping by ~160 MiB matches the ~168M-param prediction almost
exactly -- direct evidence the patch is quantizing the intended layers.
Every accuracy delta is within stderr noise; no candidate shows real
degradation. Q2's ERS/TPOT numbers are not directly comparable to Q1's --
they were measured while the accuracy gate ran concurrently on the same
GPU (to save wall-clock) and on a freshly-installed venv with a cold
torch-compile cache, both of which inflate latency independent of the
retention mechanism. The only clean signal from that run is "no crash, no
error." **Neither candidate has an official H200 number yet** -- that is
the current top priority (see queue below).

Also fixed the same day: `benchmark/gen_spec_trace.py` was copying stale
`in_warmup` flags from the old workload's trace, excluding 90/420 requests
from local ERS by default even though the official grader for the current
spec scores all 420 (warmup_count=0). Fixed trace is
`trace_grading_spec_v2.jsonl`; `run_experiment.sh` now defaults to it and
preflights the trace shape before every run.

## Current submit/probe queue (updated 2026-07-25, CPU-bottleneck hypothesis)

ShortConv candidates (Q1/Q2) are REJECTED, see verdict above. All future
candidates build on `fp8-v1` (v0.22.1, unpatched) only.

**New working hypothesis**: TBT median was 4ms in ALL FOUR ShortConv-verdict
submissions, completely unmoved by quantizing an additional ~168M
previously-unquantized params. That rules out GPU compute/weight-bandwidth
as the real H200/MIG bottleneck. Most likely culprit: vLLM V1 runs 3
processes (API server, EngineCore, GPU worker) competing for a small vCPU
allocation on the MIG slice -- CPU-side per-step overhead, not GPU work, is
probably the real floor. Reprioritized the queue around this.

Submit in this order:

1. **`submission/docker-compose.fp8-metadata-fastpath-async-seqs8.yml`**
   (image `<DOCKERHUB_USER>/lfm-serving:fp8-metadata-fastpath-async`) --
   NEW, submit FIRST. Stacks two CPU-overhead mitigations: the decode
   metadata fast-path patch (baked into the image, avoids re-uploading
   per-step scheduling metadata for pure-decode batches) plus
   `--async-scheduling` (overlaps CPU scheduling with GPU execution). Both
   attack the CPU-bottleneck hypothesis above from different angles.
   **Known risk**: a prior local probe on `--async-scheduling` with this
   same base config found 2/420 gibberish outputs (session 10,
   `EXPERIMENTS.md` "Async scheduling probe") despite good latency (ERS
   69.5 local, TBT 2.78ms) -- that probe predates this session and was
   never re-verified for coherence. Check output text before trusting any
   ERS improvement from this image, official or local.
2. **`submission/docker-compose.fp8-async-sync-seqs8.yml`** (explicit
   `--no-async-scheduling` control) and
   **`submission/docker-compose.fp8-async-seqs8.yml`** (`--async-scheduling`
   alone, no metadata patch) -- both reuse the existing `fp8-v1` image, no
   new build needed. Useful for decomposing item 1's result: if the
   combined candidate wins, these two isolate how much came from each half.
3. `submission/docker-compose.fp8-seqs16.yml`
   - Image: `siconhoccode/lfm-serving:fp8-v1`
   - Same validated FP8 config, only `max_num_seqs=8 -> 16`
   - Goal: test whether official 4-6 failures are queue/deadline starvation.
4. `submission/docker-compose.fp8-metadata-fastpath-seqs8.yml` (metadata
   patch alone, no async) -- image `siconhoccode/lfm-serving:fp8-metadata-fastpath`,
   **not yet pushed**. Lower priority than item 1 now that the combined
   candidate exists; only build this separately if item 1's result needs
   decomposing and item 2's plain-async isolation isn't enough on its own.
5. Submit `submission/docker-compose.fp8-metadata-fastpath-seqs16.yml` only if
   either probe above shows useful signal.
6. Optional one-slot W4 probe:
   `submission/docker-compose.w4a16-g64-mlp10-15-attn10-12-14-bf16.machete.yml`
   after building `siconhoccode/lfm-serving:w4a16-g64-mlp10-15-attn10-12-14-bf16`.
   This is not expected to beat FP8; local trace passed but GSM8K regressed.
   Already submitted once officially and rejected (49.13-49.71 ERS) -- only
   resubmit if a new checkpoint/backend changes the picture.

## Related image/config inventory

Verified live against the Docker Hub API (2026-07-25, public repo, no auth
needed: `curl https://hub.docker.com/v2/repositories/siconhoccode/lfm-serving/tags`)
-- table below reflects actual pushed tags, not assumed ones. This caught a
real bug: every compose file previously pointed at a bare `:fp8` tag that
was never actually pushed (only `:fp8-v1` exists) -- fixed in
`docker-compose.fp8.yml` and `docker-compose.fp8-seqs16.yml`.

| Image | Status | Configs |
|---|---|---|
| `siconhoccode/lfm-serving:fp8-v1` | Pushed; current best official image (ERS 60.89) | `docker-compose.fp8.yml`, `docker-compose.fp8-seqs16.yml` |
| `siconhoccode/lfm-serving:fp8-shortconv-quant` | Pushed; **official ERS 51.65, WORSE than baseline** -- see "New candidates" above | `docker-compose.fp8-shortconv-quant-seqs8.yml` |
| `siconhoccode/lfm-serving:fp8-shortconv-quant-v0221` | Pushed 2026-07-25, awaiting official result (isolates ShortConv fix from the v0.25.1 base swap) | `docker-compose.fp8-shortconv-quant-v0221-seqs8.yml` |
| `siconhoccode/lfm-serving:fp8-shortconv-quant-retention` | Pushed 2026-07-25, not yet submitted (blocked pending Q1 diagnosis) | `docker-compose.fp8-shortconv-quant-retention-seqs8.yml` |
| `siconhoccode/lfm-serving:fp8-v0251` | Pushed 2026-07-21 -- plain v0.25.1 base, no ShortConv patch; not referenced by any current compose file, origin/purpose unclear, do not assume it's safe to reuse without checking what it actually contains |
| `siconhoccode/lfm-serving:w4a16-g64-mlp10-15-bf16` | Pushed; official rejected | W4 MLP10-15 Marlin/Machete |
| `siconhoccode/lfm-serving:w4a16-g64-mlp10-15-attn10-12-14-bf16` | Pushed; official rejected | `docker-compose.w4a16-g64-mlp10-15-attn10-12-14-bf16.{marlin,machete}.yml` |
| `siconhoccode/lfm-serving:safe-bf16-v1` | Pushed | not currently referenced by `docker-compose.safe-bf16.yml` (that file still has the `<DOCKERHUB_USER>` placeholder -- fix before submitting) |
| `siconhoccode/lfm-serving:fp8-metadata-fastpath` | **Not pushed** -- needs local build | `docker-compose.fp8-metadata-fastpath-seqs8.yml`, `docker-compose.fp8-metadata-fastpath-seqs16.yml` |
| `siconhoccode/lfm-serving:w4a16-gptq`, `w4a16-g64-late-mlp-bf16` | **Not pushed** -- guarded/unvalidated candidates, do not build until their gates pass | `docker-compose.w4a16-gptq.yml`, `docker-compose.w4a16-g64-late-mlp-bf16.local.yml` |

## Accuracy notes

- Official successful submissions so far report `accuracy_drop=0`.
- Local FP8 accuracy was previously cleared on ARC, GSM8K, and official
  `gpqa_diamond_zeroshot` with the gated dataset after access was granted.
- W4 mixed-precision candidates can preserve ARC but repeatedly regress GSM8K
  by more than the 2.5 pp preferred threshold. Treat W4 as latency/backend probe
  only unless a new checkpoint passes the full accuracy gate.
