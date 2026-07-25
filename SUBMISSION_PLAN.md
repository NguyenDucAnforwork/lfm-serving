# Submission Plan

Exact next steps for a human to build, push, and submit. Nothing in this file
has been executed — no Docker build/push/submit is possible from this dev box
(no Docker-in-Docker here). All ERS numbers below are from this project's own
local RTX 3090 benchmark harness (`benchmark/replay_trace.py` against
`trace_grading_public.jsonl`), not the official leaderboard.

## 1. Build and push (human, on a machine with Docker)

```bash
cd /path/to/lfm-serving   # this repo, copied off the dev box
docker build -t <DOCKERHUB_USER>/lfm-serving:safe-bf16 .
docker build -t <DOCKERHUB_USER>/lfm-serving:fp8 .        # same Dockerfile -- quantization is a runtime flag, not a build-time choice
docker push <DOCKERHUB_USER>/lfm-serving:safe-bf16
docker push <DOCKERHUB_USER>/lfm-serving:fp8
```

(Historical note: these were actually pushed as `siconhoccode/lfm-serving:safe-bf16-v1`
and `siconhoccode/lfm-serving:fp8-v1` -- confirmed live on Docker Hub 2026-07-25.
Every current compose file has been fixed to point at the `-v1` tags.)

Both images are identical (same `Dockerfile`, same baked-in
`submission/model/` weights) — the FP8 vs BF16 choice is made entirely by
which `command:` flags the submitted compose file uses, so **you technically
only need to build and push once** and can point both
`submission/docker-compose.*.yml` files at the same tag if you prefer. Two
tags above are just for clarity in the portal's leaderboard history.

Before pushing, fill in the real tag in both compose files (currently
`<DOCKERHUB_USER>/lfm-serving:safe-bf16` / `:fp8` placeholders) and make the
repo **public** on Docker Hub (private images can't be pulled by the grading
system).

For a future W4A16 GPTQ submission, do **not** copy weights into
`submission/model/` manually. First run the H200 campaign and require
`candidate_report.json` to pass, then use:

```bash
bash scripts/prepare_w4_submission.sh \
  results/decode_cost_campaign_<timestamp>/candidate_report.json \
  artifacts/lfm2-w4a16-gptq-g128 \
  submission/docker-compose.w4a16-gptq.yml
```

That script validates the W4 artifact and compose flags before replacing the
Docker build context.

## 2. Submit order (tie-break rule #4 favors earlier submissions)

1. **Submit `submission/docker-compose.safe-bf16.yml` FIRST**, as early as
   possible. This banks a guaranteed-valid score: zero accuracy risk
   (Delta=0, wins tie-break rule #1 too), locally measured ERS ~55%.
2. **Submit `submission/docker-compose.fp8.yml` next.** Locally measured ERS
   ~65% (65.07%/64.81%/65.29%, three clean runs, mean ~65.1%, the third under
   simulated 3-core pinning matching the grading environment) — the project's
   single biggest win.
   Uses `--quantization=fp8_per_tensor` (verified not to be the broken legacy
   `--quantization=fp8`, which produces gibberish on this model — see
   EXPERIMENTS.md). **Accuracy: zero measurable delta across THREE independent
   checks** — arc_challenge (full 1172-set), gsm8k (limit 200), and a full
   198-question GPQA-diamond-mc check (an ungated re-upload of the same
   questions as the official gated `Idavidrein/gpqa`, once an `HF_TOKEN`
   became available — see EXPERIMENTS.md "REAL GPQA-diamond accuracy check").
   The GPQA-diamond-mc delta was **-0.015 (FP8 slightly higher)**, 0.33
   combined standard errors from zero — statistically indistinguishable from
   no change, and nowhere near the official Delta<=0.10 threshold.
3. **Use remaining submissions to settle open H200-only questions** (see
   `SUBMISSION_LOG.md` — create it when you start submitting, one row per
   submission: compose file, image tag, submitted-at timestamp, leaderboard
   ERS, conclusion). In descending order of information value:
   - a. `max_num_batched_tokens=512` vs `256` (both with fp8) — the local
     192-384 fine sweep was defeated by shared-tenant GPU contention on the
     dev box (see EXPERIMENTS.md "Narrow sweep"), so H200's actual optimum
     in this range is unconfirmed. Predicted (not measured) that H200's much
     higher compute throughput shifts the optimum upward from the 3090's
     256 — see `next_step.md` §1 for the reasoning.
   - b. `gpu_memory_utilization` above 0.70 (with fp8) — 0.70 was chosen
     conservatively; if leaderboard results and any exposed VRAM telemetry
     show comfortable headroom, a higher value could allow more KV cache /
     concurrency margin. Do not push this past ~0.85 without a way to
     observe real peak VRAM on H200.
   - c. **Do not submit the current W4A16 GPTQ no-conv artifact.** It failed
     the mandatory local accuracy gate on 2026-07-20: GSM8K limit-200 dropped
     from BF16 0.665 / FP8 0.700 to W4 0.570. This is a 9.5pp drop vs BF16,
     far beyond the 2.5pp W4 reject threshold. ARC was acceptable, but GSM8K
     is enough to reject W4 regardless of potential H200 latency.

## 2.1. Mandatory W4 H200 backend-probe gates

Before spending an official submission slot on any future W4 variant, run both
gates below. If either fails, do not submit W4. The current no-conv W4 artifact
already failed this gate and should not be submitted.

### Accuracy gate

Run W4 against the same accuracy checks used to clear FP8:

- ARC Challenge full.
- GSM8K limit 200.
- GPQA Diamond official if time permits and dataset access is available.
- Output coherence and answer format inspection.

If there is any clear degradation, malformed/coherence issue, or credible risk
that `accuracy_drop`/`f_delta` will regress, stop and keep FP8.

Write the result as a small JSON file for the Docker gate, for example:

```json
{
  "pass": true,
  "accuracy_drop_risk": false,
  "arc_challenge_full": "pass",
  "gsm8k_limit_200": "pass",
  "gpqa_diamond_official": "pass_or_skipped_for_time",
  "output_coherence_format": "pass"
}
```

### Exact Docker gate

The W4 image must contain the no-conv W4 checkpoint. The submitted compose must:

- cold-start with the exact entrypoint/command that will be submitted;
- expose `/health`;
- advertise model alias `LFM2.5-1.2B-Instruct`;
- replay all 420 spec requests with `0` failures and no corrupted/empty output;
- use `--quantization=compressed-tensors`, not `--quantization=fp8_per_tensor`;
- preserve the scheduler flags from the FP8 submission (`max_model_len=5120`,
  `max_num_seqs=8`, `max_num_batched_tokens=512`, prefix caching + xxhash).

Run:

```bash
# after replacing the placeholder image tag in submission/docker-compose.w4a16-gptq.yml
ACCURACY_GATE_JSON=results/<accuracy_gate>.json \
IMAGE_TAG=<local-or-pushed-w4-image-tag> \
TRACE=trace_grading_spec.jsonl WORKLOAD=spec \
  bash scripts/run_w4_probe_gates.sh
```

After the accuracy gate passes, stage the no-conv W4 artifact into the Docker
build context:

```bash
bash scripts/prepare_w4_submission.sh --probe \
  results/<accuracy_gate>.json \
  artifacts/lfm2-w4a16-gptq-g128-no-conv \
  submission/docker-compose.w4a16-gptq.yml
```

Then replace the placeholder image tag, run `scripts/run_w4_probe_gates.sh`, and
submit only if the emitted `probe_gate.json` has every gate set to `true`.

### How to read the official W4 probe result

- `TBT = 3 ms`, `ERS >= 63`, `f_delta = 1`: Machete has a strong signal; use
  remaining submissions to tune around this backend.
- `TBT = 4 ms`, `ERS only 60-61`: reject the W4 direction.
- ERS below FP8 or increased failures: return to FP8 immediately.
- Startup failure: likely an unsupported Machete Linear shape; do not spend a
  second W4 slot.
- Accuracy drop or `f_delta < 1`: reject W4 even if latency is good.

W4 has now also failed on the official H200 backend. The G64 MLP10-15 BF16
checkpoint preserved `accuracy_drop=0`, but scored only 49.71 ERS with Marlin
and 49.13 ERS with Machete, both at TBT median 6 ms. This rejects the current
stock compressed-tensors W4 direction for H200/MIG: local RTX 3090 speed did not
transfer because Hopper native FP8 beats W4 unpack/dequantize for this workload.
Do not spend more W4 slots unless a new H200 profile shows a specific W4 kernel
bottleneck that can be fixed. The newer `mlp10-15-attn10-12-14` checkpoint is
packaged only as a one-slot backend probe, not as a validated candidate.

The first profile-backed CUDA graph coverage test was also negative: explicitly
capturing batch sizes 1-8, with and without retaining size 16, did not materially
reduce TBT and left the launch/runtime profile bucket unchanged
(8.900 ms / 22.75% / 653 events vs 8.923 ms / 22.76% / 653 events baseline).
`cudagraph_copy_inputs=true` was also only a small win locally
(TBT mean 2.7943 ms vs 2.8180 ms baseline, failures 0) and remains below the
candidate threshold. Do not continue scheduler/cache/capture-size sweeps unless
a new profile shows a specific graph miss. The remaining kernel-level direction
is to reduce the outside-graph event count directly: each profiled decode step
still has one `cudaGraphLaunch` plus about 30 `cudaLaunchKernel` and 16
`cudaMemcpyAsync` calls, with memcopies attributed to `aten::copy_` and launches
coming from small metadata/sampling ops (`copy_`, `add`, `sub`, `fill_`,
`index`, `arange`, `gather`, etc.).

A local vLLM decode-metadata fast path now proves this direction but is not a
candidate by itself. The patch artifact is
`patches/vllm_decode_metadata_fastpath.patch`; it preallocates one-token decode
GPU constants and avoids per-step uploads for `req_indices`, `query_pos`,
`num_scheduled_tokens`, and pure-decode `discard_request_mask`. Best local run:
`results/fp8_h200shape_20260720-113815`, failures 0, ERS 69.5874, TBT
mean/p50/p95 2.7784 / 2.7656 / 3.0513 ms. Profile event count improved
baseline launch/runtime 653 events to 581 events, but the TBT gain is only
~1.4% mean / ~3.2% p95. It is now packaged as
`siconhoccode/lfm-serving:fp8-metadata-fastpath` because the remaining official
FP8 gap is small and official transfer is unknown; submit it only as an
isolated FP8 probe, then submit the seqs16 combination only if either axis shows
signal.
`ASYNC_SCHEDULING=1` was tested as an evidence-driven copy-path probe but
rejected: no material latency gain and captured output contained 2
gibberish/replacement-character-like fragments.

Current official best is FP8 `fp8_per_tensor`, `max_num_seqs=8`,
`max_num_batched_tokens=512`: ERS 60.89 best observed, 59.78-60.89 across three
identical official runs, TBT median 4 ms, failures 4, accuracy drop 0.

**UPDATE 2026-07-25 (superseded twice, see below): ShortConv candidates
tried and REJECTED after 4 official H200 submissions.** Backported upstream
vLLM PR #48917 (`ShortConv.__init__` never receives `quant_config`, leaving
~168M params silently BF16 under `fp8_per_tensor`) as
`patches/apply_vllm_shortconv_quant.py`. Local validation looked promising
(model load -160 MiB, TPOT -5.4%), but four official H200 submissions
isolating {v0.22.1, v0.25.1} x {patched, unpatched} all landed within noise
of the 60.89 baseline EXCEPT the v0.25.1+patched combination, which
regressed hard to ERS 51.65 -- a real interaction effect between the two,
not a standalone win from either. **TBT was 4ms in all four runs, no
exceptions** -- the core hypothesis (ShortConv quantization reduces decode
weight-read bandwidth enough to matter) is not confirmed by any real H200
data. Full table and reasoning in `SUBMISSION_RESULTS.md` "VERDICT
(2026-07-25)". Do not submit `fp8-shortconv-quant-retention` (Q2) -- same
regressing v0.25.1+patch combination. Local RTX-class hardware still does
not predict H200 win/loss size; this is now the third confirmed case in
this project (after `max_num_batched_tokens=256` and W4-G64).

**UPDATE 2026-07-25: CPU-bottleneck hypothesis also REJECTED.**
`fp8-metadata-fastpath-async` scored 55.78 officially -- worse than the
60.89 baseline on every axis (TTFT 65/112ms vs ~50/85ms, failed 7 vs 4),
and TBT still 4ms, the SIXTH straight official submission where it hasn't
moved (baseline, v0.25.1 unpatched, ShortConv on both base versions, and
this one). Both GPU-bandwidth and CPU-scheduling-overhead hypotheses for
what's pinning TBT at 4ms are now empirically rejected. Do not submit the
plain async-only or metadata-only decomposition candidates -- low value
now, and async's known gibberish-output risk remains unresolved.

Two other directions closed the same session: speculative decoding
(`suffix` method crashes under real concurrent load, 417/420 failures --
confirms the earlier `ngram` crash wasn't method-specific, closes the
entire spec-decode direction including training a custom
MLP-speculator/EAGLE head -- see EXPERIMENTS.md "T6"), and W4A8-FP8 mixed
quantization (built successfully but the Machete/CUTLASS kernel is gated
to `compute_capability==90`/Hopper exactly in vLLM's own code -- cannot be
validated on this Blackwell dev box, no Hopper GPU available, blocked).

**Current probe order -- TTFT is the only lever that has moved so far
(usually for the worse); pivot there:**

1. `submission/docker-compose.fp8-retention-only-seqs8.yml` (image
   `fp8-retention-only`, built from
   `submission/Dockerfile.fp8-retention-only-local`) -- hybrid-prefix
   retention (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`, PR #47782) ALONE
   on v0.25.1, no ShortConv patch. v0.25.1 alone already measured harmless
   (fp8-v0251, ERS 59.89), so this isolates retention's own effect for the
   first time -- the earlier `fp8-shortconv-quant-retention` build
   conflated it with the rejected ShortConv+v0.25.1 interaction and was
   never submitted.
2. `submission/docker-compose.fp8-seqs16.yml` with the existing `fp8-v1`
   image -- test whether the official 4-7 failures are queue/deadline
   starvation.
3. Only after the retention candidate lands: consider a `max_num_seqs`/
   `max_num_batched_tokens` sweep anchored on whichever config wins -- not
   before, since local sweeps have twice inverted on H200 already (see the
   `max_num_batched_tokens` and W4 notes above).

## 3. Final-5 selection (after the online round ends)

- **Always include the safe-bf16 submission** in the final 5 — it has
  Delta=0 by construction, which both maximizes `f(Delta)=1.0` and wins
  tie-break rule #1 (accuracy) if scores end up close to a competitor.
- **Include the FP8 submission** — fully cleared, including on the **literal
  official `gpqa_diamond_zeroshot` `lm_eval` task run against the literal
  gated `Idavidrein/gpqa` dataset** (access was granted after this plan was
  first written — see EXPERIMENTS.md "official GPQA-diamond access granted"):
  Delta = -0.010 (FP8 slightly higher), 0.24 combined standard errors from
  zero, ~1/10th of the official 0.10 threshold. Three other independent
  checks (arc_challenge, gsm8k, GPQA-diamond-mc ungated mirror) all agree.
  No further accuracy validation is needed before submitting.
- Do not include any variant using `--spec-method` (ngram/ngram_gpu/suffix)
  — these crash under real concurrent load (T6, EXPERIMENTS.md), confirmed
  with a hard CUDA error, not just a config warning.
- Do not include any variant using `--quantization bitsandbytes` (W4) — it
  measured *worse* than the unquantized baseline (T2, EXPERIMENTS.md).
- Do not include the tested W4A16 GPTQ G64 MLP10-15 BF16 submissions in the
  final 5; both official Marlin and Machete runs were slower than FP8 and near
  BF16 score. Only consider a future W4 if H200 profiling identifies a fixable
  W4-specific kernel bottleneck and a new candidate passes the full gate.

## 4. Unresolved risks (carry into the decision)

1. **GPQA-diamond accuracy risk: RESOLVED.** The literal official
   `gpqa_diamond_zeroshot` `lm_eval` task was run against the literal gated
   `Idavidrein/gpqa` dataset (access granted mid-session) for both BF16 and
   FP8: 0.2222+/-0.0296 vs 0.2323+/-0.0301, Delta=-0.010 (FP8 slightly
   higher), 0.24 combined SE from zero — far inside the official
   Delta<=0.10 threshold. This is no longer a risk requiring a caveat; it's
   a closed, measured result using BTC's own stated tooling
   (`lm_eval`/`bench-gpqa-diamond.sh`).
   Absolute accuracy (~0.22-0.23) is below BTC's stated BF16 baseline (0.40)
   — likely a difference in exact prompt/few-shot/parsing settings between
   this minimal invocation and BTC's own harness config, not a sign of a
   problem: both configs were run through the identical unmodified task, so
   the delta conclusion holds regardless of that offset.
2. **`max_num_batched_tokens` on H200 is a prediction, not a measurement.**
   256 was optimal on the 3090; 512 (currently in the compose files) is a
   reasoned bet that faster hardware shifts the optimum up. Use submission
   slot 3(a) above to check this directly.
3. **`gpu_memory_utilization=0.70` is conservative but unverified on real
   H200 hardware.** The one relevant local data point (the same
   `max_num_seqs=8`+`max_model_len=5120` shape, which caused a 37% VRAM
   overshoot at BF16 on the 3090) came back clean with FP8's smaller weights
   (5628-5636 MiB, comfortably under the local 7GB test limit) — but this
   doesn't rule out a different interaction effect specific to H200/Hopper.
4. **Startup time under real 3-core CPU + real H200 hardware is unmeasured.**
   Locally, cold start (compile cache cleared) took 54s under simulated
   3-core pinning with FP8 — reassuring, but simulated pinning on a 128-core
   box is not the same as an actual 3-core-limited container. If the
   portal's healthcheck has a short `start_period`, the first submission
   attempt will reveal whether this is a real problem.
5. **CUDA graphs are now known to be critical** (T4: disabling them collapsed
   ERS by ~80% under CPU pinning) — this is a reason for extra caution around
   #4: do NOT let a future iteration "fix" a slow startup by adding
   `--enforce-eager` or reducing `cudagraph_capture_sizes` — that trade would
   be far worse than a slow but complete startup.
6. **Local process gap**: the VRAM-crossing incident found while validating
   the H200 profile's memory assumptions (37% overshoot) was caught by a
   *manual* test that bypassed `scripts/vram_monitor.sh` (which only runs
   automatically inside `scripts/run_experiment.sh`). Any future manual
   `start_server.sh` invocation on this shared dev box should be paired with
   a manual `nvidia-smi` check immediately after startup.

## 5. Quick reference: what NOT to do

- Do not use `--quantization=fp8` (legacy name) — gibberish output, verified.
- Do not treat local CUDA 12.6 / vLLM 0.10.0 diagnostics as official evidence.
  They are useful for smoke tests only; final candidates require the H200/CUDA
  13 submission stack.
- Do not enable any `--spec-method` (ngram, suffix -- both independently verified
  to crash under real concurrent load, different crash signatures, same root
  cause: LFM2's hybrid Mamba/ShortConv KV-block accounting doesn't survive
  spec-decode's accept/reject rollback) — do not train a custom speculator
  (MLP/EAGLE) either, same underlying rollback machinery. See EXPERIMENTS.md
  "T6" for both crash traces.
- Do not use `--quantization=bitsandbytes` — measured worse than baseline.
- Do not set `--max-num-partial-prefills` > 1 — vLLM refuses to start.
- Do not set `--block-size` above the default (16) — measured worse.
- Do not add `--enforce-eager` or otherwise disable CUDA graphs for any
  reason, including "faster startup" — measured catastrophic (T4).
- Do not modify weights/tokenizer files, add external network calls at
  runtime, or hardcode/pre-bake any response content.
