# 2-Day Execution Plan (v2) — LFM2.5-1.2B serving optimization

> Status update, 2026-07-21: this plan is historical. Official H200 submissions
> now show FP8 `fp8_per_tensor` as the best reliable path: ERS 59.78-60.89
> across repeats, TBT median 4 ms, 4 failures, accuracy drop 0. Stock W4A16
> compressed-tensors with both Marlin and Machete scored around 49 ERS with TBT
> median 6 ms and is rejected for H200/MIG despite good RTX 3090 local latency.
>
> **UPDATE 2026-07-25 (superseded same day): ShortConv candidates (Q1/Q2)
> tried and REJECTED.** Backported upstream PR #48917
> (`patches/apply_vllm_shortconv_quant.py`, ShortConv layers were silently
> unquantized under `fp8_per_tensor`) -- looked promising locally, but 4
> official H200 submissions isolating {v0.22.1, v0.25.1} x {patched,
> unpatched} showed TBT unchanged at 4ms in ALL FOUR runs, and only the
> v0.25.1+patched combination regressed (ERS 51.65 vs 60.89 baseline) -- a
> real interaction effect, not a win from either half alone. Full data in
> `SUBMISSION_RESULTS.md` "VERDICT (2026-07-25)". Do not submit
> `fp8-shortconv-quant-retention` -- same regressing combination.
>
> **New working hypothesis (2026-07-25)**: TBT median was 4ms in ALL FOUR
> ShortConv-verdict submissions, unmoved by quantizing ~168M more params.
> Rules out GPU compute/bandwidth as the real H200/MIG bottleneck -- most
> likely culprit is CPU-side per-step overhead under a constrained vCPU
> allocation (vLLM V1's 3 competing processes: API server, EngineCore, GPU
> worker). Current action queue (pivoted back to `fp8-v1`, no ShortConv):
>
> 1. Submit `submission/docker-compose.fp8-metadata-fastpath-async-seqs8.yml`
>    (image `fp8-metadata-fastpath-async`) FIRST -- stacks the decode
>    metadata fast-path patch + `--async-scheduling`, both attacking the
>    CPU-bottleneck hypothesis. **Known risk**: a prior local probe on
>    `--async-scheduling` found 2/420 gibberish outputs (see "Async
>    scheduling probe" in `EXPERIMENTS.md`, session 10) -- re-verify output
>    coherence before trusting an ERS win here.
> 2. Submit `submission/docker-compose.fp8-async-sync-seqs8.yml` (explicit
>    `--no-async-scheduling` control) and
>    `submission/docker-compose.fp8-async-seqs8.yml` (`--async-scheduling`
>    alone) to decompose item 1's result if needed -- both reuse the
>    existing `fp8-v1` image, no build needed.
> 3. Submit `submission/docker-compose.fp8-seqs16.yml` using existing
>    `siconhoccode/lfm-serving:fp8-v1`.
> 4. Build/push `siconhoccode/lfm-serving:fp8-metadata-fastpath` with
>    `submission/Dockerfile.fp8-metadata-fastpath-local`, then submit
>    `submission/docker-compose.fp8-metadata-fastpath-seqs8.yml`.
> 4. Submit `submission/docker-compose.fp8-metadata-fastpath-seqs16.yml` only if
>    either axis improves official failures/ERS.
>
> Local CUDA diagnostics are smoke/correctness evidence only. Do not resurrect
> BitsAndBytes W4, scheduler/cache sweeps, or stock W4 compressed-tensors unless
> new H200 profiler evidence identifies a specific fixable bottleneck.

Work order for the executing agent. Self-contained: read top-to-bottom before running
anything. Supersedes v1 of this file. Prior findings live in `EXPERIMENTS.md` — read
"Notes and decisions", "GPU contention", "H200 submission profile" first.

---

## 0. Pre-flight checklist (10 min)

- [ ] `source /workspace/lfm-serving/.venv/bin/activate` (isolated; NEVER touch `/venv/main`)
- [ ] `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` — note
      other tenant's PID(s); they are OFF-LIMITS (observe only, never kill/renice)
- [ ] Read `EXPERIMENTS.md` sections listed above
- [ ] `df -h /` (need ≥5GB free), `free -h`
- [ ] Confirm harness works: `bash -n scripts/*.sh && python3 -m py_compile benchmark/*.py scripts/*.py`
- [ ] Read the cached model card eval file:
      `cat /workspace/.hf_home/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/*/.eval_results/gpqa.yaml 2>/dev/null` or the blobs dir —
      this is likely the source of BTC's baseline GPQA=0.40; record what Liquid reports.

---

## 1. Mission model: where the ERS points actually are (read this, it drives everything)

Official scoring (confirmed exact match with our `benchmark/compute_ers.py`):

```
s(x; F, C) = clamp((C - x)/(C - F), 0, 1)^2      # gamma = 2
S_request  = 0.5*s(TTFT; 10ms, 400ms) + 0.5*s(TPOT_mean; 1ms, 10ms)
ERS        = mean over 330 scored requests; errors/timeout/0-token = 0
Final      = 100 * ERS * f(GPQA_delta)           # f=1.0 while delta <= 0.10, baseline 0.40
```

### Current standing (RTX 3090, clean runs, configs/best.env)
ERS 55.18% ± 0.10pp. Decomposition: TTFT mean 89ms → s_ttft ≈ 0.64; TPOT mean 4.09ms
→ s_tpot ≈ 0.43. **TPOT is the weaker half and the bigger lever.**

### Grading-hardware Fermi model (MiG H200 1g.18gb: ~1/7 compute ≈ 140 TFLOPS BF16,
### ~1/7–1/8 bandwidth ≈ 600–685 GB/s — NOTE: *less* bandwidth than our 3090's 936 GB/s)

Decode at effective batch 1–4 is weight-read bandwidth-bound:
`TPOT_floor ≈ weight_bytes / BW + per-step CPU/launch overhead`

| Config | weight bytes | TPOT est (ms) | s_tpot | TTFT est* | s_ttft | **ERS proj** |
|---|---|---|---|---|---|---|
| H0: BF16 (safe) | 2.34 GB | ~3.9 | 0.46 | ~0.75 mix | 0.75 | **~0.60** |
| H1: +FP8 W8A8 | 1.17 GB | ~2.1 | 0.77 | 0.75 | 0.75 | **~0.76** |
| H2: +W4 online | ~0.6 GB | ~1.3 | 0.93 | 0.75 | 0.75 | **~0.84** |
| H3: H1/H2 + working spec-decode (2x accept) | — | ~0.7–1.1 | ~1.0 | 0.75 | 0.75 | **~0.87** |
| H4: everything + TTFT tail work | — | — | ~1.0 | 0.80 | 0.80 | **~0.90** |

\* TTFT mix: ~1/6 requests are turn-0 (full ~4k prefill ≈ 140–180ms → s≈0.38); ~5/6 are
prefix-cache hits (tokenize+hash+suffix ≈ 30–60ms → s≈0.8+). Weighted ≈ 0.75.

These are HYPOTHESES with ±30% error bars, not promises. The point: **quantization of
weights is worth ~16–24 ERS points on the grading hardware; nothing else on the list
comes close.** Everything in this plan is sequenced around confirming/denying that
safely (accuracy gate) plus defending TTFT and the 3-CPU-core constraint.
Update this table with real numbers as data arrives (first leaderboard result
calibrates the whole column).

---

## 2. Ground rules (unchanged from v1 — condensed)

1. Never touch other tenants' processes. Observe only.
2. **≤7GB VRAM** for our processes on this box. Manual (non-harness) server starts:
   check `nvidia-smi` immediately after startup; kill OUR server if >7000 MiB.
   Known trap: `max_num_seqs` and `max_model_len` raised *together* cost +37% memory.
3. Contention protocol: `bash scripts/wait_for_idle_gpu.sh 180 5` before benchmark runs;
   check `CONTENDED` marker after; mark `INVALID`, still log to EXPERIMENTS.md; trust
   only >10pp differences from contended data; never wait >30 min — use buffer blocks.
4. One variable at a time; every run logged via `scripts/log_experiment.py`.
5. Never delete caches/venv/model/results; `configs/best.env` frozen.
6. No docker build/compose on this box. Packaging = files only.
7. Compliance: **online quantization only** (quantize at model load from the BF16
   checkpoint via vLLM flags — shipping pre-modified weights risks the "no weight
   tampering" rule). No `ignore_eos` or output-modifying flags in any server config.
   Submission runtime must make zero external network calls (`HF_HUB_OFFLINE=1`,
   `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`).
8. Time-box every debugging rabbit hole to 30 min, then execute the documented backup.

---

## 3. Technique portfolio (each: hypothesis → experiment → accept/reject → backup)

### T1. FP8 W8A8 online quantization — `--quantization fp8`  [TOP PRIORITY]
- **Hypothesis**: TPOT 3.9→~2.1ms on H200 (halved weight reads) → +14–16 ERS pp.
  On the 3090 (SM86) vLLM falls back to Marlin W8A16 (verified: fp8
  `get_min_capability()==75`) — still bandwidth-halving, so local TPOT should drop
  measurably (~4.1 → ~2.5–3.0ms expected); treat local gain as a *lower bound*.
- **Prior**: high confidence mechanism (decode = weight-bandwidth-bound at batch≈1;
  see NVIDIA FP8 paper, Micikevicius et al. 2022; SmoothQuant ICML'23 shows W8A8
  accuracy is near-lossless at 8-bit). Main risk is architectural: LFM2's 10 conv
  layers may not have a quant path in vLLM 0.22.1.
- **Experiment**: smoke start (`QUANTIZATION=fp8` env once wired, §Day1-B) → verify
  coherent output + VRAM drop (~2.2GB→~1.2GB weights) + grep log for which layers got
  marlin/fp8 kernels → full trace run → accuracy eval (§8).
- **Accept**: server starts; output coherent; clean-run ERS ≥ best.env−1pp locally
  (any TPOT improvement is a bonus locally; H200 is where it pays); accuracy gate §8.
- **Failure modes & diagnosis**:
  - Load-time traceback mentioning conv/`ShortConv`/unsupported layer → LFM2 quant
    path missing. Try `--quantization-config` with an ignore/skip pattern for conv
    layers if 0.22.1 supports it (check `--help` + `vllm/model_executor/layers/quantization/fp8.py`
    `ignored_layers`); 30-min box.
  - Gibberish output → quantized conv/embedding numerically broken → same ignore-list
    fix, else reject T1.
  - Local TPOT does NOT improve → check log: are linear layers actually on marlin
    kernels, or did it silently fall back to unquantized? (`grep -i marlin` server log).
- **Backup**: T3 (KV fp8) alone + T4/T5; and note in SUBMISSION_PLAN that a
  W8A8-on-Hopper-only path may still work on H200 even if the Ampere fallback path
  fails locally (different kernels) — a leaderboard submission can test what local
  hardware can't. Mark that variant "untested locally" if used.

### T2. W4 weight-only online quantization — torchao int4 (else bitsandbytes NF4)
- **Hypothesis**: weight reads → ~0.6GB → TPOT ~1.3ms on H200 → +7–9 ERS pp *on top of*
  FP8. The single biggest possible jump after T1.
- **Prior**: medium-low. W4 groupwise (AWQ/GPTQ-style, MLSys'24 best paper AWQ; GPTQ
  ICLR'23) loses ~1–3pp on reasoning benchmarks at 1.2B scale — the GPQA gate
  (Δ≤0.10 vs baseline 0.40) has room, but a 1.2B model is more fragile than 7B+.
  Kernel path matters: naive dequant kernels can *lose* speed; marlin/tinygemm-class
  kernels (Marlin repo, IST-DASLab; QServe MLSys'24) are needed for real gains.
- **Experiment**: FIRST check what 0.22.1 actually supports online:
  `python3 -c "from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS; print(QUANTIZATION_METHODS)"`
  Look for `torchao`, `bitsandbytes`, `gptq`* (gptq needs offline ckpt — not allowed),
  `awq` (offline — not allowed). Then: torchao int4wo if present (may need
  `--quantization torchao` + config JSON); else bitsandbytes
  (`--quantization bitsandbytes`, online NF4). Smoke → coherence → trace run →
  accuracy eval (§8, stricter threshold).
- **Accept**: TPOT improves ≥15% locally vs T1 AND accuracy passes §8 W4 threshold.
- **Failure/diagnosis**: bnb frequently *slower* in vLLM (dequant overhead, no fused
  gemv for this arch) → if TPOT worsens locally, it will also worsen on H200 → reject
  cleanly, do not submit "hoping H200 differs" (unlike T1, bnb kernels don't change
  class on Hopper).
- **Backup**: none needed — T1 remains the quant ceiling. Do NOT pivot to offline
  AWQ/GPTQ checkpoints (compliance rule 7).

### T3. KV-cache FP8 — `--kv-cache-dtype fp8`
- **Hypothesis**: small TPOT gain (KV is tiny here: only 6/16 layers are attention,
  GQA 8 KV-heads ≈ 12KB/token BF16) but reduces attention reads at 4k context;
  worth +0.5–1.5 ERS pp, near-zero accuracy cost.
- **Experiment**: separate config, smoke (watch FLASH_ATTN-on-Ampere compat; try
  `fp8_e5m2` if plain `fp8` rejected) → trace run. **Watch for conflict with mamba
  'align' prefix-caching mode** — if startup asserts, prefix caching wins (55% vs 28%
  ERS), reject T3.
- **Accept**: no startup conflict, ERS ≥ baseline−0.5pp locally (its real payoff is
  H200), coherence OK.

### T4. CUDA-graph / compile tuning — decode CPU-overhead reduction  [3-CPU-CORE CRITICAL]
- **Hypothesis**: on 3 CPU cores, per-step CPU work (scheduling, launch, sampling
  bookkeeping) becomes a bigger TPOT term than on our 128-core box. Full CUDA graphs +
  higher compile effort shave 0.2–0.8ms/step there → up to +3–6 ERS pp on H200,
  ~neutral locally.
- **Facts from logs**: default is already `FULL_AND_PIECEWISE`, capture sizes [1,2,4,8],
  async scheduling ON, `-O2`-equivalent compile. Knobs: `-O3` / `--compilation-config`
  (mode, `compile_sizes`, `cudagraph_capture_sizes`, `max_cudagraph_capture_size`) —
  check exact 0.22.1 CLI spelling via `--help` first.
- **Experiment (do under 3-core pinning, else effects are invisible)**:
  1. **TPOT decomposition diagnostic** (30 min, high info): single isolated request,
     measure TPOT with (a) default config, (b) `--enforce-eager`. The (b)−(a) gap ≈
     CUDA-graph benefit = current CPU/launch overhead per step. If gap under 3-core
     pinning is ≥1ms, T4 has real headroom; if <0.3ms, deprioritize T4.
  2. A/B: default vs `-O3` (or compilation-config mode max) vs
     `compile_sizes=[1,2,4,8]`. One variable per run.
- **Accept**: ≥1pp clean ERS or ≥0.3ms TPOT improvement under 3-core pinning.
- **Failure/diagnosis**: `-O3` may blow up startup time (see T8) — record startup
  seconds for every variant; a config that's +1pp ERS but +5min startup may fail
  BTC's healthcheck → startup time is a first-class metric.

### T5. CPU-overhead pack (3 cores): xxhash, logs off, process/tokenizer tuning
- **Hypothesis**: sha256 block-hashing of ~4k-token prompts, uvicorn access logs, and
  stats logging each cost CPU that is free on 128 cores and expensive on 3. Combined
  +1–3 ERS pp on H200.
- **Experiment**: under `taskset -c 0-2` (via `CPU_PIN` env, §Day1-B): baseline vs
  +xxhash vs +logs-off vs both (4 runs). Additionally inspect
  `vllm/tokenizers/hf.py: maybe_make_thread_pool` — if a tokenizer thread-pool env/arg
  exists, test 1 vs 2 threads under pinning.
- **Accept**: any measurable clean-run gain under pinning → adopt in H200 profile
  (already staged there; this experiment finally *measures* them).
- **Note**: also record whether pinning by itself degrades TTFT/TPOT vs unpinned —
  that calibrates how much worse H200's 3 cores are than our local numbers.

### T6. Speculative decoding: `ngram`, `ngram_gpu`, `suffix`  [HIGH VARIANCE]
- **Hypothesis**: output tokens often copy spans from a 4k-token context
  (document-QA-style workloads) → prompt-lookup acceptance 1.5–2.5x → TPOT /1.5–2.5,
  lossless (spec decode preserves the output distribution; Leviathan ICML'23). GPU is
  idle ~83% of the time in this trace (avg concurrency 1.17) → verification compute is
  nearly free. Potential +5–10 ERS pp *if it runs on this architecture*.
- **Known unknowns**: (a) hybrid conv/mamba models need state rewind on rejection —
  vLLM support for LFM2 is unverified; 0.22.1's spec-method list includes `ngram`,
  `ngram_gpu`, `suffix` (Snowflake SuffixDecoding, ArcticInference repo). (b) spec
  decode may force async scheduling OFF (check startup log line!) — on 3 cores that
  could net negative. (c) our synthetic filler prompts are random-ish words →
  ngram acceptance locally will *understate* real-trace acceptance; a local tie is
  promising, a local loss ≈ its true overhead cost.
- **Experiment**: read `--help` for exact syntax (`--spec-method ngram --spec-tokens 4`
  or `--speculative-config '{"method":"ngram","num_speculative_tokens":4,...}'`).
  Smoke on top of best.env → if it starts: coherence check, grep logs for
  acceptance-rate/async-scheduling lines, trace run. Try `ngram_gpu` too (moves the
  ngram matching to GPU — relevant with 3 CPU cores). Then `suffix` once.
- **Accept**: clean ERS ≥ baseline (a tie is enough, per (c) above) AND async
  scheduling still on (or net ERS still ≥ baseline if off).
- **Failure/diagnosis**: startup assertion mentioning hybrid/mamba/spec → unsupported,
  record verbatim, drop T6 entirely (expected probability ~50%). Degenerate/looping
  output → sampler interaction bug → drop.

### T7. Scheduler micro-tuning: `--max-num-partial-prefills`, `--block-size`
- **Hypothesis A** (partial prefills): with budget 256 and Poisson bursts (peak
  concurrency 7), two arriving prefills serialize → the 2nd request's TTFT eats the
  1st's full prefill. `--max-num-partial-prefills 2` interleaves them → p95 TTFT drops
  (tie-break metric #2!), mean roughly neutral. +0.5–1.5 ERS pp and a fatter p95 win.
- **Hypothesis B** (block size): default 16; block 32 halves the number of blocks to
  hash/manage (CPU win on 3 cores) but coarsens prefix-cache hit granularity
  (negligible here — shared prefixes are thousands of tokens). Small CPU-side win.
- **Experiment**: A/B each against best.env (partial-prefills run also under 3-core
  pinning); check `--long-prefill-token-threshold` in help (may gate hypothesis A).
- **Accept**: A: p95 TTFT improves ≥15% with mean ERS not worse. B: any improvement
  under pinning.

### T8. Startup-time risk (BTC healthcheck)  [RISK MITIGATION, NOT A GAIN]
- **Problem**: on H200 the image starts cold: torch.compile + CUDA graph capture on
  **3 CPU cores**. Locally (128 cores) compile took ~15s fresh; on 3 cores it could be
  minutes. If BTC's healthcheck times out before "Application startup complete",
  the submission scores 0 regardless of config quality. Compile cache CANNOT be baked
  in the image (arch-specific; build machine ≠ H200).
- **Experiment**: measure wall-clock from process start → "Application startup
  complete" under `taskset -c 0-2` with a cold compile cache
  (`rm -rf ~/.cache/vllm/torch_compile_cache` — ONLY this cache, nothing else), for:
  default config, `-O3` (if T4 adopts it), and a "fast-boot" variant (lowest compile
  mode that keeps cudagraphs, e.g. `-O1` — check help).
- **Deliverable**: startup-seconds table in EXPERIMENTS.md + a `fast-boot` compose
  variant on standby; SUBMISSION_PLAN must state expected cold-start time so the human
  can compare against portal behavior on the first submission.

### T9. Leaderboard A/B (the only clean H200 instrument) — planning only here
Every submission = one clean benchmark on true hardware. Reserve the *decisions that
only H200 can settle*: batchtok 256-vs-512(-vs-1024 — prefill is faster there, sweet
spot plausibly shifts up), FP8 on/off, spec on/off. Protocol in §9.

### Explicitly rejected (do not spend time)
- Offline quant checkpoints (AWQ/GPTQ/llm-compressor): compliance risk (rule 7).
- EAGLE/Medusa/draft-model spec decode: requires trained draft weights that don't
  exist for LFM2.5; training them = new weights (compliance + time).
- Custom CUDA/Triton kernels, vLLM source patches: post-review risk + low ROI vs T1/T2.
- Semantic caching: output-modifying → "gaming" risk.
- CPU/NVMe KV offload: VRAM is plentiful; pure loss.
- `max_model_len`/`max_num_seqs` re-sweeps locally: settled (see EXPERIMENTS.md).

---

## 4. Day 1 schedule (~10h)

**Block A (1.5h) — Mandatory packaging (unchanged from v1 plan):**
Stage model (`cp -L` the snapshot → `submission/model/`, verify no symlinks,
~2.3GB safetensors + tokenizer/config/chat-template files). New `Dockerfile`
(FROM vllm/vllm-openai:v0.22.1; pip install xxhash; COPY submission/model /model;
ENV HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1; NO entrypoint override).
Two compose files in `submission/` matching BTC's fixed-entrypoint format
(`python3 -m vllm.entrypoints.openai.api_server`, flags in `command:`, shm_size 2g):
`docker-compose.safe-bf16.yml` (flags = H200 profile from EXPERIMENTS.md:
maxlen 5120, util 0.70, seqs 8, batchtok 512, prefix caching, xxhash, logs off,
served-name LFM2.5-1.2B-Instruct) and `docker-compose.fp8.yml` (same +
`--quantization=fp8`, header-commented DO-NOT-SUBMIT-until-cleared). Validate the
safe-bf16 command *verbatim* locally (host 127.0.0.1, util 0.22) — one completion
must return served name `LFM2.5-1.2B-Instruct`. Update README submission section.

**Block B (1h) — Wire new knobs + 3-core emulation baseline:**
Extend `scripts/start_server.sh` with opt-in env vars (default empty = no change;
keep `bash -n` clean): `QUANTIZATION`, `KV_CACHE_DTYPE`, `SPEC_METHOD`/`SPEC_TOKENS`
(or `SPECULATIVE_CONFIG` raw JSON), `COMPILATION_ARGS`, `EXTRA_VLLM_ARGS`
(word-split, appended last), `CPU_PIN` (→ `exec taskset -c $CPU_PIN python ...`).
Write `scripts/rss_monitor.sh` (per-second sum of VmRSS over the server process tree
→ CSV). Then run T5 matrix under pinning: `cpu3_base`, `cpu3_xxhash`, `cpu3_logsoff`,
`cpu3_opt(all)` — 4 trace runs with RSS monitoring on at least one. Record startup
seconds for each (T8 data starts here).

**Block C (3h) — Quantization ladder (T1 → T3 → T2):**
T1 fp8 smoke + trace run (+ log which kernels engaged). T3 kvfp8 smoke + trace run.
T2 support check (`QUANTIZATION_METHODS`) → torchao-or-bnb smoke + trace run if
available. Set up accuracy harness (§8): install lm-eval (separate uv venv if deps
clash with vllm — do NOT break `.venv`), check GPQA gating/HF token, run **BF16
reference** evals first (server from best.env) so every later config compares against
the same local reference.

**Block D (1.5h) — T4 compile/cudagraph under pinning:**
TPOT-decomposition diagnostic (enforce-eager gap) → then `-O3` / compile_sizes A/Bs
if the diagnostic shows ≥0.5ms headroom. Startup seconds recorded per variant (T8).

**Block E (1h) — T7 scheduler micro-tuning:**
partial-prefills 1 vs 2 (pinned, watch p95 TTFT specifically); block-size 16 vs 32.

**Block F (2h) — Buffer:** contention retries of anything marked CONTENDED, quant
accuracy evals for whichever of T1/T2/T3 passed smoke (start these even if Block C
ran long — accuracy results gate Day-2 combinations), EXPERIMENTS.md bookkeeping.

**End-of-Day-1 gate**: a written status table — for each of T1–T5, T7: tested? clean
ERS? adopted/rejected/pending-accuracy? This determines Day 2's combination matrix.

---

## 5. Day 2 schedule (~10h)

**Block G (2h) — T6 speculative decoding:**
ngram → ngram_gpu → suffix (smoke each; full trace run only for ones that start and
stay coherent). Check async-scheduling log line every time. If architecture-blocked:
document verbatim, move on early.

**Block H (2.5h) — Combination testing (greedy stacking):**
Start from best.env; add adopted techniques one at a time in descending single-run
gain order (e.g. +fp8 → +kvfp8 → +compile → +spec → +partial-prefills). One trace run
per addition; drop any addition that regresses ≥1pp. Finish with 2 clean repeats of
the final stack (stability bar: ±1pp; best.env achieved ±0.1pp).
Also one run of the final stack under `CPU_PIN=0-2` + RSS monitor (grading-shape
rehearsal) and one cold-start timing of it (T8 final number).

**Block I (2h) — Accuracy finalization (§8):**
Same-settings evals for every candidate that changes numerics (fp8, kvfp8, W4, and
the final stack). Fill the accuracy table. Apply thresholds → mark each candidate
CLEARED / CONDITIONAL / REJECTED.

**Block J (1.5h) — H200 shape rehearsal & memory bounds:**
Reproduce the memory-interaction effect deliberately (watchdog on, util 0.22):
seqs 8 + maxlen 5120 → record exact VRAM peak (bounds the H200 util=0.70 assumption).
If Day-1/2 adopted new memory-affecting features (fp8 lowers weights; spec adds
buffers), re-measure the final stack's VRAM at util 0.22 and recompute the H200
headroom paragraph in EXPERIMENTS.md.

**Block K (1.5h) — Candidate matrix + submission artifacts:**
Final table in EXPERIMENTS.md ("Submission candidates"): C1 SAFE-BF16 (must exist,
accuracy risk zero), C2 best-cleared-quant, C3 aggressive stack (only
individually-validated parts). One ready compose per candidate under `submission/`
(same image tag placeholder). Repo hygiene pass (`bash -n`, `py_compile`, no orphan
processes, GPU idle, temp files gone).

**Block L (0.5h) — `SUBMISSION_PLAN.md` + final report:**
Human steps (docker build/push from their machine; submit C1 EARLY — tie-break #4
favors earlier submissions), leaderboard A/B order (§9), final-5 guidance
(always include C1: guaranteed f(Δ)=1.0 + accuracy tie-break #1 advantage),
open risks, cold-start expectation from T8.

---

## 6. Scenario playbook (pre-committed decisions)

- **S1 — fp8 fails to load (conv layers)**: capture traceback → try quantization-config
  ignore-list for conv/embedding (30-min box) → else: kvfp8+T4+T5 path locally, and
  note the option of ONE leaderboard probe of fp8 anyway (Hopper W8A8 kernels differ
  from the Ampere fallback that failed) marked "untested locally".
- **S2 — no online W4 in 0.22.1 / bnb slower**: drop W4 permanently; T1 is the quant
  ceiling. Do not switch to offline checkpoints.
- **S3 — spec decode architecture-blocked**: drop T6 (this is the ~50% expected case);
  reallocate Block G time to Block H repeats.
- **S4 — spec runs but disables async scheduling**: keep only if net ERS ≥ baseline
  under 3-core pinning (that's where async loss hurts most).
- **S5 — GPQA gated & no HF_TOKEN**: proxy chain (§8). Mark quant candidates
  CONDITIONAL, not CLEARED; SUBMISSION_PLAN must tell the human to either provide a
  token for a local GPQA pass or accept the risk consciously for non-C1 picks.
- **S6 — cold start under pinning > ~4 min**: promote the fast-boot variant (T8) to a
  first-class candidate; submit it before any slow-boot config; measure its ERS cost.
- **S7 — RSS > ~6.5GB in rehearsal**: bisect (tokenizer pool? compile workers —
  try `TORCHINDUCTOR_COMPILE_THREADS=2`? engine vs API split?); document floor; if
  irreducible >7.5GB, flag CRITICAL in SUBMISSION_PLAN (8GB grading limit).
- **S8 — GPU contention all day**: CPU-pinned A/Bs still yield *relative* signal within
  the same window; fine ERS comparisons move to the leaderboard (T9); never block
  >30 min on waiting.
- **S9 — a leaderboard submission fails healthcheck**: first suspect T8 (cold start);
  second: compose format drift from BTC sample; third: RAM OOM-kill (S7). The verbatim
  local command test (Block A) rules out flag errors in advance.

---

## 7. Error-analysis toolbox (use for every anomalous result)

- **Per-request forensics** (`results/<run>/results.jsonl`): group TTFT by `turn_idx`
  (turn-0 vs rest = prefix-cache effectiveness; hits should be ~2-4x faster); plot/p50
  vs p95 gap (bimodality = queueing spikes, not steady-state slowness); correlate slow
  requests' `scheduled_offset_ms` with overlapping requests (interference windows);
  `prompt_tokens_actual` sanity (~3985) — drift means prompt-builder bug.
- **Server log greps**: `Prefix cache hit rate` (expect →75%+ by trace end; low = cache
  broken/thrashing), `Avg generation throughput`, `Capturing CUDA graphs`,
  `Asynchronous scheduling` (presence/absence per config!), `JIT` warnings (kernel
  compile during serving = TPOT spikes), `Aborting`, `marlin|fp8|quant` lines
  (which kernels actually engaged), any `WARNING.*fall.*back`.
- **Contention audit**: `vram_samples.csv` column 5 (other-tenant MiB) over the run's
  timeline — a mid-run tenant spike explains a p95 blowup with a clean p50.
- **Decision noise bar**: clean-run σ ≈ 0.1pp (best.env, n=4). Adopt a change only on
  ≥1.5pp clean-pair difference (or consistent direction across 2 pairs). Contended
  runs: direction-only evidence, and only if >10pp.
- **VRAM anomalies**: compare `vram_peak_mib.txt` against the 5.6GB baseline; any
  jump >500 MiB from a "free" feature deserves a note (cf. the +37% interaction trap).

---

## 8. Accuracy-gate protocol (the thing that can zero a submission)

- Official: GPQA-diamond post-online, baseline 0.40, f=1.0 while Δ≤0.10 (score floor
  0.30), linear to 0 at Δ=0.16. BF16 = automatically safe (Δ=0).
- Local reference first: run the eval against **our BF16 server** with fixed settings;
  all quant configs compare against that local reference (relative Δ), never against
  the absolute 0.40 (different harness details shift absolutes).
- Tooling: `lm-evaluation-harness` (`lm_eval --model local-chat-completions
  --model_args model=...,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=4
  --tasks <task>`). List exact task names first (`lm_eval --tasks list | grep -i gpqa`).
- **GPQA is HF-gated and this box has no HF token** (verified in capabilities). If
  `HF_TOKEN` is present in env, use `gpqa_diamond_zeroshot` (198 q). Else proxy chain,
  identical settings across configs: `arc_challenge` (full) + `gsm8k` (`--limit 200`,
  math is the most quant-sensitive tripwire at this scale).
- **Thresholds (pre-committed)**: FP8/kvfp8: CLEARED if GPQA Δ_local ≤ 0.05, or (proxy)
  arc Δ ≤ 2pp AND gsm8k Δ ≤ 3pp → then CONDITIONAL. W4: GPQA Δ ≤ 0.04 or proxy
  arc ≤ 1.5pp AND gsm8k ≤ 2.5pp (stricter: W4 damage is less uniform). Anything worse:
  REJECTED for submission regardless of ERS.
- Record every eval (task, version, settings, raw accuracy, Δ) in an EXPERIMENTS.md
  "Accuracy evals" table.

---

## 9. Submission strategy & leaderboard A/B (for SUBMISSION_PLAN.md)

1. **Submit C1 SAFE-BF16 early** (tie-break #4 = earlier submission wins ties; C1 also
   holds tie-break #1, zero accuracy drop). This banks a guaranteed-valid score.
2. Each later submission answers ONE H200-only question, in order of information value:
   (a) batchtok 512 vs 256 (settles the biggest untested H200 assumption);
   (b) FP8 on/off (the +14pp hypothesis — the single most valuable datapoint);
   (c) best stack from Day 2.
   Maintain `SUBMISSION_LOG.md`: compose → image tag → submitted-at → leaderboard ERS
   → conclusion. Update the §1 projection table after each result.
3. **Final-5 selection**: always include C1. Include quant/stack variants only with
   their accuracy status (CLEARED/CONDITIONAL) written next to them; if the team can
   run a real GPQA before the deadline (token available), upgrade CONDITIONAL first.

---

## 10. References (grounding for the hypotheses)

Papers — scheduling/serving: Kwon et al., *PagedAttention/vLLM* (SOSP'23); Agrawal et
al., *Sarathi-Serve* (OSDI'24 — chunked-prefill token budget = our batchtok=256 win);
Zheng et al., *SGLang/RadixAttention* (NeurIPS'24 — prefix-cache framing); Ye et al.,
*FlashInfer* (MLSys'25 best paper); Shah et al., *FlashAttention-3* (2024).
Speculative: Leviathan et al. (ICML'23); Chen et al. (2023); Saxena, *Prompt Lookup
Decoding* (repo apoorvumang/prompt-lookup-decoding); Oliaro et al., *SuffixDecoding*
(2024; repo snowflakedb/ArcticInference); (rejected-for-compliance context: Cai et al.
*Medusa* ICML'24, Li et al. *EAGLE-1/2/3*).
Quantization: Micikevicius et al., *FP8 Formats* (2022); Xiao et al., *SmoothQuant*
(ICML'23); Lin et al., *AWQ* (MLSys'24 best paper); Frantar et al., *GPTQ* (ICLR'23);
Frantar et al., *Marlin* kernels (repo IST-DASLab/marlin); Lin et al., *QServe W4A8KV4*
(2024); Dettmers et al., *LLM.int8* (NeurIPS'22).
Repos: vllm-project/vllm; sgl-project/sglang; flashinfer-ai/flashinfer; pytorch/ao;
vllm-project/llm-compressor (context only — offline path not allowed);
EleutherAI/lm-evaluation-harness; snowflakedb/ArcticInference.
Model: LiquidAI LFM2 blog + model card (hybrid 10-conv/6-attention, GQA 8 KV heads —
why KV is tiny, why spec/prefix-cache have mamba-align caveats).

---

## 11. Reporting requirements (end of each day)

Day-1 end: the T1–T5/T7 status table (§4 gate), all runs logged in EXPERIMENTS.md,
accuracy harness status (GPQA accessible or proxy mode), startup-seconds table draft.
Day-2 end: candidate matrix with ERS + accuracy status + VRAM + startup time per
candidate; updated §1 projection table; `SUBMISSION_PLAN.md` + `SUBMISSION_LOG.md`
skeleton; unresolved risks list; exact next human action. Every experiment — clean,
contended, failed, or abandoned — must appear in EXPERIMENTS.md with a decision.
