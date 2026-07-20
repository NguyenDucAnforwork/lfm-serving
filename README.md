# LFM2.5-1.2B-Instruct vLLM serving

Optimizing a vLLM OpenAI-compatible server for `LiquidAI/LFM2.5-1.2B-Instruct`
for a BTC competition: ERS (Efficiency/Response Score, from TTFT/TPOT) scored
online on a MiG H200 (18GB VRAM, 3 CPU cores, 8GB RAM), with a post-hoc GPQA
accuracy gate on up to 5 chosen submissions. See `EXPERIMENTS.md` for the full
experiment log; `next_step.md` for the technique portfolio and reasoning behind
what was tried; this file is the how-to-run guide.

**Current best result: ERS 65.07% / 64.81% / 65.29% (3 clean runs, mean
~65.1%) via FP8 online quantization** -- up from a 41.8% naive baseline and
55.2% after scheduler tuning alone. See "Key results" below.

## Layout

```
scripts/start_server.sh      # starts the vLLM server for LOCAL DEV/experiments only (see below)
scripts/run_experiment.sh    # orchestrates one experiment: start server, replay trace, log, stop
scripts/run_profile_capture.sh       # H200/CUDA-13 FP8 baseline profile capture
scripts/run_decode_cost_campaign.sh  # constrained H200 decode-cost campaign: profile FP8, test W4, report
scripts/run_accuracy.sh              # config-based GPQA mirror run with per-sample JSONL
scripts/vram_monitor.sh      # polls nvidia-smi for one PID + total GPU usage; kills server past the VRAM limit
scripts/rss_monitor.sh       # polls total RSS across the server's process tree (checks the 8GB grading RAM limit)
scripts/wait_for_idle_gpu.sh # waits (read-only) until no other tenant process is using the GPU
scripts/stop_server.sh       # stops any server this project started
scripts/log_experiment.py    # appends one row to EXPERIMENTS.md's summary table
scripts/quantize_w4a16_gptq.py       # exports W4A16 GPTQ compressed-tensors artifact
scripts/validate_w4_candidate.py     # checks W4 artifact/run constraints and Marlin/Machete backend evidence
scripts/validate_submission_compose.py # checks compose flags before packaging/submission
scripts/prepare_w4_submission.sh     # guarded W4 packaging after candidate_report PASS
benchmark/trace_utils.py     # loads the trace, generates token-length-matched synthetic prompts (strong/weak overlap modes)
benchmark/replay_trace.py    # replays the trace against a running server, records per-request metrics
benchmark/compute_ers.py     # ERS scoring model + summary stats (also importable as a library)
benchmark/summarize_run.py            # ERS/TBT/TTFT/throughput/failure summary JSON
benchmark/analyze_profile_trace.py    # Chrome trace bucket attribution: GEMM, FP8 scaling, conv/state, attention, launch
benchmark/analyze_failures.py         # failure clustering by type/turn/length/concurrency/time
benchmark/compare_accuracy.py         # exact per-sample accuracy diff grouping
benchmark/candidate_report.py         # final PASS vs FAIL/INCOMPLETE report for candidate gates
configs/*.env                # one file per experiment; consumed by start_server.sh
configs/best.env             # RTX-3090-validated winning config (BF16, no quantization)
configs/fp8_h200shape.env    # FP8 + exact H200 submission shape, locally validated (BEST)
configs/w4a16_gptq_machete.env # H200 W4A16 GPTQ candidate, forced Machete
configs/w4a16_gptq_marlin.env  # H200 W4A16 GPTQ fallback, forced Marlin
configs/local_cu126_*.env      # local-only CUDA 12.6/vLLM 0.10 diagnostics, not submission-equivalent
configs/submission_h200.env  # H200 candidate profile, annotated assumptions -- see EXPERIMENTS.md
results/<run>/               # results.jsonl, results.csv, summary.json, server.log, vram_samples.csv, other_tenant_peak_mib.txt, rss_peak_kb.txt per run
results/accuracy/             # lm-eval accuracy comparison outputs (BF16 vs FP8)
Dockerfile                    # submission image: bakes the model, installs xxhash, NO entrypoint override
submission/model/             # dereferenced model weights, baked into the image (not a symlink)
submission/docker-compose.safe-bf16.yml  # C1: submit first, zero accuracy risk
submission/docker-compose.fp8.yml        # C2: +9-10pp ERS, FULLY CLEARED (4 accuracy checks incl. the literal official GPQA-diamond gate)
submission/docker-compose.w4a16-gptq.yml # guarded W4 template; submit only after candidate_report PASS
next_step.md                  # 2-day technique portfolio / execution plan (source of the T1-T8 naming used throughout)
SUBMISSION_PLAN.md            # human next steps: build/push/submit, leaderboard A/B order, final-5 guidance
```

**Format note**: the official BTC portal's `docker-compose.yml` fixes the
entrypoint to `python3 -m vllm.entrypoints.openai.api_server`, with all vLLM
flags passed via `command:`. So `scripts/start_server.sh` (env-var driven) is
**dev-only** now -- the actual submission compose files under `submission/`
hardcode flags directly, matching the portal's required format.

## Environment

- Isolated venv at `.venv` (NOT the shared `/venv/main`). Two runtime tracks
  matter:
  - **Submission/H200 track:** `vllm/vllm-openai:v0.22.1` with CUDA 13, used
    by `Dockerfile` and the official FP8 submission. This is the authoritative
    target for any final candidate.
  - **Current local debug track:** this dev box has driver `560.35.03`
    (`nvidia-smi` CUDA 12.6), so `.venv` was adjusted in-place to
    `vllm==0.10.0+cu126`, `torch==2.7.1+cu126`, `transformers==4.57.6`.
    This lets the RTX 3090 run local smoke/diagnostic experiments, but it is
    **not submission-equivalent**: vLLM 0.10.0 lacks `fp8_per_tensor`,
    `--linear-backend`, `--profiler-config`, and `xxhash` prefix hashing, and
    LFM2 falls back to the Transformers backend. See `EXPERIMENTS.md`
    "Session 5".
- Model cache: `HF_HOME=/workspace/.hf_home` (shared HF cache on this box).
- This is a shared GPU instance; scripts avoid touching other users'
  processes and only manage PIDs they themselves started. **In practice this
  box has another tenant running near-continuous training/eval jobs**, which
  visibly degrades ERS measurements when it overlaps a benchmark run (down to
  ~24% vs ~55% clean, same config, no code change) -- `scripts/vram_monitor.sh`
  and `scripts/wait_for_idle_gpu.sh` exist specifically to detect and work
  around this; see EXPERIMENTS.md "GPU contention" for the full story. This
  does not apply to the final grading environment (an isolated MiG H200
  slice).

## Reproduce from scratch (bootstrap)

`.venv/`, `.venv-eval/`, and `submission/model/`'s weight files are **not**
committed (too large / regenerable) -- `.gitignore` excludes them. To rebuild
this environment on a fresh clone (CUDA-capable GPU, `uv` installed):

```bash
cd lfm-serving

# 1a. Main serving venv for the H200/submission-equivalent stack
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.22.1      # pins torch==2.11.0; installs matching CUDA build
uv pip install xxhash            # needed for PREFIX_CACHE_HASH_ALGO=xxhash (see EXPERIMENTS.md near-miss note)
# Optional, only needed to re-run the rejected T2/experimental quantization paths:
uv pip install bitsandbytes torchao

# 1b. If you are on this current RTX 3090 / driver 560 / CUDA 12.6 box instead,
#     do NOT create another venv; install a local-debug-compatible stack in .venv:
uv pip install --python .venv/bin/python \
  'https://github.com/vllm-project/vllm/releases/download/v0.10.0/vllm-0.10.0+cu126-cp38-abi3-manylinux1_x86_64.whl' \
  --extra-index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python 'transformers>=4.53.2,<5' datasets
# This local stack is only for smoke/diagnostics; do not use it as final H200 evidence.

# 2. Accuracy-eval venv (separate on purpose -- avoids any dependency clash with vLLM)
uv venv .venv-eval --python 3.12
source .venv-eval/bin/activate
uv pip install "lm-eval[api]" transformers datasets
# HF_TOKEN (env var or a local .env you keep OUT of git) is only needed for the
# gated gpqa_diamond_zeroshot check; every other benchmark/eval works without one.

# 3. Model weights, for local dev (server can auto-download from the HF Hub --
#    set HF_HOME to a writable cache dir first) or for building the submission
#    image (needs the real files on disk at submission/model/, see Dockerfile):
export HF_HOME=/path/to/writable/cache
python3 -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('LiquidAI/LFM2.5-1.2B-Instruct'))"
# then, to populate submission/model/ (dereferencing the cache's symlinks --
# the Dockerfile COPYs real files, not symlinks):
mkdir -p submission/model
cp -L "$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('LiquidAI/LFM2.5-1.2B-Instruct'))")"/* submission/model/

# 4. Sanity check
bash -n scripts/*.sh && python3 -m py_compile benchmark/*.py scripts/*.py && echo OK
bash scripts/start_server.sh   # from .venv; Ctrl+C to stop, or use scripts/stop_server.sh
```

`trace_grading_public.jsonl` (the public grading trace) **is** committed --
it's a required input, not regenerable, and small (~64KB).

## Local dev: start the server

```bash
source .venv/bin/activate
bash scripts/start_server.sh          # binds 127.0.0.1:8000 by default in dev
```

Config is env-var driven (see the top of `start_server.sh` for the full
list). Defaults already reflect the best known config from `EXPERIMENTS.md`
(`max_model_len=4608`, `gpu_memory_utilization=0.22`, `max_num_seqs=4`,
`max_num_batched_tokens=256`, prefix caching on). Override any of them, e.g.:

```bash
MAX_NUM_BATCHED_TOKENS=512 bash scripts/start_server.sh
```

Additional opt-in knobs (default off/unset -- no behavior change unless set;
see EXPERIMENTS.md "Additional low-risk vLLM tuning" for what each does and
why): `PERFORMANCE_MODE`, `PREFIX_CACHE_HASH_ALGO` (needs the `xxhash` pip
package if set to `xxhash`), `DISABLE_LOG_STATS`, `DISABLE_ACCESS_LOG`,
`SERVED_MODEL_NAME`.

Smoke test:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"LiquidAI/LFM2.5-1.2B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":20,"stream":true}'
```

## Running a benchmark / experiment

`scripts/run_experiment.sh <config_name>` starts the server from
`configs/<config_name>.env`, waits for readiness, monitors VRAM (auto-kills
the server if it crosses `VRAM_LIMIT_MIB`, default 7000 MiB), replays the
selected trace in full, stops the server, and writes everything to
`results/<config_name>_<timestamp>/`:

```bash
TRACE=trace_grading_spec.jsonl WORKLOAD=spec bash scripts/run_experiment.sh fp8_h200shape
python3 scripts/log_experiment.py results/baseline_<timestamp> --decision "KEEP — ..."
```

To just replay the trace against a server you started yourself:

```bash
python3 benchmark/replay_trace.py --trace trace_grading_public.jsonl --name my_run --verbose
python3 benchmark/compute_ers.py results/my_run/results.jsonl
```

When `--model` is a served alias, pass the tokenizer repo separately:

```bash
python3 benchmark/replay_trace.py \
  --trace trace_grading_spec.jsonl \
  --workload spec \
  --model LFM2.5-1.2B-Instruct \
  --tokenizer-model LiquidAI/LFM2.5-1.2B-Instruct \
  --name my_spec_run --verbose
```

## Decode-cost campaign / W4 candidate

The current official target is to reduce S2's TBT median from 4ms toward
3.0-3.5ms without increasing failures or accuracy drop. The prepared H200
campaign is intentionally mechanism-driven:

```bash
TRACE=trace_grading_spec.jsonl WORKLOAD=spec W4_RUNS=3 DO_ACCURACY=1 \
  bash scripts/run_decode_cost_campaign.sh
```

It profiles the matched FP8 baseline, validates/uses a W4A16 GPTQ
compressed-tensors artifact, tests forced Machete first with Marlin fallback,
aggregates 3 clean runs, runs GPQA mirror diffing, and writes
`candidate_report.{md,json}`. A W4 submission is allowed only if that report
says `PASS`; then use:

```bash
bash scripts/prepare_w4_submission.sh \
  results/decode_cost_campaign_<timestamp>/candidate_report.json \
  artifacts/lfm2-w4a16-gptq-g128 \
  submission/docker-compose.w4a16-gptq.yml
```

Do not use BitsAndBytes and do not combine W4 with `fp8_per_tensor`; the
validators reject both.

## How the benchmark works (and why)

**Two workloads.** `trace_grading_spec.jsonl` (generated by
`benchmark/gen_spec_trace.py` from the official `grading_workflow_spec.jsonl`,
18/07/2026) is the current authoritative shape; replay it with
`--workload spec`. `trace_grading_public.jsonl` is the older reverse-engineered
shape that all Session-1/2 numbers were measured on; replay it with the default
`--workload legacy`. See `EXPERIMENTS.md` "Session 3" for the full diff.

The traces carry only timing/length metadata (`conv_id`, `turn_idx`,
`timestamp_ms`, `think_ms`, `in_tokens_est`, `out_tokens_max`) — no real
prompt text, per the task constraints. `benchmark/trace_utils.py` generates
synthetic prompts that match the token-length metadata using the official
tokenizer.

- **`--workload spec` (official):** a real multi-turn *growing-context*
  structure — one 1000-token system prefix **byte-identical across all
  conversations** (a genuine global prefix-cache hit), a 1000-token per-conv
  prefix on turn 0, ~150 fresh user tokens per turn, output pinned at 300, and
  the model's own reply threaded back as the assistant turn so turn `t`'s
  prompt genuinely extends turn `t-1`. Input grows `2150 + 450*t` (2150→4400).
- **`--workload legacy` (old shape):** `in_tokens_est` stayed ~constant (~4000)
  across all 6 turns rather than growing — each turn re-sent a large shared
  context. Each conversation gets one long filler "context" (deterministic per
  `conv_id` given `--seed`) and each turn takes a token-exact prefix of it plus
  a short suffix, giving prefix-cache reuse across turns of the same
  conversation.

Scheduling follows the trace exactly: turn 0 of each conversation fires at
`benchmark_start + timestamp_ms` (Poisson arrival); turns 1-5 fire at
`(previous turn's completion time) + think_ms`. No client-side concurrency
cap is imposed — realistic overlap comes purely from arrivals + think_ms, and
server-side queuing shows up as added latency, which is exactly what should
be measured. `--ignore-eos` (on by default) forces generation to run to
exactly `out_tokens_max`, giving stable, comparable TTFT/TPOT across
different server configs instead of variance from the model's own stopping
behavior.

Warmup requests (`in_warmup==true`, the first 15 conversations) are sent
(to warm the server and its prefix cache) but excluded from ERS.

`--prefix-overlap {strong,weak}` (default `strong`) controls how cacheable
the synthetic prompts are: `strong` (used for all tuning decisions) shares
the whole body across turns of a conversation; `weak` only repeats a fixed
`--weak-prefix-tokens` anchor (default 256), used as a pessimism check that
the winning config isn't overfit to a best-case cache-hit rate -- see
EXPERIMENTS.md.

## ERS scoring

`benchmark/compute_ers.py` implements: TTFT floor 10ms/ceiling 400ms, TPOT
floor 1ms/ceiling 10ms, gamma=2, equal TTFT/TPOT weight, errors/timeouts/
zero-token outputs score 0. The exact published formula for combining
floor/ceiling/gamma wasn't given verbatim, only the parameters — we use the
standard `clamp((ceiling - x)/(ceiling - floor), 0, 1) ** gamma` shape, the
conventional way to turn those three numbers into a bounded score. This is
isolated in one function (`compute_ers.metric_score`) so it's a one-line
change if the official grader's formula differs.

## Key results

### Session 1: scheduler tuning (BF16, no quantization)

Best config found (`configs/best.env`): **ERS 55.2%** on the full public
trace (up from 41.8% naive baseline), TTFT mean/p95 89/241ms, TPOT mean/p95
4.1/4.9ms, 0% errors, ~5.6 GiB VRAM peak — a 32% relative ERS gain from one
scheduler knob (`max_num_batched_tokens=256` instead of vLLM's default 2048),
no quantization/custom kernels/speculative decoding.

**Reproducibility**: 4 independent clean runs of `best.env` (2 from initial
tuning, 2 from a later validation pass) give mean 55.18%, stdev 0.10pp, range
55.03-55.25%, 0 errors across 1320 requests -- the config's behavior is
effectively deterministic modulo normal run noise.

**Narrow sweep** (192/224/256/288/320/384 around the winner): clean
measurements at 192 (53.1%) and 512 (53.7%, from the original sweep) bracket
256 (55.2%) as a clear local optimum; finer-grained neighbors couldn't be
measured cleanly this session due to persistent shared-tenant GPU contention
(all readings, regardless of the specific value, landed at or below 256's
clean value, consistent with 256 remaining best, but too noisy to rank
precisely against each other). See EXPERIMENTS.md for the full table and
reasoning.

**Robustness**: verified with a real ~4000in/200out request, 6 concurrent
long requests (0 errors, 0 zero-token outputs, VRAM steady), 3 clean
restart cycles, and 6 classes of malformed/invalid requests (all correct
400/404s, no crashes). Context headroom at `max_model_len=4608` is ~9%
(423 tokens) over the worst-case ~4185-token need.

**Additional tuning explored**: `--performance-mode interactivity` tested
head-to-head, no measurable benefit (our `max_num_seqs=4` never triggers the
padding overhead it targets) -- not adopted. `--prefix-caching-hash-algo
xxhash` and `--disable-log-stats`/`--disable-uvicorn-access-log` reduce
CPU-side overhead relevant to the final environment's 3-CPU-core budget;
**catch worth knowing about**: `xxhash` requires the Python package, which
isn't installed by default -- omitting it makes every request 500. Fixed by
adding `RUN pip install xxhash` to the Dockerfile and confirmed working
end-to-end (not just via `--help`). Full reasoning for all of these in
EXPERIMENTS.md "Additional low-risk vLLM tuning explored".

**Prompt-overlap sensitivity check**: `benchmark/trace_utils.py` also
supports `--prefix-overlap weak` (only a 256-token anchor repeats across
turns instead of the whole ~4000-token body), used to check the winning
config isn't overfit to a best-case synthetic cache-hit rate. Both 256 and
512 collapse to similarly low ERS (~12-14%) under this pessimistic mode --
expected, since there's much more unavoidable prefill compute either way --
and neither config catastrophically underperforms the other, so the choice
of 256 carries little downside risk if the hidden trace turns out less
cacheable than assumed. We do not claim to know the hidden trace's actual
cache-hit rate.

Full writeup, the complete experiment table, and all reasoning are in
`EXPERIMENTS.md`.

### Session 2: official rules obtained, full technique portfolio (see `next_step.md`)

The official BTC rules confirmed our ERS formula exactly, revealed the submission
format (fixed-entrypoint compose, model baked into the image), and opened up
quantization/speculative decoding as explicitly allowed techniques. Executed
`next_step.md`'s T1-T8 portfolio:

- **T1 FP8 online quantization -- the project's biggest win, fully cleared
  on the literal official accuracy gate.** `--quantization fp8_per_tensor`
  (vLLM 0.22.1's newer online-quant framework; **not** the legacy
  `--quantization fp8`, which loads without error but produces gibberish
  output on this model -- caught by testing actual completions, not just
  server startup). **ERS 55.2% -> 65.1%** (mean of 3 clean runs on the exact
  H200 submission shape: 65.07/64.81/65.29%), TPOT mean cut ~30% (4.09ms ->
  2.85ms). **Zero measurable accuracy degradation across four independent
  checks**: arc_challenge full 1172-set (0.4258 -> 0.4266), gsm8k limit-200
  (0.66 -> 0.66 exact match), a 198-question GPQA-diamond-mc ungated-mirror
  check (Delta=-0.015), and -- after the HF account accepted the dataset's
  access terms mid-session -- **the literal official `gpqa_diamond_zeroshot`
  `lm_eval` task run against the literal gated `Idavidrein/gpqa` dataset**:
  0.2222 -> 0.2323 (Delta=-0.010, FP8 slightly *higher*, 0.24 combined
  standard errors from zero -- about 1/10th of the official 0.10 threshold).
  See EXPERIMENTS.md "official GPQA-diamond access granted" for the full
  writeup, including why the absolute numbers (~0.22-0.23) sit below BTC's
  stated 0.40 baseline (almost certainly a harness/prompt-setup difference,
  not a quantization problem -- both configs used the identical task).
- **T2 W4 (bitsandbytes NF4): rejected.** Loaded and ran coherently, but ERS
  29.3% -- *worse* than the unquantized baseline (no fused low-bit GEMV
  kernel for this model/size, so dequant overhead exceeds bandwidth savings).
- **T3 KV-cache FP8: neutral**, not adopted standalone (LFM2 has only 6/16
  attention layers with GQA -- KV cache is already small, quantizing it
  further doesn't move ERS).
- **T4 CUDA graphs: confirmed load-bearing, not tuned further.** Disabling
  them (`--enforce-eager`) under simulated 3-CPU-core pinning collapsed ERS
  from 64.9% to 13.7% (TPOT 4.4x worse). Default settings already capture
  graphs; no evidence a different capture-size list would help, and touching
  this for faster startup would be actively dangerous.
- **T5 CPU-overhead flags (xxhash hashing, disabled logging): adopted**,
  small but consistent positive signal (+0.68pp) under 3-core pinning; RSS
  unaffected (~3GB, well under the grading env's 8GB).
- **T6 Speculative decoding (ngram): rejected -- critical.** Passed a
  single-request smoke test, then **crashed with a CUDA "illegal memory
  access" error under real concurrent trace load**, in both BF16 and FP8
  configurations (84% and 100% request failure respectively). This is an
  architectural incompatibility between vLLM's spec-decode rollback and
  LFM2's hybrid conv/mamba layers, not a config mistake -- do not enable any
  `--spec-method` for this model. This is the clearest illustration in the
  whole project of why single-request smoke tests aren't sufficient and
  full-trace validation under real concurrency matters.
- **T7 Scheduler micro-tuning: both tested, both rejected.**
  `--max-num-partial-prefills 2` is flatly unsupported for this
  model/config (vLLM raises `NotImplementedError` at startup).
  `--block-size 32` measured 1.7pp worse than the default 16 (larger blocks
  reduce prefix-cache hit granularity for this workload's long shared
  prefixes).
- **T8 Cold-start risk: reassessed as low.** With the torch.compile cache
  forcibly cleared, cold start under 3-core pinning + FP8 took 54s, vs 43s
  unpinned/BF16 -- only +26%, not the feared "minutes" scenario. Combined
  with the T4 finding, no fast-boot/reduced-compilation fallback was built;
  recommend a generous (90-120s) healthcheck `start_period` instead.

**Final validated combined stack** (`configs/fp8_h200shape.env` / `fp8_h200shape_cpu3.env`
= the exact flags in `submission/docker-compose.fp8.yml`): fp8_per_tensor +
xxhash + disabled logging + `max_model_len=5120` + `max_num_seqs=8` +
`max_num_batched_tokens=512`. **ERS 65.07% / 64.81% / 65.29%** (3 clean runs,
mean ~65.1%, the third under simulated 3-core pinning -- the most
representative single measurement of the grading environment), 0 errors,
VRAM 5628-5638 MiB, RSS 2.93GB (well under the grading env's 8GB RAM). This
also resolved a session-1 memory concern: the same `max_num_seqs=8` +
`max_model_len=5120` combination at BF16 had used 37% more VRAM than naively
expected (7236 MiB) -- FP8's smaller weights brought that back down
comfortably.

Full reasoning, every experiment (including the crashes and rejections), and
the complete session-2 table are in `EXPERIMENTS.md` "Session 2".

### Session 4/5: decode-cost campaign prep and local CUDA 12.6 diagnostics

- Official baseline for the new decode-cost goal is `SUBMISSION_RESULTS.md`
  S2: FP8 per-tensor, ERS 60.20, TBT median 4ms, failed_count 4,
  accuracy_drop 0.
- Added H200 campaign tooling for profiler-backed W4A16 GPTQ
  compressed-tensors experiments and guarded submission packaging.
- Current dev host can run CUDA 12.6 locally with `vllm==0.10.0+cu126`, but
  that stack is not submission-equivalent. Local diagnostics on
  `trace_grading_spec.jsonl`:
  - BF16: ERS 55.51, TBT mean/p50/p95 5.34/5.17/7.36ms, 0 failures.
  - legacy `--quantization=fp8`: ERS 55.37, TBT mean/p50/p95
    5.45/5.08/7.86ms, 0 failures.
  Legacy FP8 is therefore rejected locally; the prepared H200 W4 campaign
  remains the next real decode-cost step.

## Submission packaging (prepared, not executed)

**The official BTC rules fix the submission format**: the portal's
`docker-compose.yml` hardcodes `entrypoint: [python3, -m,
vllm.entrypoints.openai.api_server]` and passes all vLLM flags via
`command:`. So our image does **not** set its own `ENTRYPOINT` —
`scripts/start_server.sh` is dev-only now (used for local experiments, never
in the container). Model weights must be **baked into the image** at
`/model` (external network calls at grading time are forbidden by the
anti-cheat rules, ruling out a runtime HF download).

- `Dockerfile`: `FROM vllm/vllm-openai:v0.22.1`, `RUN pip install xxhash`
  (see the near-miss note above), `COPY submission/model /model` (the exact
  official LiquidAI/LFM2.5-1.2B-Instruct files, dereferenced from the local
  HF cache's symlinks — no weight/tokenizer modification), and
  `HF_HUB_OFFLINE=1`/`VLLM_NO_USAGE_STATS=1`/`DO_NOT_TRACK=1` for the
  no-network-calls rule.
- `submission/docker-compose.safe-bf16.yml` — **C1, submit first & early**
  (tie-break rule #4 favors earlier submissions; zero accuracy risk also
  wins tie-break #1). Matches the BTC sample's exact structure. Flags =
  the H200 profile (`configs/submission_h200.env`): `max_model_len=5120`,
  `gpu_memory_utilization=0.70`, `max_num_seqs=8`,
  `max_num_batched_tokens=512`, prefix caching + xxhash, logs disabled.
  **Verified end-to-end on this dev box** (functional check, not a VRAM
  match — H200 is 18GB, this box 24GB): server starts in ~43s cold, serves
  the correct name `LFM2.5-1.2B-Instruct`, and answers correctly.
- `submission/docker-compose.fp8.yml` — **C2, the leading candidate, FULLY
  CLEARED** (see `EXPERIMENTS.md` "T1: FP8 online quantization" and "official
  GPQA-diamond access granted"): +9.3-9.9pp ERS, zero measurable accuracy
  delta across four independent checks, including the **literal official
  `gpqa_diamond_zeroshot` task on the literal gated `Idavidrein/gpqa`
  dataset** (Delta=-0.010, FP8 slightly higher, 0.24 SE from zero — ~1/10th
  of the official 0.10 threshold). Same as C1 plus
  **`--quantization=fp8_per_tensor`** (not the legacy `--quantization=fp8`,
  which produces gibberish on this model — see "Key results" above).
- `submission/docker-compose.w4a16-gptq.yml` — guarded W4A16 GPTQ
  compressed-tensors template only. Do not submit it until
  `benchmark/candidate_report.py` says PASS after 3 clean H200/spec runs,
  backend logs prove Machete or Marlin for `CompressedTensorsWNA16`, and
  accuracy diff is non-negative. Use `scripts/prepare_w4_submission.sh` to
  populate `submission/model` only after that PASS report.

**All numeric values above (except `max_model_len`/`gpu_memory_utilization`
choices, which are RTX-3090-informed but H200-untested) are reasoned
assumptions** — see `EXPERIMENTS.md` "H200 submission profile" for the
value-by-value reasoning, including a real incident: raising
`max_num_seqs`+`max_model_len` together cost 37% more VRAM than either alone
would predict at BF16 (7236 MiB vs ~5.3GB expected at the same
`gpu_memory_utilization`) — re-tested with FP8 and the same combination now
peaks at only 5628-5636 MiB, so FP8's smaller weights absorbed that effect on
this GPU. `GPU_MEM_UTIL` was still kept at a conservative 0.70 (not the
initially-planned 0.78) for extra margin. See `next_step.md` for the full
technique portfolio and `SUBMISSION_PLAN.md` for exact human next steps
(`docker build`/`push` from a machine with Docker — not possible on this
box — then submit via the BTC portal).
