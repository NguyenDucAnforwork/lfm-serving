# Submission image for LiquidAI/LFM2.5-1.2B-Instruct on vLLM.
#
# NOT BUILT/RUN on this dev box (no Docker-in-Docker here) -- build/push from a
# machine with Docker. The BTC portal's docker-compose.yml FIXES the entrypoint
# to `python3 -m vllm.entrypoints.openai.api_server` and passes all flags via
# `command:` (see submission/docker-compose.*.yml) -- this image therefore does
# NOT set its own ENTRYPOINT/CMD; scripts/start_server.sh is dev-only.
#
# Required base image (pins vLLM 0.22.1, matching the dev venv used for every
# experiment in EXPERIMENTS.md).
FROM vllm/vllm-openai:v0.22.1

# xxhash is required by --prefix-caching-hash-algo xxhash (used in the H200
# profile). Caught during validation (EXPERIMENTS.md "Additional vLLM tuning"
# near-miss note): without this package every request 500s with
# ModuleNotFoundError -- vLLM does not fall back silently. Not present by
# default in this base image's vLLM install.
RUN pip install --no-cache-dir xxhash

# Model baked into the image at the fixed path /model (BTC's compose sets
# --model=/model). No weight/tokenizer modification -- these are the exact
# files from the official LiquidAI/LFM2.5-1.2B-Instruct repo, only
# dereferenced from the local HF cache's symlinks (see submission/model/).
# Baking (rather than a runtime download) is required by the anti-cheat rule
# against external network calls at grading time.
COPY submission/model /model

# Anti-cheat / compliance: no external network calls at runtime.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1

EXPOSE 8000
