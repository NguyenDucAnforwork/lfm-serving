# Local W4A16 Docker build notes

Server SSH target:

```bash
ssh -i ~/.ssh/id_ed25519 -p 11289 root@59.127.5.172
```

Historical preferred artifact, already tested officially and rejected for score:

```text
artifacts/lfm2-w4a16-gptq-g64-mlp10-15-bf16
```

Current optional one-slot W4 probe artifact, exported into `/dev/shm`:

```text
/dev/shm/artifacts-work/lfm2-w4a16-gptq-g64-mlp10-15-attn10-12-14-bf16
```

Fallback/debug artifact:

```text
artifacts/lfm2-w4a16-gptq-g64-late-mlp-bf16
```

Do not treat any W4 image here as a validated final candidate. Official H200
results rejected the `mlp10-15` artifact with both Marlin and Machete
(approximately 49 ERS, TBT median 6 ms). The `mlp10-15-attn10-12-14` artifact
passed local trace but failed local GSM8K limit-200; submit at most as a backend
probe.

## 1. Sync repo-side build files

Run from the local directory where you want the repo copy:

```bash
mkdir -p lfm-serving
rsync -avz \
  -e "ssh -i ~/.ssh/id_ed25519 -p 11289" \
  root@59.127.5.172:/workspace/lfm-serving/Dockerfile \
  root@59.127.5.172:/workspace/lfm-serving/.dockerignore \
  root@59.127.5.172:/workspace/lfm-serving/EXPERIMENTS.md \
  ./lfm-serving/

mkdir -p ./lfm-serving/submission ./lfm-serving/configs ./lfm-serving/recipes

rsync -avz \
  -e "ssh -i ~/.ssh/id_ed25519 -p 11289" \
  root@59.127.5.172:/workspace/lfm-serving/submission/ \
  ./lfm-serving/submission/

rsync -avz \
  -e "ssh -i ~/.ssh/id_ed25519 -p 11289" \
  root@59.127.5.172:/workspace/lfm-serving/configs/w4a16_gptq_marlin_g64_mlp10_15_bf16.env \
  root@59.127.5.172:/workspace/lfm-serving/configs/w4a16_gptq_machete_g64_mlp10_15_bf16.env \
  root@59.127.5.172:/workspace/lfm-serving/configs/w4a16_gptq_marlin_g64_late_mlp_bf16.env \
  root@59.127.5.172:/workspace/lfm-serving/configs/w4a16_gptq_machete_g64_late_mlp_bf16.env \
  ./lfm-serving/configs/

rsync -avz \
  -e "ssh -i ~/.ssh/id_ed25519 -p 11289" \
  root@59.127.5.172:/workspace/lfm-serving/recipes/w4a16_gptq_group64_mlp10_15_bf16.yaml \
  root@59.127.5.172:/workspace/lfm-serving/recipes/w4a16_gptq_group64_late_mlp_bf16.yaml \
  ./lfm-serving/recipes/
```

## 2. Sync the selected checkpoint into `submission/model`

Preferred candidate:

```bash
cd lfm-serving
rm -rf submission/model
mkdir -p submission/model
rsync -az --info=progress2 \
  -e "ssh -i ~/.ssh/id_ed25519 -p 11289" \
  root@59.127.5.172:/workspace/lfm-serving/artifacts/lfm2-w4a16-gptq-g64-mlp10-15-bf16/ \
  ./submission/model/
```

Current optional probe:

```bash
cd lfm-serving
rm -rf submission/model
mkdir -p submission/model
rsync -az --info=progress2 \
  -e "ssh -i ~/.ssh/id_ed25519 -p 11289" \
  root@59.127.5.172:/dev/shm/artifacts-work/lfm2-w4a16-gptq-g64-mlp10-15-attn10-12-14-bf16/ \
  ./submission/model/
```

Fallback/debug artifact:

```bash
cd lfm-serving
rm -rf submission/model
mkdir -p submission/model
rsync -az --info=progress2 \
  -e "ssh -i ~/.ssh/id_ed25519 -p 11289" \
  root@59.127.5.172:/workspace/lfm-serving/artifacts/lfm2-w4a16-gptq-g64-late-mlp-bf16/ \
  ./submission/model/
```

Expected checkpoint sanity checks after sync:

```bash
test -f submission/model/config.json
test -f submission/model/tokenizer.json
test -f submission/model/model.safetensors.index.json || ls submission/model/*.safetensors
grep -R "\"quant_method\".*compressed-tensors\\|compressed-tensors" submission/model/config.json
```

## 3. Build and push

Replace the image name with your registry/user.

Preferred candidate:

```bash
docker build -f submission/Dockerfile.w4a16-local \
  -t YOUR_DOCKERHUB_USER/lfm-serving:w4a16-g64-mlp10-15-bf16 .

docker push YOUR_DOCKERHUB_USER/lfm-serving:w4a16-g64-mlp10-15-bf16
```

Current optional probe:

```bash
docker build -f submission/Dockerfile.w4a16-local \
  -t YOUR_DOCKERHUB_USER/lfm-serving:w4a16-g64-mlp10-15-attn10-12-14-bf16 .

docker push YOUR_DOCKERHUB_USER/lfm-serving:w4a16-g64-mlp10-15-attn10-12-14-bf16
```

Fallback/debug:

```bash
docker build -f submission/Dockerfile.w4a16-local \
  -t YOUR_DOCKERHUB_USER/lfm-serving:w4a16-g64-late-mlp-bf16 .

docker push YOUR_DOCKERHUB_USER/lfm-serving:w4a16-g64-late-mlp-bf16
```

## 4. Local cold-start gate

Edit the selected compose and replace `YOUR_DOCKERHUB_USER/...` with the pushed
image tag, then:

```bash
docker compose -f submission/docker-compose.w4a16-g64-mlp10-15-bf16.local.yml up
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | grep 'LFM2.5-1.2B-Instruct'
docker logs "$(docker compose -f submission/docker-compose.w4a16-g64-mlp10-15-bf16.local.yml ps -q model)" 2>&1 \
  | grep -E 'Machete|Marlin|CompressedTensorsWNA16'
```

The compose intentionally omits explicit quantization:

```text
--linear-backend=machete
```

Do not add `--quantization=compressed-tensors`; vLLM 0.22.1 must autodetect the
compressed-tensors metadata from `/model/config.json`. The explicit flag caused
a competition startup failure. These W4 configs intentionally do not contain
`--quantization=fp8_per_tensor` and do not use BitsAndBytes.
