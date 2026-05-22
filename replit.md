# T-one Streaming ASR Service

A low-latency Russian speech-to-text WebSocket service powered by the [t-tech/T-one](https://huggingface.co/t-tech/T-one) model via ONNX Runtime.

## Run & Operate

- **ASR Service** — workflow `ASR Service (T-one)` runs on port 8000
  - Start command: `bash artifacts/asr-service/start.sh`
  - Health check: `curl http://localhost:8000/health`
  - API info: `curl http://localhost:8000/info`
- `pnpm --filter @workspace/api-server run dev` — run the Node.js API server (port 5000, if needed)
- `pnpm run typecheck` — full typecheck across all Node.js packages
- `pnpm run build` — typecheck + build all Node.js packages

## Stack

- **ASR service**: Python 3.11, FastAPI, Uvicorn, ONNX Runtime (CPU), NumPy, librosa
- **Model**: t-tech/T-one — stateful streaming Conformer-CTC, Russian ASR, ONNX format
- **Node API server**: Express 5, TypeScript, Drizzle ORM (available but unused by default)

## Where things live

- `artifacts/asr-service/main.py` — FastAPI app + WebSocket handler
- `artifacts/asr-service/model.py` — ONNX model loader, streaming session, CTC decode
- `artifacts/asr-service/preprocessing.py` — audio pipeline (DC removal, pre-emphasis, normalise, VAD, denoise)
- `artifacts/asr-service/metrics.py` — per-session and global aggregate metrics
- `artifacts/asr-service/start.sh` — service entrypoint (CUDA auto-detect)
- `artifacts/asr-service/test_client.py` — WebSocket test client
- `artifacts/asr-service/Dockerfile` — CPU production image (two-stage build)
- `artifacts/asr-service/Dockerfile.gpu` — GPU image (CUDA 12.1 + cuDNN 8)
- `artifacts/asr-service/docker-compose.yml` — orchestrates CPU and GPU services
- `artifacts/asr-service/requirements.txt` — pinned CPU dependencies
- `artifacts/asr-service/requirements-gpu.txt` — pinned GPU dependencies

## Docker

### CPU (default)
```bash
cd artifacts/asr-service

# Build image (~3–5 min first time; subsequent builds use layer cache)
docker build -t t-one-asr .

# Run — model is downloaded on first start and cached in the volume
docker run -p 8000:8000 -v t-one-model-cache:/cache t-one-asr

# Or use Compose
docker compose up asr-cpu
```

### GPU (CUDA 12.1)
Requires NVIDIA driver ≥ 530 and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/).
```bash
docker build -f Dockerfile.gpu -t t-one-asr-gpu .
docker run --gpus all -p 8000:8000 -v t-one-model-cache:/cache t-one-asr-gpu

# Or use Compose
docker compose --profile gpu up asr-gpu
```

### Tips
- Pass `HF_TOKEN=hf_xxx` as an env var to avoid HuggingFace anonymous download rate limits.
- The named volume `t-one-model-cache` (~250 MB) persists the ONNX model between container restarts.
- Health check: `curl http://localhost:8000/health`

## WebSocket Protocol

Connect to `ws://localhost:8000/ws/transcribe`

| Step | Direction | Payload |
|------|-----------|---------|
| 1 | → | `{"type":"config","sample_rate":16000}` |
| 2 | ← | `{"type":"ack","message":"…"}` |
| 3 | → | Binary: raw PCM **float32** mono 16 kHz (any chunk size) |
| 4 | ← | `{"type":"partial","text":"…","is_final":false}` *(periodic)* |
| 5 | → | `{"type":"end"}` |
| 6 | ← | `{"type":"final","text":"…","is_final":true}` |

**Audio format**: PCM float32 in [-1, 1] **or** int16. The model internally segments into 2400-sample (150 ms) chunks and threads stateful attention context between them for streaming accuracy.

## Model Architecture

- Input: `signal [batch, 2400, 1]` — raw int32 PCM, 150 ms chunks
- Output: `logprobs [batch, 10, 35]` — CTC log-probs, 35-class Russian vocab
- State: `[batch, 219729]` float16 — attention carry-over between chunks
- Decode: greedy CTC over the concatenated logprobs of all chunks

## Testing

```bash
# Connectivity test (no audio file needed)
python3 artifacts/asr-service/test_client.py

# Transcribe a WAV file
python3 artifacts/asr-service/test_client.py path/to/audio.wav
```

## Architecture Decisions

- **ONNX over native transformers**: T-one uses a custom `ToneForCTC` architecture not registered in HuggingFace transformers. The repo ships `model.onnx` which runs without any custom Python class.
- **Stateful per-connection sessions**: The model carries `state` / `state_next` tensors between 150 ms chunks, giving it full utterance context despite streaming chunk-by-chunk.
- **CPU-only PyTorch + onnxruntime**: GPU CUDA builds exceed the disk quota; CPU inference is sufficient for this model size.
- **Greedy CTC decode**: A KenLM language model binary (`kenlm.bin`) is included in the HF repo and could be used for beam-search rescoring to improve accuracy — not yet integrated.

## Gotchas

- The model expects raw PCM int32 internally; the `model.py` converts float32 → int32 automatically.
- NumPy must be pinned to `<2` due to a binary compatibility issue with torch 2.2.2.
- Always resample audio to 16 kHz before sending (or pass the correct `sample_rate` in the config message).

## User Preferences

_Populate as you build._
