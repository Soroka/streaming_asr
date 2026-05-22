# T-one Streaming ASR Service

A low-latency Russian speech-to-text WebSocket service powered by the [t-tech/T-one](https://huggingface.co/t-tech/T-one) model via **native PyTorch** (no ONNX Runtime required for inference).

## Run & Operate

- **ASR Service** — workflow `ASR Service (T-one)` runs on port 8000
  - Start command: `bash artifacts/asr-service/start.sh`
  - Health check: `curl http://localhost:8000/health`
  - API info: `curl http://localhost:8000/info`
- `pnpm --filter @workspace/api-server run dev` — run the Node.js API server (port 5000, if needed)
- `pnpm run typecheck` — full typecheck across all Node.js packages
- `pnpm run build` — typecheck + build all Node.js packages

## Stack

- **ASR service**: Python 3.11, FastAPI, Uvicorn, PyTorch 2.2.2 (CPU), torchaudio, transformers 4.x, tone package
- **Model**: t-tech/T-one — stateful streaming Conformer-CTC, Russian ASR, PyTorch weights
- **Decoder**: KenLM beam-search CTC (via pyctcdecode); greedy fallback available
- **Node API server**: Express 5, TypeScript, Drizzle ORM (available but unused by default)

## Where things live

- `artifacts/asr-service/main.py` — FastAPI app + WebSocket handler
- `artifacts/asr-service/model.py` — PyTorch model (`TorchStreamingCTCModel`), streaming session, phrase splitting
- `artifacts/asr-service/preprocessing.py` — audio pipeline (DC removal, normalise, VAD, denoise; pre-emphasis disabled — model applies it internally)
- `artifacts/asr-service/metrics.py` — per-session and global aggregate metrics
- `artifacts/asr-service/start.sh` — service entrypoint (installs tone package, CUDA auto-detect)
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
- The named volume `t-one-model-cache` persists model weights between container restarts.
- Health check: `curl http://localhost:8000/health`

## WebSocket Protocol

Connect to `ws://localhost:8000/ws/transcribe`

| Step | Direction | Payload |
|------|-----------|---------|
| 1 | → | `{"type":"config","sample_rate":16000}` |
| 2 | ← | `{"type":"ack","message":"…"}` |
| 3 | → | Binary: raw PCM **float32** mono (any sample rate — resampled to 8 kHz internally) |
| 4 | ← | `{"type":"partial","text":"…","start_time":0.5,"end_time":1.2,"is_final":false}` *(emitted per complete phrase)* |
| 5 | → | `{"type":"end"}` |
| 6 | ← | `{"type":"final","text":"…","is_final":true,"phrases":[…],"metrics":{…}}` |

**Audio format**: PCM float32 in [-1, 1] **or** int16. Any input sample rate is accepted and resampled
to 8 kHz (model native rate). The model processes 2400-sample (300 ms) chunks with stateful attention
context. Partial messages are emitted immediately when a phrase boundary is detected.

**Final response `phrases`**: array of `{text, start_time, end_time}` covering the whole utterance.

## Model Architecture

- **Acoustic model**: `ToneForCTC` (PyTorch `PreTrainedModel`) — 71M parameter streaming Conformer-CTC
- **Feature extraction**: log-mel filterbank (80 coefficients, 8 kHz, 20 ms window, 10 ms stride); applied inside `Tone.forward_for_export()`
- **Input chunk**: `[batch, 2400, 1]` — raw int32 PCM @ 8 kHz = 300 ms
- **Acoustic output**: `logprobs [batch, 10, 35]` — CTC log-probs, 35-class Russian vocab
- **Streaming state**: 7 PyTorch tensors (preprocessor + encoder MHSA/conv/subsampling/reduction states)
- **Phrase splitter**: `StreamingLogprobSplitter` — silence-based phrase boundary detection
- **Decoder**: `BeamSearchCTCDecoder` — KenLM 5-gram LM beam search (beam=200); optional greedy fallback

## Testing

```bash
# Connectivity test (no audio file needed)
python3 artifacts/asr-service/test_client.py

# Transcribe a WAV file
python3 artifacts/asr-service/test_client.py path/to/audio.wav
```

## Architecture Decisions

- **PyTorch over ONNX**: replaced direct ONNX Runtime inference with `ToneForCTC.from_pretrained()` and `Tone.forward_for_export()`. State is managed as a 7-tuple of PyTorch tensors instead of a single flat float16 array. ONNX Runtime remains installed (imported at module level by the `tone` package), but is not used for inference.
- **8 kHz sample rate**: the model was trained on telephone speech at 8 kHz. Audio from clients at any rate (typically 16 kHz) is resampled down internally via librosa. Clients do not need to change their sample rate.
- **KenLM beam-search decode**: the tone package's `BeamSearchCTCDecoder` uses a 5.46 GB KenLM binary from the HuggingFace repo, providing significantly lower WER than greedy decoding.
- **Phrase-level streaming output**: `StreamingLogprobSplitter` detects silence boundaries and emits complete phrases immediately; each partial WebSocket message includes `start_time`/`end_time` (seconds relative to utterance start).
- **Pre-emphasis disabled in preprocessor**: the `Tone` model applies pre-emphasis internally in `FilterbankFeatures.forward_streaming()`; enabling it again in `preprocessing.py` would double-apply it.
- **Stateful per-connection sessions**: each WebSocket connection owns an independent `StreamingSession` with its own 7-tuple model state, logprob splitter state, and audio buffer.

## Gotchas

- The model expects raw PCM int32 in [-32768, 32767] internally; `model.py` converts float32 → int32 automatically.
- NumPy must be pinned to `<2` (numpy==1.26.4) due to binary compatibility with torch 2.2.2.
- `transformers` must stay `<5.0` — version 5.x requires PyTorch ≥ 2.4 but we use 2.2.2.
- The `tone` package is not on PyPI; `start.sh` installs it from GitHub on first run.
- KenLM binary (`kenlm.bin`) is ~5.5 GB; download takes ~1 min on first run and is cached by huggingface_hub.
- Always pass the correct `sample_rate` in the config message so the resampler works correctly.

## User Preferences

_Populate as you build._
