import asyncio
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

asr_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model
    from model import ASRModel
    asr_model = ASRModel()
    logger.info("ASR service ready")
    yield
    logger.info("ASR service shutting down")


app = FastAPI(title="T-one Streaming ASR", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "t-tech/T-one"}


@app.get("/info")
async def info():
    return {
        "model": "t-tech/T-one",
        "description": "Russian streaming ASR — stateful Conformer CTC via ONNX",
        "websocket_endpoint": "/ws/transcribe",
        "protocol": {
            "step_1_connect": "ws://<host>/ws/transcribe",
            "step_2_config":  '→ send JSON  {"type":"config","sample_rate":16000}',
            "step_3_stream":  "→ send binary frames: raw PCM float32 [-1,1] or int16, mono",
            "step_4_partial": '← receive   {"type":"partial","text":"…","is_final":false}',
            "step_5_end":     '→ send JSON  {"type":"end"}',
            "step_6_final":   '← receive   {"type":"final","text":"…","is_final":true}',
        },
        "audio_format": {
            "encoding": "PCM float32 (preferred) or int16",
            "channels": 1,
            "sample_rate": 16000,
            "chunk_size_samples": 2400,
            "chunk_duration_ms": 150,
        },
        "notes": [
            "Each 150 ms audio chunk is processed with stateful attention for low-latency streaming.",
            "Partial results are emitted after every 2 s of buffered complete chunks.",
            "The final result uses global CTC decoding over the full utterance for best accuracy.",
        ],
    }


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    session = asr_model.new_session()
    loop = asyncio.get_event_loop()
    PARTIAL_EMIT_CHUNKS = 13   # ~2 s of audio before emitting a partial

    chunk_count = 0

    try:
        while True:
            data = await websocket.receive()

            # ── Text (control) messages ─────────────────────────────────
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type")

                if msg_type == "config":
                    sr = int(msg.get("sample_rate", 16000))
                    session.configure(sr)
                    await websocket.send_json({
                        "type": "ack",
                        "message": f"Config accepted — sample_rate={sr}",
                    })
                    logger.info(f"Config: sample_rate={sr}")

                elif msg_type == "end":
                    logger.info("End-of-stream — flushing remaining audio")
                    final_text = await loop.run_in_executor(None, session.flush)
                    await websocket.send_json({
                        "type": "final",
                        "text": final_text,
                        "is_final": True,
                    })
                    logger.info(f"Final transcript: {final_text!r}")
                    break

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type!r}",
                    })

            # ── Binary (audio) frames ───────────────────────────────────
            elif "bytes" in data:
                raw = data["bytes"]
                partial = await loop.run_in_executor(None, session.push, raw)
                chunk_count += 1

                if partial and chunk_count % PARTIAL_EMIT_CHUNKS == 0:
                    await websocket.send_json({
                        "type": "partial",
                        "text": partial,
                        "is_final": False,
                    })
                    logger.debug(f"Partial transcript: {partial!r}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.exception(f"Unhandled WebSocket error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
