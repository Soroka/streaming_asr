import asyncio
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
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


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model": "t-tech/T-one"}


@app.get("/info")
async def info():
    return {
        "model": "t-tech/T-one",
        "description": "Russian streaming ASR — stateful Conformer CTC via ONNX",
        "device": asr_model.device if asr_model else "not loaded",
        "websocket_endpoint": "/ws/transcribe",
        "protocol": {
            "step_1_connect": "ws://<host>/ws/transcribe",
            "step_2_config":  '→ send JSON  {"type":"config","sample_rate":16000,"reference":"optional ref for WER"}',
            "step_3_stream":  "→ send binary frames: raw PCM float32 [-1,1] or int16, mono",
            "step_4_partial": '← receive   {"type":"partial","text":"…","is_final":false}',
            "step_5_end":     '→ send JSON  {"type":"end"}',
            "step_6_final":   '← receive   {"type":"final","text":"…","is_final":true,"metrics":{…}}',
        },
        "audio_format": {
            "encoding": "PCM float32 (preferred) or int16",
            "channels": 1,
            "sample_rate": 16000,
            "chunk_size_samples": 2400,
            "chunk_duration_ms": 150,
        },
        "metrics_endpoint": "GET /metrics",
    }


@app.get("/metrics")
async def get_metrics(last_n: int = Query(default=20, ge=1, le=500)):
    """
    Return global aggregate metrics over all completed sessions.

    Query params:
      last_n  — number of recent sessions to include in the 'recent_sessions'
                list (default 20, max 500).

    Metrics explained:
      chunk_latency_ms  — ONNX inference time per 150 ms audio chunk
      ttft_ms           — time-to-first-token: ms from first audio frame to first partial result
      e2e_latency_ms    — end-to-end: ms from first audio frame to final result
      rtf               — real-time factor = total_inference_time / audio_duration
                          (< 1.0 means faster than real-time)
      wer               — word error rate (only sessions where a reference was provided)
    """
    from metrics import global_metrics
    return global_metrics.snapshot(last_n=last_n)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

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
                    reference = str(msg.get("reference", ""))
                    preprocessing = msg.get("preprocessing")  # optional dict
                    session.configure(sr, reference=reference, preprocessing=preprocessing)

                    from preprocessing import PreprocessorConfig
                    active_cfg = session.preprocessor.config
                    await websocket.send_json({
                        "type": "ack",
                        "message": f"Config accepted — sample_rate={sr}"
                                   + (", reference set" if reference else ""),
                        "preprocessing": {
                            "dc_removal":    active_cfg.dc_removal,
                            "pre_emphasis":  active_cfg.pre_emphasis,
                            "normalize":     active_cfg.normalize,
                            "vad_enabled":   active_cfg.vad_enabled,
                            "vad_aggressiveness": active_cfg.vad_aggressiveness,
                            "denoise":       active_cfg.denoise,
                        },
                    })
                    logger.info(
                        f"Config: sample_rate={sr}, reference={'set' if reference else 'none'}, "
                        f"preprocessing={preprocessing or 'defaults'}"
                    )

                elif msg_type == "end":
                    logger.info("End-of-stream — flushing remaining audio")
                    final_text = await loop.run_in_executor(None, session.flush)

                    sm = session.metrics
                    from metrics import global_metrics
                    global_metrics.record(sm)

                    session_summary = sm.summary()
                    logger.info(
                        f"Session {sm.session_id} done | "
                        f"rtf={session_summary['rtf']} | "
                        f"e2e={session_summary['e2e_latency_ms']} ms | "
                        f"wer={session_summary['wer']}"
                    )

                    await websocket.send_json({
                        "type": "final",
                        "text": final_text,
                        "is_final": True,
                        "metrics": session_summary,
                    })
                    break

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type!r}",
                    })

            # ── Binary (audio) frames ───────────────────────────────────
            elif "bytes" in data:
                raw = data["bytes"]
                partial, prep_info = await loop.run_in_executor(None, session.push, raw)
                chunk_count += 1

                if partial and chunk_count % PARTIAL_EMIT_CHUNKS == 0:
                    await websocket.send_json({
                        "type": "partial",
                        "text": partial,
                        "is_final": False,
                        "speech_fraction": prep_info.get("speech_fraction"),
                    })
                    logger.debug(f"Partial: {partial!r} (speech={prep_info.get('speech_fraction')})")

    except WebSocketDisconnect:
        from metrics import global_metrics
        global_metrics.record_error()
        logger.info("WebSocket disconnected")
    except Exception as exc:
        from metrics import global_metrics
        global_metrics.record_error()
        logger.exception(f"Unhandled WebSocket error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
