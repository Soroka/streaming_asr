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


app = FastAPI(title="T-one Streaming ASR", version="2.0.0", lifespan=lifespan)

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
    device = str(asr_model.device) if asr_model else "not loaded"
    decoder = asr_model.decoder_type if asr_model else "unknown"
    return {
        "model": "t-tech/T-one",
        "description": (
            "Russian streaming ASR — stateful Conformer CTC via native PyTorch. "
            "Includes KenLM beam-search decoder and phrase-level timestamps."
        ),
        "device": device,
        "decoder": decoder,
        "websocket_endpoint": "/ws/transcribe",
        "protocol": {
            "step_1_connect": "ws://<host>/ws/transcribe",
            "step_2_config":  '→ send JSON  {"type":"config","sample_rate":16000}',
            "step_3_stream":  "→ send binary frames: raw PCM float32 [-1,1] or int16, mono",
            "step_4_partial": '← receive   {"type":"partial","text":"…","start_time":0.5,"end_time":1.2,"is_final":false}',
            "step_5_end":     '→ send JSON  {"type":"end"}',
            "step_6_final":   '← receive   {"type":"final","text":"…","is_final":true,"phrases":[…],"metrics":{…}}',
        },
        "audio_format": {
            "encoding": "PCM float32 (preferred) or int16",
            "channels": 1,
            "sample_rate": "any — resampled internally to 8000 Hz",
            "recommended_client_sample_rate": 16000,
            "chunk_size_samples": 2400,
            "chunk_duration_ms": 300,
            "model_sample_rate": 8000,
        },
        "metrics_endpoint": "GET /metrics",
    }


@app.get("/metrics")
async def get_metrics(last_n: int = Query(default=20, ge=1, le=500)):
    """
    Return global aggregate metrics over all completed sessions.

    Metrics explained:
      chunk_latency_ms  — PyTorch inference time per 300 ms audio chunk
      ttft_ms           — time-to-first-token: ms from first audio to first phrase
      e2e_latency_ms    — end-to-end: ms from first audio to final result
      rtf               — real-time factor = total_inference_time / audio_duration
      wer               — word error rate (sessions with reference only)
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

    try:
        while True:
            data = await websocket.receive()

            # ── Text (control) messages ──────────────────────────────────
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type")

                if msg_type == "config":
                    sr            = int(msg.get("sample_rate", 16000))
                    reference     = str(msg.get("reference", ""))
                    preprocessing = msg.get("preprocessing")
                    session.configure(sr, reference=reference, preprocessing=preprocessing)

                    active_cfg = session.preprocessor.config
                    await websocket.send_json({
                        "type": "ack",
                        "message": (
                            f"Config accepted — sample_rate={sr} "
                            f"(resampled internally to 8000 Hz)"
                            + (", reference set" if reference else "")
                        ),
                        "preprocessing": {
                            "dc_removal":         active_cfg.dc_removal,
                            "pre_emphasis":       active_cfg.pre_emphasis,
                            "normalize":          active_cfg.normalize,
                            "vad_enabled":        active_cfg.vad_enabled,
                            "vad_aggressiveness": active_cfg.vad_aggressiveness,
                            "denoise":            active_cfg.denoise,
                        },
                    })
                    logger.info(
                        "Config: sample_rate=%d, reference=%s",
                        sr, "set" if reference else "none",
                    )

                elif msg_type == "end":
                    logger.info("End-of-stream — flushing remaining audio")
                    final_text = await loop.run_in_executor(None, session.flush)

                    sm = session.metrics
                    from metrics import global_metrics
                    global_metrics.record(sm)

                    summary = sm.summary()
                    logger.info(
                        "Session %s done | rtf=%s | e2e=%s ms | wer=%s",
                        sm.session_id, summary["rtf"],
                        summary["e2e_latency_ms"], summary["wer"],
                    )

                    # Build phrases list for the final message
                    phrases_out = [
                        {
                            "text":       p.text,
                            "start_time": round(p.start_time, 3),
                            "end_time":   round(p.end_time,   3),
                        }
                        for p in session.phrases()
                    ]

                    await websocket.send_json({
                        "type":    "final",
                        "text":    final_text,
                        "is_final": True,
                        "phrases": phrases_out,
                        "metrics": summary,
                    })
                    break

                else:
                    await websocket.send_json({
                        "type":    "error",
                        "message": f"Unknown message type: {msg_type!r}",
                    })

            # ── Binary (audio) frames ────────────────────────────────────
            elif "bytes" in data:
                raw = data["bytes"]
                new_phrases, prep_info = await loop.run_in_executor(
                    None, session.push, raw
                )

                # Emit a partial message for every newly completed phrase
                for phrase in new_phrases:
                    await websocket.send_json({
                        "type":            "partial",
                        "text":            phrase.text,
                        "start_time":      round(phrase.start_time, 3),
                        "end_time":        round(phrase.end_time,   3),
                        "is_final":        False,
                        "speech_fraction": prep_info.get("speech_fraction"),
                    })
                    logger.debug(
                        "Partial phrase: %r  [%.2f–%.2f s]",
                        phrase.text, phrase.start_time, phrase.end_time,
                    )

    except WebSocketDisconnect:
        from metrics import global_metrics
        global_metrics.record_error()
        logger.info("WebSocket disconnected")
    except Exception as exc:
        from metrics import global_metrics
        global_metrics.record_error()
        logger.exception("Unhandled WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
