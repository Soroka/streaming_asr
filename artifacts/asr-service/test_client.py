"""
Test client for the T-one streaming ASR WebSocket service.

Usage:
    # Test with real audio file:
    python3 artifacts/asr-service/test_client.py path/to/audio.wav

    # Test connectivity with a sine-wave tone (no file needed):
    python3 artifacts/asr-service/test_client.py

Install deps:
    pip install websockets soundfile
"""

import asyncio
import json
import sys
import numpy as np

try:
    import websockets
except ImportError:
    print("Install: pip install websockets")
    sys.exit(1)

WS_URL = "ws://localhost:8000/ws/transcribe"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 2400   # 150 ms — match model's native chunk size


async def transcribe_file(audio_path: str):
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(f"Audio: {len(audio)/sr:.2f}s @ {sr} Hz → {len(audio)} samples")

    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"type": "config", "sample_rate": sr}))
        ack = json.loads(await ws.recv())
        print(f"[ack] {ack['message']}")

        for i in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[i: i + CHUNK_SAMPLES]
            await ws.send(chunk.astype(np.float32).tobytes())
            # Drain any ready partial results
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                result = json.loads(msg)
                if result.get("text"):
                    print(f"[partial] {result['text']}")
            except asyncio.TimeoutError:
                pass

        await ws.send(json.dumps({"type": "end"}))

        while True:
            msg = json.loads(await ws.recv())
            if msg.get("is_final"):
                print(f"\n[FINAL] {msg['text']!r}")
                break
            if msg.get("text"):
                print(f"[partial] {msg['text']}")


async def test_connectivity():
    """Connectivity test: send a 440 Hz tone, expect non-empty final message."""
    t = np.linspace(0, 3.0, SAMPLE_RATE * 3, endpoint=False, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print(f"Sending 3s 440 Hz tone to {WS_URL}")
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"type": "config", "sample_rate": SAMPLE_RATE}))
        ack = json.loads(await ws.recv())
        print(f"[ack] {ack['message']}")

        for i in range(0, len(tone), CHUNK_SAMPLES):
            chunk = tone[i: i + CHUNK_SAMPLES]
            await ws.send(chunk.tobytes())

        await ws.send(json.dumps({"type": "end"}))

        while True:
            msg = json.loads(await ws.recv())
            print(f"[recv] {msg}")
            if msg.get("is_final"):
                break

    print("Connectivity test passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(transcribe_file(sys.argv[1]))
    else:
        print("No audio file given — running connectivity test with a 440 Hz tone\n")
        asyncio.run(test_connectivity())
