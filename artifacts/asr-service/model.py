import logging
import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "t-tech/T-one"

# Vocabulary: index → character
VOCAB = {
    0: "а", 1: "б", 2: "в", 3: "г", 4: "д", 5: "е", 6: "ё",
    7: "ж", 8: "з", 9: "и", 10: "й", 11: "к", 12: "л", 13: "м",
    14: "н", 15: "о", 16: "п", 17: "р", 18: "с", 19: "т", 20: "у",
    21: "ф", 22: "х", 23: "ц", 24: "ч", 25: "ш", 26: "щ", 27: "ъ",
    28: "ы", 29: "ь", 30: "э", 31: "ю", 32: "я",
    33: " ",   # word delimiter "|"
    34: "",    # CTC blank / PAD — excluded from output
}
BLANK_ID = 34

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 2400       # 150 ms @ 16 kHz — fixed model input size
FRAMES_PER_CHUNK = 10      # CTC frames emitted per chunk
STATE_SIZE = 219729        # Stateful attention state vector length


def _to_int32_pcm(audio_f32: np.ndarray) -> np.ndarray:
    """Convert float32 [-1,1] audio to int32 PCM in [-32767, 32767]."""
    return (audio_f32 * 32767.0).clip(-32768, 32767).astype(np.int32)


def _ctc_decode(logprobs: np.ndarray) -> str:
    """Greedy CTC decode. logprobs: [time, vocab_size]."""
    ids = np.argmax(logprobs, axis=-1)
    prev = -1
    chars = []
    for idx in ids:
        idx = int(idx)
        if idx != prev:
            if idx != BLANK_ID:
                chars.append(VOCAB.get(idx, ""))
        prev = idx
    return "".join(chars).strip()


class ASRModel:
    """
    T-one (t-tech/T-one) ONNX-based streaming ASR model.

    ONNX interface:
        input  'signal'     [batch, 2400, 1]     int32   — raw PCM audio
        input  'state'      [batch, 219729]       float16 — attention state
        output 'logprobs'   [batch, 10, 35]       float   — CTC log-probs
        output 'state_next' [batch, 219729]       float16 — updated state
    """

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading ONNX model from {MODEL_ID} …")
        model_path = hf_hub_download(MODEL_ID, "model.onnx")
        logger.info(f"Model cached at {model_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX session ready — stateful streaming model")

    # ------------------------------------------------------------------
    # Low-level: single chunk inference
    # ------------------------------------------------------------------

    def _infer_chunk(
        self,
        chunk_int32: np.ndarray,   # [1, 2400, 1] int32
        state: np.ndarray,         # [1, STATE_SIZE] float16
    ):
        """Run one chunk through the model, return (logprobs, state_next)."""
        outputs = self.session.run(
            ["logprobs", "state_next"],
            {"signal": chunk_int32, "state": state},
        )
        logprobs = outputs[0][0]    # [10, 35]
        state_next = outputs[1]     # [1, STATE_SIZE]
        return logprobs, state_next

    def _zero_state(self) -> np.ndarray:
        return np.zeros((1, STATE_SIZE), dtype=np.float16)

    # ------------------------------------------------------------------
    # Prepare audio for chunk-by-chunk processing
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_chunks(audio_f32: np.ndarray) -> list[np.ndarray]:
        """
        Normalize, pad to multiple of CHUNK_SAMPLES, split into chunks.
        Returns list of int32 arrays each shaped [1, CHUNK_SAMPLES, 1].
        """
        # Normalise int16-range PCM to [-1, 1] if needed
        if np.abs(audio_f32).max() > 1.0:
            audio_f32 = audio_f32 / 32768.0

        # Pad to multiple of CHUNK_SAMPLES
        remainder = len(audio_f32) % CHUNK_SAMPLES
        if remainder:
            audio_f32 = np.concatenate(
                [audio_f32, np.zeros(CHUNK_SAMPLES - remainder, dtype=np.float32)]
            )

        pcm = _to_int32_pcm(audio_f32)
        chunks = []
        for i in range(0, len(pcm), CHUNK_SAMPLES):
            chunk = pcm[i: i + CHUNK_SAMPLES].reshape(1, CHUNK_SAMPLES, 1)
            chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------
    # Full-utterance transcription (sequential with state threading)
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if len(audio) == 0:
            return ""

        audio = audio.astype(np.float32)

        if sample_rate != SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            except ImportError:
                logger.warning("librosa unavailable — skipping resample")

        chunks = self._prepare_chunks(audio)
        state = self._zero_state()
        all_logprobs = []

        for chunk in chunks:
            lp, state = self._infer_chunk(chunk, state)
            all_logprobs.append(lp)

        logprobs = np.concatenate(all_logprobs, axis=0)  # [total_frames, 35]
        return _ctc_decode(logprobs)

    # ------------------------------------------------------------------
    # Streaming session (used by the WebSocket handler)
    # ------------------------------------------------------------------

    def new_session(self) -> "StreamingSession":
        return StreamingSession(self)


class StreamingSession:
    """
    Manages per-connection streaming state.

    Usage:
        session = model.new_session()
        for raw_bytes in audio_chunks:
            partial = session.push(raw_bytes, sample_rate)
            if partial:
                send_to_client(partial)
        final = session.flush()
    """

    def __init__(self, model: ASRModel):
        self._model = model
        self._state = model._zero_state()
        self._sample_rate = SAMPLE_RATE
        self._pcm_buffer = np.array([], dtype=np.int32)
        self._all_logprobs: list[np.ndarray] = []

    def configure(self, sample_rate: int):
        self._sample_rate = sample_rate

    def push(self, audio_bytes: bytes) -> str | None:
        """
        Accept raw audio bytes (float32 or int16 PCM).
        Processes complete 2400-sample chunks and returns a partial
        transcript string if any new complete words are emitted,
        or None if nothing new.
        """
        # Try float32 first, fall back to int16
        try:
            audio_f32 = np.frombuffer(audio_bytes, dtype=np.float32)
        except ValueError:
            audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        # Resample if needed
        if self._sample_rate != SAMPLE_RATE:
            try:
                import librosa
                audio_f32 = librosa.resample(
                    audio_f32, orig_sr=self._sample_rate, target_sr=SAMPLE_RATE
                )
            except ImportError:
                pass

        # Normalise
        if len(audio_f32) > 0 and np.abs(audio_f32).max() > 1.0:
            audio_f32 = audio_f32 / 32768.0

        new_pcm = _to_int32_pcm(audio_f32)
        self._pcm_buffer = np.concatenate([self._pcm_buffer, new_pcm])

        # Process all complete chunks
        chunk_logprobs = []
        while len(self._pcm_buffer) >= CHUNK_SAMPLES:
            chunk = self._pcm_buffer[:CHUNK_SAMPLES].reshape(1, CHUNK_SAMPLES, 1)
            self._pcm_buffer = self._pcm_buffer[CHUNK_SAMPLES:]
            lp, self._state = self._model._infer_chunk(chunk, self._state)
            chunk_logprobs.append(lp)
            self._all_logprobs.append(lp)

        if not chunk_logprobs:
            return None

        partial_text = _ctc_decode(np.concatenate(chunk_logprobs, axis=0))
        return partial_text if partial_text.strip() else None

    def flush(self) -> str:
        """
        Process any remaining buffered audio and return the full transcript
        decoded from all chunks seen in this session.
        """
        if len(self._pcm_buffer) > 0:
            pad = CHUNK_SAMPLES - len(self._pcm_buffer)
            chunk = np.concatenate(
                [self._pcm_buffer, np.zeros(pad, dtype=np.int32)]
            ).reshape(1, CHUNK_SAMPLES, 1)
            lp, self._state = self._model._infer_chunk(chunk, self._state)
            self._all_logprobs.append(lp)
            self._pcm_buffer = np.array([], dtype=np.int32)

        if not self._all_logprobs:
            return ""

        all_lp = np.concatenate(self._all_logprobs, axis=0)
        return _ctc_decode(all_lp)
