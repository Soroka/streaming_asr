import logging
import time
import numpy as np
from preprocessing import AudioPreprocessor, PreprocessorConfig

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


def _detect_providers() -> list:
    """
    Return the best available ONNX Runtime execution providers in priority order.
    Prefers CUDA when a GPU is present, falls back to CPU.
    """
    import onnxruntime as ort
    available = ort.get_available_providers()
    logger.info(f"ORT available providers: {available}")

    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4 GB cap
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            )
        )
        logger.info("CUDA GPU detected — using CUDAExecutionProvider")
    else:
        logger.info("No CUDA GPU — using CPUExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


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

    Automatically uses CUDAExecutionProvider when a GPU is available,
    and falls back to CPUExecutionProvider otherwise.
    """

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading ONNX model from {MODEL_ID} …")
        model_path = hf_hub_download(MODEL_ID, "model.onnx")
        logger.info(f"Model cached at {model_path}")

        providers = _detect_providers()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
        )

        active = self.session.get_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in active else "cpu"
        logger.info(f"ONNX session ready — active providers: {active} | device={self.device}")

    # ------------------------------------------------------------------
    # Low-level: single chunk inference (returns latency_ms too)
    # ------------------------------------------------------------------

    def _infer_chunk(
        self,
        chunk_int32: np.ndarray,   # [1, 2400, 1] int32
        state: np.ndarray,         # [1, STATE_SIZE] float16
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Run one chunk through the model.
        Returns (logprobs [10,35], state_next [1,STATE_SIZE], latency_ms).
        """
        t0 = time.perf_counter()
        outputs = self.session.run(
            ["logprobs", "state_next"],
            {"signal": chunk_int32, "state": state},
        )
        latency_ms = (time.perf_counter() - t0) * 1_000
        logprobs = outputs[0][0]    # [10, 35]
        state_next = outputs[1]     # [1, STATE_SIZE]
        return logprobs, state_next, latency_ms

    def _zero_state(self) -> np.ndarray:
        return np.zeros((1, STATE_SIZE), dtype=np.float16)

    # ------------------------------------------------------------------
    # Prepare audio for chunk-by-chunk processing
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_chunks(audio_f32: np.ndarray) -> list[np.ndarray]:
        """
        Normalize, pad to a multiple of CHUNK_SAMPLES, split into chunks.
        Returns list of int32 arrays each shaped [1, CHUNK_SAMPLES, 1].
        """
        if np.abs(audio_f32).max() > 1.0:
            audio_f32 = audio_f32 / 32768.0

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
    # Full-utterance transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        preprocessor: AudioPreprocessor | None = None,
    ) -> str:
        if len(audio) == 0:
            return ""

        audio = audio.astype(np.float32)

        if sample_rate != SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            except ImportError:
                logger.warning("librosa unavailable — skipping resample")

        if preprocessor is not None:
            audio, info = preprocessor.process_utterance(audio)
            logger.debug(f"Utterance preprocessing: {info}")

        chunks = self._prepare_chunks(audio)
        state = self._zero_state()
        all_logprobs = []

        for chunk in chunks:
            lp, state, _ = self._infer_chunk(chunk, state)
            all_logprobs.append(lp)

        logprobs = np.concatenate(all_logprobs, axis=0)
        return _ctc_decode(logprobs)

    # ------------------------------------------------------------------
    # Streaming session factory
    # ------------------------------------------------------------------

    def new_session(self) -> "StreamingSession":
        return StreamingSession(self)


class StreamingSession:
    """
    Per-connection streaming state with built-in metrics collection.

    Usage:
        session = model.new_session()
        session.configure(sample_rate=16000, reference="optional ref text")
        for raw_bytes in audio_chunks:
            partial = session.push(raw_bytes)
            if partial:
                send_to_client(partial)
        final = session.flush()   # also finalises session.metrics
    """

    def __init__(self, model: ASRModel):
        from metrics import SessionMetrics
        self._model = model
        self._state = model._zero_state()
        self._sample_rate = SAMPLE_RATE
        self._reference = ""
        self._pcm_buffer = np.array([], dtype=np.int32)
        self._all_logprobs: list[np.ndarray] = []
        self.metrics = SessionMetrics()
        self.metrics.start()
        self.preprocessor = AudioPreprocessor(sample_rate=SAMPLE_RATE)

    def configure(
        self,
        sample_rate: int,
        reference: str = "",
        preprocessing: dict | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._reference = reference
        self.metrics.sample_rate = sample_rate
        if preprocessing is not None:
            cfg = PreprocessorConfig.from_dict(preprocessing)
        else:
            cfg = PreprocessorConfig()
        self.preprocessor.configure(cfg, sample_rate)

    def push(self, audio_bytes: bytes) -> tuple[str | None, dict]:
        """
        Accept raw audio bytes (float32 or int16 PCM).
        Applies the preprocessing pipeline, then processes complete
        2400-sample chunks through the ONNX model.

        Returns (partial_text | None, preprocessing_info).
          partial_text     — non-empty decoded text if any chunks produced output
          preprocessing_info — dict with 'speech_fraction', 'is_silent', 'steps_applied'
        """
        self.metrics.record_first_audio()

        try:
            audio_f32 = np.frombuffer(audio_bytes, dtype=np.float32).copy()
        except ValueError:
            audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        # Resample before preprocessing so VAD etc. always sees 16 kHz
        if self._sample_rate != SAMPLE_RATE:
            try:
                import librosa
                audio_f32 = librosa.resample(
                    audio_f32, orig_sr=self._sample_rate, target_sr=SAMPLE_RATE
                )
            except ImportError:
                pass

        # Normalise int16-range input to [-1, 1] before preprocessing
        if len(audio_f32) > 0 and np.abs(audio_f32).max() > 1.0:
            audio_f32 = audio_f32 / 32768.0

        # ── Preprocessing ────────────────────────────────────────────
        audio_f32, prep_info = self.preprocessor.process_chunk(audio_f32)

        self.metrics.record_audio_samples(len(audio_f32))

        new_pcm = _to_int32_pcm(audio_f32)
        self._pcm_buffer = np.concatenate([self._pcm_buffer, new_pcm])

        chunk_logprobs = []
        while len(self._pcm_buffer) >= CHUNK_SAMPLES:
            chunk = self._pcm_buffer[:CHUNK_SAMPLES].reshape(1, CHUNK_SAMPLES, 1)
            self._pcm_buffer = self._pcm_buffer[CHUNK_SAMPLES:]
            lp, self._state, lat_ms = self._model._infer_chunk(chunk, self._state)
            self.metrics.record_chunk_latency(lat_ms)
            chunk_logprobs.append(lp)
            self._all_logprobs.append(lp)

        if not chunk_logprobs:
            return None, prep_info

        partial_text = _ctc_decode(np.concatenate(chunk_logprobs, axis=0))
        if partial_text.strip():
            self.metrics.record_first_partial()
            return partial_text, prep_info
        return None, prep_info

    def flush(self) -> str:
        """
        Process any remaining buffered audio, decode the full utterance,
        finalise self.metrics, and return the final transcript.
        """
        self.metrics.record_first_audio()   # guard in case no audio was pushed

        if len(self._pcm_buffer) > 0:
            pad = CHUNK_SAMPLES - len(self._pcm_buffer)
            chunk = np.concatenate(
                [self._pcm_buffer, np.zeros(pad, dtype=np.int32)]
            ).reshape(1, CHUNK_SAMPLES, 1)
            lp, self._state, lat_ms = self._model._infer_chunk(chunk, self._state)
            self.metrics.record_chunk_latency(lat_ms)
            self._all_logprobs.append(lp)
            self._pcm_buffer = np.array([], dtype=np.int32)

        if not self._all_logprobs:
            self.metrics.finalise("", self._reference)
            return ""

        all_lp = np.concatenate(self._all_logprobs, axis=0)
        text = _ctc_decode(all_lp)
        self.metrics.finalise(text, self._reference)
        return text
