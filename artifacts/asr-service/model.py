"""
model.py — T-one streaming ASR using the tone Python package (PyTorch backend).

Architecture
------------
  1. TorchStreamingCTCModel  — wraps ToneForCTC.tone for streaming inference
                               using forward_for_export(); no ONNX Runtime used.
  2. ASRModel                — top-level: loads weights, creates decoder, owns
                               a single TorchStreamingCTCModel + shared decoder.
  3. StreamingSession        — per-connection: audio buffer, pipeline state,
                               preprocessing, metrics.

Audio format
------------
  Sample rate : 8 000 Hz  (model trained on telephony speech)
  Chunk size  : 2 400 samples = 300 ms per inference step
  Input dtype : int32 PCM  ∈ [-32 768, 32 767]
  Padding     : 2 400-sample (300 ms) silence head+tail added in offline mode
                to improve phrase-boundary quality.

Output
------
  push()  → list[TextPhrase]  phrases with .text / .start_time / .end_time
  flush() → str               combined text of all finalized phrases

WebSocket client may send audio at any sample rate — the session resamples
to 8 kHz before inference.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from preprocessing import AudioPreprocessor, PreprocessorConfig

logger = logging.getLogger(__name__)

MODEL_ID       = "t-tech/T-one"
SAMPLE_RATE    = 8_000    # Hz — model trained on telephone speech
CHUNK_SAMPLES  = 2_400    # 300 ms @ 8 kHz — fixed model input size
PADDING        = 2_400    # silence head+tail for offline; mirrors pipeline.PADDING


# ---------------------------------------------------------------------------
# 1. TorchStreamingCTCModel
# ---------------------------------------------------------------------------

class TorchStreamingCTCModel:
    """
    PyTorch-native streaming acoustic model wrapper.

    Exposes the same forward(audio_chunk, state) interface expected by
    StreamingCTCPipeline so the pipeline can use it as a drop-in replacement
    for the ONNX-backed StreamingCTCModel.

    State is a 7-tuple of PyTorch tensors returned by Tone.get_initial_state().
    """

    # Constants mirrored from StreamingCTCModel for pipeline compatibility
    SAMPLE_RATE        = SAMPLE_RATE
    MEAN_TIME_BIAS     = 0.33   # seconds
    AUDIO_CHUNK_SAMPLES = CHUNK_SAMPLES
    FRAME_SIZE         = 0.03   # seconds per CTC output frame

    def __init__(self, tone_model, device: torch.device) -> None:
        """
        Args:
            tone_model: the ToneForCTC instance (already on `device`, in eval mode).
            device:     torch.device to run inference on.
        """
        self._model = tone_model
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        audio_chunk: np.ndarray,          # int32  [B, CHUNK_SAMPLES, 1]
        state: tuple | None = None,
    ) -> tuple[np.ndarray, tuple]:
        """
        Run one 300 ms chunk through the acoustic model.

        Returns:
            logprobs  — float32 [B, T, 35]
            next_state — 7-tuple of tensors to pass to the next call
        """
        B = audio_chunk.shape[0]
        if state is None:
            state = self._model.tone.get_initial_state(
                batch_size=B,
                device=self.device,
                target="export",
            )

        # int32 numpy → int32 tensor on device
        t = torch.from_numpy(audio_chunk.astype(np.int32)).to(self.device)

        with torch.no_grad():
            out = self._model.tone.forward_for_export(t, *state)

        # out[0]: logprobs float32[B, T, 35]  (cast by forward_for_export)
        # out[1:]: 7 state tensors
        logprobs   = out[0].cpu().float().numpy()
        next_state = tuple(s.detach() for s in out[1:])
        return logprobs, next_state


# ---------------------------------------------------------------------------
# 2. ASRModel
# ---------------------------------------------------------------------------

class ASRModel:
    """
    Top-level model object.

    Loads ToneForCTC from HuggingFace (safetensors weights), wraps it in a
    TorchStreamingCTCModel, and instantiates the phrase splitter + decoder
    from the tone package.

    Shared across all StreamingSession instances; only the per-session state
    (audio buffer, pipeline state, metrics) lives in StreamingSession.
    """

    def __init__(self, decoder: str = "beam_search") -> None:
        """
        Args:
            decoder: "beam_search" (KenLM, higher quality) or "greedy" (faster).
        """
        from tone.training.model_wrapper import ToneForCTC

        logger.info("Loading ToneForCTC from %s …", MODEL_ID)
        self._tone_model = ToneForCTC.from_pretrained(MODEL_ID)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", self.device)

        self._tone_model.to(self.device).eval()
        self.acoustic = TorchStreamingCTCModel(self._tone_model, self.device)

        if decoder == "beam_search":
            from tone.decoder import BeamSearchCTCDecoder
            logger.info("Loading BeamSearchCTCDecoder (KenLM) from %s …", MODEL_ID)
            self._decoder = BeamSearchCTCDecoder.from_hugging_face()
        else:
            from tone.decoder import GreedyCTCDecoder
            self._decoder = GreedyCTCDecoder()

        self.decoder_type = decoder
        logger.info("ASR model ready — decoder=%s  device=%s", decoder, self.device)

    # ------------------------------------------------------------------

    def new_session(self) -> "StreamingSession":
        return StreamingSession(self)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        preprocessor: AudioPreprocessor | None = None,
    ) -> str:
        """
        Batch-transcribe a full audio array.  Returns the combined text of all
        detected phrases.
        """
        from tone.logprob_splitter import StreamingLogprobSplitter

        if len(audio) == 0:
            return ""

        audio = audio.astype(np.float32)

        # Resample to 8 kHz if necessary
        if sample_rate != SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE
                )
            except ImportError:
                logger.warning("librosa unavailable — skipping resample")

        if preprocessor is not None:
            audio, _ = preprocessor.process_utterance(audio)

        # Convert to int32 PCM
        if np.abs(audio).max() <= 1.0:
            pcm = (audio * 32_767).clip(-32_768, 32_767).astype(np.int32)
        else:
            pcm = audio.clip(-32_768, 32_767).astype(np.int32)

        # Add head + tail padding (mirrors StreamingCTCPipeline.forward_offline)
        pcm = np.pad(pcm, (PADDING, PADDING))
        remainder = len(pcm) % CHUNK_SAMPLES
        if remainder:
            pcm = np.pad(pcm, (0, CHUNK_SAMPLES - remainder))

        chunks = np.split(pcm, len(pcm) // CHUNK_SAMPLES)
        state:  tuple | None  = None
        sp_state              = None
        splitter = StreamingLogprobSplitter()
        all_texts: list[str] = []

        for i, chunk in enumerate(chunks):
            chunk_in = chunk[None, :, None].astype(np.int32)
            logprobs, state = self.acoustic.forward(chunk_in, state)
            is_last = (i == len(chunks) - 1)
            phrases, sp_state = splitter.forward(
                logprobs[0].astype(np.float32), sp_state, is_last=is_last
            )
            for phrase in phrases:
                text = self._decoder.forward(phrase.logprobs)
                if text.strip():
                    all_texts.append(text)

        return " ".join(all_texts)


# ---------------------------------------------------------------------------
# 3. StreamingSession
# ---------------------------------------------------------------------------

@dataclass
class _Phrase:
    text:       str
    start_time: float
    end_time:   float


class StreamingSession:
    """
    Per-WebSocket-connection streaming state.

    Usage
    -----
        session = model.new_session()
        session.configure(sample_rate=16000)    # client sends 16 kHz
        for raw_bytes in audio_stream:
            phrases, prep_info = session.push(raw_bytes)
            for p in phrases:
                send_partial(p.text, p.start_time, p.end_time)
        final_text = session.flush()            # finalises metrics too
    """

    def __init__(self, model: ASRModel) -> None:
        from tone.logprob_splitter import StreamingLogprobSplitter
        from metrics import SessionMetrics

        self._model         = model
        self._pipeline_state: tuple | None = None   # (model_state, splitter_state)
        self._splitter      = StreamingLogprobSplitter()
        self._pcm_buffer    = np.array([], dtype=np.int32)
        self._all_phrases:  list[_Phrase] = []

        # Client-side sample rate (may differ from SAMPLE_RATE=8000)
        self._client_sr     = SAMPLE_RATE
        self._reference     = ""

        self.metrics        = SessionMetrics()
        self.metrics.start()
        self.preprocessor   = AudioPreprocessor(
            config=PreprocessorConfig(
                pre_emphasis=False,     # model applies pre-emphasis internally
                dc_removal=True,
                normalize=True,
                vad_enabled=True,
                vad_aggressiveness=2,
                denoise=False,
            ),
            sample_rate=SAMPLE_RATE,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        sample_rate: int,
        reference:   str          = "",
        preprocessing: dict | None = None,
    ) -> None:
        self._client_sr = sample_rate
        self._reference = reference
        self.metrics.sample_rate = sample_rate

        if preprocessing is not None:
            cfg = PreprocessorConfig.from_dict(preprocessing)
            # Never let the client re-enable pre-emphasis — the model does it
            cfg.pre_emphasis = False
        else:
            cfg = PreprocessorConfig(pre_emphasis=False)

        self.preprocessor.configure(cfg, sample_rate=SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Streaming push
    # ------------------------------------------------------------------

    def push(self, audio_bytes: bytes) -> tuple[list[_Phrase], dict]:
        """
        Accept raw audio bytes (float32 or int16 PCM at client sample rate).

        Runs preprocessing → resampling → 300 ms chunk inference.

        Returns:
            (newly_detected_phrases, prep_info)
            newly_detected_phrases — list[_Phrase] (may be empty)
            prep_info              — dict with speech_fraction / steps_applied
        """
        self.metrics.record_first_audio()

        # ── Decode bytes ─────────────────────────────────────────────
        try:
            audio_f32 = np.frombuffer(audio_bytes, dtype=np.float32).copy()
        except ValueError:
            audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        # Normalise int16-range to [-1, 1]
        if len(audio_f32) > 0 and np.abs(audio_f32).max() > 1.0:
            audio_f32 = audio_f32 / 32_768.0

        # ── Resample to 8 kHz ────────────────────────────────────────
        if self._client_sr != SAMPLE_RATE:
            try:
                import librosa
                audio_f32 = librosa.resample(
                    audio_f32, orig_sr=self._client_sr, target_sr=SAMPLE_RATE
                )
            except ImportError:
                pass

        # ── Preprocessing (at 8 kHz) ─────────────────────────────────
        audio_f32, prep_info = self.preprocessor.process_chunk(audio_f32)

        # Audio sample tracking uses the 8 kHz length
        self.metrics.record_audio_samples(len(audio_f32))

        # ── Float32 → int32 PCM ──────────────────────────────────────
        pcm = (audio_f32 * 32_767.0).clip(-32_768, 32_767).astype(np.int32)
        self._pcm_buffer = np.concatenate([self._pcm_buffer, pcm])

        # ── Run complete 300 ms chunks ───────────────────────────────
        new_phrases: list[_Phrase] = []
        while len(self._pcm_buffer) >= CHUNK_SAMPLES:
            chunk     = self._pcm_buffer[:CHUNK_SAMPLES]
            self._pcm_buffer = self._pcm_buffer[CHUNK_SAMPLES:]
            chunk_in  = chunk[None, :, None].astype(np.int32)

            t0 = time.perf_counter()

            model_state   = self._pipeline_state[0] if self._pipeline_state else None
            sp_state      = self._pipeline_state[1] if self._pipeline_state else None

            logprobs, model_state_next = self._model.acoustic.forward(chunk_in, model_state)
            lat_ms = (time.perf_counter() - t0) * 1_000
            self.metrics.record_chunk_latency(lat_ms)

            phrases, sp_state_next = self._splitter.forward(
                logprobs[0].astype(np.float32), sp_state
            )
            self._pipeline_state = (model_state_next, sp_state_next)

            for lp_phrase in phrases:
                text = self._model._decoder.forward(lp_phrase.logprobs)
                if text.strip():
                    p = _Phrase(
                        text       = text,
                        start_time = lp_phrase.start_frame * TorchStreamingCTCModel.FRAME_SIZE,
                        end_time   = lp_phrase.end_frame   * TorchStreamingCTCModel.FRAME_SIZE,
                    )
                    self._all_phrases.append(p)
                    new_phrases.append(p)
                    self.metrics.record_first_partial()

        return new_phrases, prep_info

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def flush(self) -> str:
        """
        Process remaining buffered audio, flush the phrase splitter, finalise
        metrics, and return the combined text of the entire utterance.
        """
        self.metrics.record_first_audio()   # guard: no-op if already set

        # Pad remaining buffer to one full chunk and run it with is_last=True
        if len(self._pcm_buffer) > 0:
            pad_len = CHUNK_SAMPLES - len(self._pcm_buffer)
            chunk = np.concatenate(
                [self._pcm_buffer, np.zeros(pad_len, dtype=np.int32)]
            )
        else:
            chunk = np.zeros(CHUNK_SAMPLES, dtype=np.int32)

        chunk_in = chunk[None, :, None].astype(np.int32)

        t0 = time.perf_counter()
        model_state = self._pipeline_state[0] if self._pipeline_state else None
        sp_state    = self._pipeline_state[1] if self._pipeline_state else None

        logprobs, _ = self._model.acoustic.forward(chunk_in, model_state)
        lat_ms = (time.perf_counter() - t0) * 1_000
        self.metrics.record_chunk_latency(lat_ms)

        # Flush with is_last=True to get remaining phrases
        phrases, _ = self._splitter.forward(
            logprobs[0].astype(np.float32), sp_state, is_last=True
        )
        for lp_phrase in phrases:
            text = self._model._decoder.forward(lp_phrase.logprobs)
            if text.strip():
                self._all_phrases.append(
                    _Phrase(
                        text       = text,
                        start_time = lp_phrase.start_frame * TorchStreamingCTCModel.FRAME_SIZE,
                        end_time   = lp_phrase.end_frame   * TorchStreamingCTCModel.FRAME_SIZE,
                    )
                )

        self._pcm_buffer = np.array([], dtype=np.int32)

        final_text = " ".join(p.text for p in self._all_phrases)
        self.metrics.finalise(final_text, self._reference)
        return final_text

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def phrases(self) -> list[_Phrase]:
        """All phrases completed so far (including from flush)."""
        return list(self._all_phrases)
