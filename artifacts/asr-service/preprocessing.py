"""
Audio preprocessing pipeline for the T-one ASR service.

Each step is independently toggleable via PreprocessorConfig.
The pipeline is applied:
  - per-chunk in StreamingSession.push()   (streaming path)
  - to the whole utterance in ASRModel.transcribe()  (batch path)

Steps (in order):
  1. DC offset removal         — subtract signal mean
  2. Pre-emphasis filter       — y[n] = y[n] - α·y[n-1]  (α=0.97)
  3. RMS normalisation         — scale to target dBFS level
  4. VAD (webrtcvad)           — detect voiced frames; returns a speech mask
  5. Noise reduction           — spectral gating via noisereduce (opt-in)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorConfig:
    # ── Step 1: DC offset ──────────────────────────────────────────────
    dc_removal: bool = True

    # ── Step 2: Pre-emphasis ───────────────────────────────────────────
    pre_emphasis: bool = True
    pre_emphasis_coef: float = 0.97   # standard ASR value

    # ── Step 3: RMS normalisation ──────────────────────────────────────
    normalize: bool = True
    normalize_target_dbfs: float = -20.0
    normalize_min_rms: float = 1e-4   # skip if audio is too quiet (silence)

    # ── Step 4: VAD ────────────────────────────────────────────────────
    vad_enabled: bool = True
    # 0 = least aggressive (keep most audio), 3 = most aggressive
    vad_aggressiveness: int = 2
    # Frame length fed to webrtcvad: 10, 20, or 30 ms
    vad_frame_ms: int = 30
    # Fraction of frames that must be voiced for the chunk to be kept.
    # Chunks below this threshold are flagged as silent (not dropped — the
    # model still runs, but the caller knows the chunk is mostly silence).
    vad_speech_ratio: float = 0.2

    # ── Step 5: Noise reduction ────────────────────────────────────────
    denoise: bool = False             # CPU-heavy — disabled by default
    denoise_stationary: bool = False  # True = fan/HVAC; False = mixed noise
    denoise_prop_decrease: float = 1.0  # 0-1: how aggressively to suppress noise

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessorConfig":
        """Build a config from a (possibly partial) dict, keeping defaults."""
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0


def _to_int16(audio_f32: np.ndarray) -> bytes:
    """Convert float32 [-1,1] to int16 PCM bytes for webrtcvad."""
    pcm = (audio_f32 * 32767.0).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# Core processing functions (stateless, operate on np.ndarray float32)
# ---------------------------------------------------------------------------

def apply_dc_removal(audio: np.ndarray) -> np.ndarray:
    return audio - np.mean(audio)


def apply_pre_emphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    if len(audio) < 2:
        return audio
    return np.append(audio[0], audio[1:] - coef * audio[:-1]).astype(np.float32)


def apply_normalization(
    audio: np.ndarray,
    target_dbfs: float = -20.0,
    min_rms: float = 1e-4,
) -> np.ndarray:
    rms = _rms(audio)
    if rms < min_rms:
        return audio  # silent frame — skip
    target_rms = 10 ** (target_dbfs / 20.0)
    return (audio * (target_rms / rms)).astype(np.float32)


def apply_vad(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    speech_ratio: float = 0.2,
) -> tuple[np.ndarray, float]:
    """
    Run webrtcvad over the audio.
    Returns (audio unchanged, speech_fraction).
    speech_fraction is the ratio of voiced frames to total frames (0–1).
    """
    try:
        import webrtcvad
    except ImportError:
        logger.warning("webrtcvad not installed — VAD skipped")
        return audio, 1.0

    vad = webrtcvad.Vad(aggressiveness)
    frame_samples = int(sample_rate * frame_ms / 1000)
    total_frames = 0
    speech_frames = 0

    for start in range(0, len(audio) - frame_samples + 1, frame_samples):
        frame = audio[start: start + frame_samples]
        try:
            is_speech = vad.is_speech(_to_int16(frame), sample_rate)
        except Exception:
            is_speech = True   # on error, assume speech
        total_frames += 1
        if is_speech:
            speech_frames += 1

    speech_fraction = speech_frames / total_frames if total_frames > 0 else 0.0
    return audio, speech_fraction


def apply_denoise(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    stationary: bool = False,
    prop_decrease: float = 1.0,
) -> np.ndarray:
    """
    Apply spectral-gating noise reduction via noisereduce.
    Works best on full utterances; safe to call on any length ≥ ~0.5 s.
    """
    try:
        import noisereduce as nr
    except ImportError:
        logger.warning("noisereduce not installed — denoising skipped")
        return audio

    if len(audio) < sample_rate * 0.1:   # < 100 ms — skip
        return audio

    return nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=stationary,
        prop_decrease=prop_decrease,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# AudioPreprocessor — stateful, streaming-aware
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """
    Stateful per-session preprocessor.

    Streaming path  (process_chunk):
        Applies dc_removal → pre_emphasis → normalize → VAD.
        Optionally applies denoise using a noise profile estimated from
        the first `noise_est_s` seconds of audio.

    Full-utterance path  (process_utterance):
        Applies all enabled steps to the complete signal.
    """

    def __init__(self, config: PreprocessorConfig | None = None, sample_rate: int = SAMPLE_RATE):
        self.config = config or PreprocessorConfig()
        self.sample_rate = sample_rate

        # Streaming noise estimation state
        self._noise_buffer: list[np.ndarray] = []
        self._noise_profile_ready = False
        self._noise_est_s = 1.5   # seconds to collect before estimating noise

    def configure(self, config: PreprocessorConfig, sample_rate: int = SAMPLE_RATE) -> None:
        self.config = config
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def process_chunk(self, audio: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Preprocess a single audio chunk (any length) for streaming inference.
        Returns (processed_audio, info_dict).

        info_dict keys:
          speech_fraction  — 0.0–1.0 VAD score (1.0 if VAD disabled)
          is_silent        — True if speech_fraction < vad_speech_ratio
          steps_applied    — list of step names that ran
        """
        cfg = self.config
        info: dict = {"steps_applied": [], "speech_fraction": 1.0, "is_silent": False}

        if len(audio) == 0:
            return audio, info

        # 1. DC removal
        if cfg.dc_removal:
            audio = apply_dc_removal(audio)
            info["steps_applied"].append("dc_removal")

        # 2. Pre-emphasis
        if cfg.pre_emphasis:
            audio = apply_pre_emphasis(audio, cfg.pre_emphasis_coef)
            info["steps_applied"].append("pre_emphasis")

        # 3. Normalisation
        if cfg.normalize:
            audio = apply_normalization(audio, cfg.normalize_target_dbfs, cfg.normalize_min_rms)
            info["steps_applied"].append("normalize")

        # 4. VAD
        if cfg.vad_enabled:
            audio, speech_frac = apply_vad(
                audio,
                sample_rate=self.sample_rate,
                aggressiveness=cfg.vad_aggressiveness,
                frame_ms=cfg.vad_frame_ms,
                speech_ratio=cfg.vad_speech_ratio,
            )
            info["speech_fraction"] = round(speech_frac, 3)
            info["is_silent"] = speech_frac < cfg.vad_speech_ratio
            info["steps_applied"].append("vad")

        # 5. Streaming denoise (stateful noise-profile estimation)
        if cfg.denoise:
            audio = self._streaming_denoise(audio)
            info["steps_applied"].append("denoise")

        return audio, info

    def _streaming_denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Collect audio into a noise-estimation buffer until we have enough,
        then apply spectral subtraction on all subsequent chunks.
        """
        try:
            import noisereduce as nr
        except ImportError:
            return audio

        cfg = self.config
        noise_est_samples = int(self._noise_est_s * self.sample_rate)

        if not self._noise_profile_ready:
            self._noise_buffer.append(audio)
            buffered = sum(len(b) for b in self._noise_buffer)
            if buffered >= noise_est_samples:
                self._noise_profile_ready = True
                logger.debug("Streaming noise profile estimated from %.1f s of audio", buffered / self.sample_rate)
            return audio   # return unprocessed until profile is ready

        noise_clip = np.concatenate(self._noise_buffer[:3])  # use first few chunks as noise reference
        return nr.reduce_noise(
            y=audio,
            y_noise=noise_clip,
            sr=self.sample_rate,
            stationary=cfg.denoise_stationary,
            prop_decrease=cfg.denoise_prop_decrease,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Full utterance
    # ------------------------------------------------------------------

    def process_utterance(self, audio: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Preprocess a complete utterance for batch inference.
        All enabled steps are applied, including full-signal denoising.
        Returns (processed_audio, info_dict).
        """
        cfg = self.config
        info: dict = {"steps_applied": [], "speech_fraction": 1.0, "is_silent": False}

        if len(audio) == 0:
            return audio, info

        if cfg.dc_removal:
            audio = apply_dc_removal(audio)
            info["steps_applied"].append("dc_removal")

        if cfg.pre_emphasis:
            audio = apply_pre_emphasis(audio, cfg.pre_emphasis_coef)
            info["steps_applied"].append("pre_emphasis")

        if cfg.normalize:
            audio = apply_normalization(audio, cfg.normalize_target_dbfs, cfg.normalize_min_rms)
            info["steps_applied"].append("normalize")

        if cfg.vad_enabled:
            audio, speech_frac = apply_vad(
                audio,
                sample_rate=self.sample_rate,
                aggressiveness=cfg.vad_aggressiveness,
                frame_ms=cfg.vad_frame_ms,
            )
            info["speech_fraction"] = round(speech_frac, 3)
            info["is_silent"] = speech_frac < cfg.vad_speech_ratio
            info["steps_applied"].append("vad")

        if cfg.denoise:
            audio = apply_denoise(
                audio,
                sample_rate=self.sample_rate,
                stationary=cfg.denoise_stationary,
                prop_decrease=cfg.denoise_prop_decrease,
            )
            info["steps_applied"].append("denoise")

        return audio, info
