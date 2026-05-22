"""
Metrics for the T-one ASR service.

Per-session metrics
-------------------
Collected inside StreamingSession and finalised on flush():
  - chunk_latencies_ms   : inference time for each 150 ms ONNX chunk
  - total_audio_s        : seconds of audio processed (excl. padding)
  - ttft_ms              : time from first audio byte to first non-empty partial
  - e2e_latency_ms       : wall-clock time from first audio byte to final result
  - rtf                  : total_inference_time_s / total_audio_s  (<1 = faster than real-time)
  - wer                  : word error rate vs optional reference transcript

Global aggregate metrics
------------------------
GlobalMetrics keeps a rolling window of the last MAX_HISTORY sessions and
exposes aggregate statistics (mean, p50, p90, p99) for each numeric metric.
Thread-safe via threading.Lock.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate using Levenshtein distance on word sequences.
    Returns a value in [0, ∞) where 0.0 is perfect and 1.0 is 100 % errors.
    Values >1 are possible when the hypothesis is longer than the reference.
    """
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    N = len(ref)
    if N == 0:
        return 0.0 if len(hyp) == 0 else float(len(hyp))

    M = len(hyp)
    dp = list(range(M + 1))
    for i in range(1, N + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, M + 1):
            temp = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[M] / N


# ---------------------------------------------------------------------------
# Per-session metrics
# ---------------------------------------------------------------------------

@dataclass
class SessionMetrics:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # timing
    _wall_start: float = field(default=0.0, repr=False)
    _first_audio_ts: float = field(default=0.0, repr=False)
    _first_partial_ts: float = field(default=0.0, repr=False)
    _end_ts: float = field(default=0.0, repr=False)

    # per-chunk inference latencies (ms)
    chunk_latencies_ms: list[float] = field(default_factory=list)

    # audio bookkeeping
    total_audio_samples: int = 0
    sample_rate: int = 16000

    # transcript
    hypothesis: str = ""
    reference: str = ""

    # computed on finalise()
    ttft_ms: float | None = None
    e2e_latency_ms: float | None = None
    rtf: float | None = None
    wer: float | None = None

    def start(self) -> None:
        self._wall_start = time.perf_counter()

    def record_first_audio(self) -> None:
        if self._first_audio_ts == 0.0:
            self._first_audio_ts = time.perf_counter()

    def record_chunk_latency(self, latency_ms: float) -> None:
        self.chunk_latencies_ms.append(latency_ms)

    def record_audio_samples(self, n: int) -> None:
        self.total_audio_samples += n

    def record_first_partial(self) -> None:
        if self._first_partial_ts == 0.0:
            self._first_partial_ts = time.perf_counter()

    def finalise(self, hypothesis: str, reference: str = "") -> None:
        self._end_ts = time.perf_counter()
        self.hypothesis = hypothesis
        self.reference = reference

        t0 = self._first_audio_ts or self._wall_start

        if self._first_partial_ts > 0.0:
            self.ttft_ms = (self._first_partial_ts - t0) * 1_000

        self.e2e_latency_ms = (self._end_ts - t0) * 1_000

        total_audio_s = self.total_audio_samples / max(1, self.sample_rate)
        total_infer_s = sum(self.chunk_latencies_ms) / 1_000
        self.rtf = total_infer_s / total_audio_s if total_audio_s > 0 else None

        if reference:
            self.wer = compute_wer(reference, hypothesis)

    def summary(self) -> dict:
        lats = self.chunk_latencies_ms
        return {
            "session_id": self.session_id,
            "audio_duration_s": round(self.total_audio_samples / max(1, self.sample_rate), 3),
            "chunks_processed": len(lats),
            "chunk_latency_ms": {
                "mean": round(_mean(lats), 2) if lats else None,
                "p50":  round(_percentile(lats, 50), 2) if lats else None,
                "p90":  round(_percentile(lats, 90), 2) if lats else None,
                "p99":  round(_percentile(lats, 99), 2) if lats else None,
                "max":  round(max(lats), 2) if lats else None,
            },
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "e2e_latency_ms": round(self.e2e_latency_ms, 2) if self.e2e_latency_ms is not None else None,
            "rtf": round(self.rtf, 4) if self.rtf is not None else None,
            "wer": round(self.wer, 4) if self.wer is not None else None,
            "hypothesis": self.hypothesis,
            "reference": self.reference or None,
        }


# ---------------------------------------------------------------------------
# Global rolling metrics
# ---------------------------------------------------------------------------

MAX_HISTORY = 500


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(len(s) * p / 100) - 1)
    return s[idx]


class GlobalMetrics:
    """Thread-safe rolling aggregate of completed ASR sessions."""

    def __init__(self, max_history: int = MAX_HISTORY) -> None:
        self._lock = threading.Lock()
        self._history: Deque[dict] = deque(maxlen=max_history)
        self._total_sessions: int = 0
        self._error_sessions: int = 0

    def record(self, sm: SessionMetrics) -> None:
        with self._lock:
            self._total_sessions += 1
            self._history.append(sm.summary())

    def record_error(self) -> None:
        with self._lock:
            self._error_sessions += 1

    def snapshot(self, last_n: int = 20) -> dict:
        with self._lock:
            history = list(self._history)
            total = self._total_sessions
            errors = self._error_sessions

        def _agg(key: str) -> dict | None:
            vals = [s[key] for s in history if s.get(key) is not None]
            if not vals:
                return None
            return {
                "mean": round(_mean(vals), 4),
                "p50":  round(_percentile(vals, 50), 4),
                "p90":  round(_percentile(vals, 90), 4),
                "p99":  round(_percentile(vals, 99), 4),
            }

        def _agg_nested(outer: str, inner: str) -> dict | None:
            vals = [
                s[outer][inner]
                for s in history
                if s.get(outer) and s[outer].get(inner) is not None
            ]
            if not vals:
                return None
            return {
                "mean": round(_mean(vals), 4),
                "p50":  round(_percentile(vals, 50), 4),
                "p90":  round(_percentile(vals, 90), 4),
                "p99":  round(_percentile(vals, 99), 4),
            }

        wer_sessions = [s for s in history if s.get("wer") is not None]

        return {
            "sessions": {
                "total": total,
                "in_window": len(history),
                "errors": errors,
            },
            "chunk_latency_ms": _agg_nested("chunk_latency_ms", "mean"),
            "ttft_ms":          _agg("ttft_ms"),
            "e2e_latency_ms":   _agg("e2e_latency_ms"),
            "rtf":              _agg("rtf"),
            "wer": {
                "sessions_with_reference": len(wer_sessions),
                **(_agg("wer") or {}),
            },
            "recent_sessions": history[-last_n:],
        }


# Module-level singleton
global_metrics = GlobalMetrics()
