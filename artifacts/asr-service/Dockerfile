# ─────────────────────────────────────────────────────────────────────────────
# T-one Streaming ASR Service — CPU image
# ─────────────────────────────────────────────────────────────────────────────
# Two-stage build:
#   builder  — installs gcc/python-dev to compile C extensions (webrtcvad)
#   runtime  — minimal image with only runtime system libs
#
# Usage:
#   docker build -t t-one-asr .
#   docker run -p 8000:8000 -v t-one-model-cache:/cache t-one-asr
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --upgrade pip wheel

COPY requirements.txt .

# Install everything into a prefix directory so we can copy it cleanly
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="T-one Streaming ASR (CPU)"
LABEL org.opencontainers.image.description="Low-latency Russian speech-to-text via ONNX Runtime"

# Runtime system libraries
#   libsndfile1  — required by soundfile (audio I/O)
#   libgomp1     — OpenMP, required by numpy / torch / librosa
#   curl         — used by HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy service source code
COPY main.py model.py preprocessing.py metrics.py test_client.py ./

# ── HuggingFace model cache ───────────────────────────────────────────────────
# Mount a named volume here at runtime to persist the downloaded ONNX model
# (avoids re-downloading ~250 MB on every container start).
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface
RUN mkdir -p /cache/huggingface

# ── Optional HF token (set at runtime, never baked in) ───────────────────────
# Pass via: docker run -e HF_TOKEN=hf_xxx ...
ENV HF_TOKEN=""

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1000 asr \
    && chown -R asr:asr /app /cache
USER asr

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
