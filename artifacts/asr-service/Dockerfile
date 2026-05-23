# ─────────────────────────────────────────────────────────────────────────────
# T-one Streaming ASR Service — CPU image
# ─────────────────────────────────────────────────────────────────────────────
# Two-stage build:
#   builder  — compiles C/C++ extensions (webrtcvad, kenlm) and clones the
#              tone package from GitHub, then installs everything into /install
#   runtime  — minimal image; only standard runtime system libs required
#
# Usage:
#   docker build -t t-one-asr .
#   docker run -p 8000:8000 -v t-one-model-cache:/cache t-one-asr
#
# The first run downloads model weights (~500 MB) + KenLM binary (~5.5 GB)
# from HuggingFace and caches them in the named volume.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        python3-dev \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --upgrade pip wheel

# Clone the tone package from GitHub (not on PyPI).
# --depth 1 keeps the layer small (~5 MB).
RUN git clone --depth 1 https://github.com/voicekit-team/T-one.git /build/tone_repo

# Install all Python dependencies (torch CPU, onnxruntime, pyctcdecode, kenlm, …)
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Install the tone package itself (all its deps are already installed above)
RUN pip install --prefix=/install --no-cache-dir /build/tone_repo --no-deps


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="T-one Streaming ASR (CPU)"
LABEL org.opencontainers.image.description="Low-latency Russian speech-to-text — PyTorch + KenLM beam-search"

# Runtime system libraries:
#   libsndfile1  — required by soundfile (audio I/O)
#   libgomp1     — OpenMP, required by torch / numpy
#   curl         — used by HEALTHCHECK
# kenlm only links libc/libstdc++/libgcc_s — no extra boost libs needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all installed Python packages from the builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy service source files (include benchmark.py so it can run inside the container)
COPY main.py model.py preprocessing.py metrics.py test_client.py benchmark.py ./

# ── HuggingFace model cache ───────────────────────────────────────────────────
# Mount a named volume here at runtime to persist the downloaded weights +
# KenLM binary (avoids re-downloading ~6 GB on every container restart).
#   docker run -v t-one-model-cache:/cache t-one-asr
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface
RUN mkdir -p /cache/huggingface

# ── Optional HF token (set at runtime, never baked in) ───────────────────────
# docker run -e HF_TOKEN=hf_xxx ...
ENV HF_TOKEN=""

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1000 asr \
    && chown -R asr:asr /app /cache
USER asr

EXPOSE 8000

# Allow up to 3 minutes for the first start (weight + KenLM download).
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=6 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
