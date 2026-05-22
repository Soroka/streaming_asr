#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# ── Install the tone package from GitHub if not present ──────────────────────
# tone is not on PyPI; install directly from the Git repo.
# We check first to avoid re-installing on every restart.
if ! python3 -c "import tone" 2>/dev/null; then
    echo "[start.sh] Installing tone package from GitHub …"
    python3 -m pip install --quiet --no-cache-dir \
        "git+https://github.com/voicekit-team/T-one.git" \
        2>&1 | tail -5 \
    || {
        echo "[start.sh] GitHub install failed — trying local clone …"
        if [ -d /tmp/tone_repo ]; then
            python3 -m pip install --quiet --no-cache-dir /tmp/tone_repo --no-deps
        else
            git clone --depth 1 https://github.com/voicekit-team/T-one.git /tmp/tone_repo
            python3 -m pip install --quiet --no-cache-dir /tmp/tone_repo --no-deps
        fi
    }
    echo "[start.sh] tone package installed"
fi

# ── Auto-upgrade to GPU runtime when CUDA is present ─────────────────────────
if python3 - <<'EOF'
import subprocess, sys
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, timeout=5
    )
    if result.returncode == 0 and result.stdout.strip():
        sys.exit(0)   # CUDA present
except Exception:
    pass
sys.exit(1)  # no CUDA
EOF
then
    echo "[start.sh] CUDA GPU detected — installing GPU-accelerated torch + torchaudio"
    python3 -m pip install --quiet --no-cache-dir \
        "torch==2.2.2" "torchaudio==2.2.2" \
        --index-url https://download.pytorch.org/whl/cu121 \
        2>&1 | tail -5
    echo "[start.sh] GPU packages ready"
else
    echo "[start.sh] No CUDA GPU found — running on CPU"
fi

exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
