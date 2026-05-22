#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# ── Auto-install GPU runtime when CUDA is present ────────────────────────────
if python3 - <<'EOF'
import subprocess, sys
# Check if a CUDA device is actually visible
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
    echo "[start.sh] CUDA GPU detected — ensuring onnxruntime-gpu is installed"
    python3 -m pip install --quiet --no-cache-dir \
        "torch==2.2.2" --index-url https://download.pytorch.org/whl/cu121 \
        onnxruntime-gpu \
        2>&1 | tail -5
    echo "[start.sh] GPU packages ready"
else
    echo "[start.sh] No CUDA GPU found — running on CPU"
fi

exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
