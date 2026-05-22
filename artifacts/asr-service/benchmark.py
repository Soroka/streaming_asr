#!/usr/bin/env python3
"""
benchmark.py — Benchmark T-one ASR on the Golos 100 h far-field dataset.

Downloads samples from HuggingFace (bond005/sberdevices_golos_100h_farfield),
streams each through the WebSocket API, and reports WER / RTF / latency
statistics — both per-sample and aggregated.

Usage
-----
  # Service already running (workflow or bare docker run):
  python3 benchmark.py

  # Auto-start the CPU Docker container, run, then stop it:
  python3 benchmark.py --docker

  # 200 samples, 4 concurrent workers, GPU container, save results:
  python3 benchmark.py --docker --docker-image t-one-asr:gpu --gpus all \\
                        --samples 200 --workers 4 --output results.json

  # Use the validation split (default); switch to test:
  python3 benchmark.py --split test --samples 500

Required packages (install once):
  pip install datasets soundfile websockets tqdm numpy
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Dependency checks ─────────────────────────────────────────────────────────

def _need(module: str, install: str) -> Any:
    try:
        import importlib
        return importlib.import_module(module)
    except ImportError:
        print(f"[benchmark] Missing dependency: {module}  →  pip install {install}")
        sys.exit(1)


_need("websockets",  "websockets>=12")
_need("tqdm",        "tqdm")
_need("soundfile",   "soundfile")
_need("datasets",    "datasets")
import numpy as np

import websockets
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset, Audio

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_ID   = "bond005/sberdevices_golos_100h_farfield"
SAMPLE_RATE  = 16_000
CHUNK_SAMPLES = 2_400   # 150 ms — matches model's native chunk size

log = logging.getLogger("benchmark")


# ── Per-sample result ─────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    index:            int
    audio_duration_s: float | None = None
    reference:        str          = ""
    hypothesis:       str          = ""
    wer:              float | None = None
    rtf:              float | None = None
    e2e_latency_ms:   float | None = None
    ttft_ms:          float | None = None
    chunk_lat_mean_ms: float | None = None
    chunk_lat_p90_ms: float | None = None
    chunks_processed: int          = 0
    error:            str | None   = None


# ── Docker helpers ────────────────────────────────────────────────────────────

def docker_start(image: str, port: int, gpus: str | None, name: str) -> None:
    """Start a detached container; raise if docker is not available."""
    cmd = ["docker", "run", "-d", "--rm",
           "--name", name,
           "-p", f"{port}:{port}",
           "-v", "t-one-model-cache:/cache"]
    if gpus:
        cmd += ["--gpus", gpus]
    cmd.append(image)
    log.info("Starting Docker container: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[benchmark] docker run failed:\n{result.stderr}")
        sys.exit(1)
    print(f"[benchmark] Container '{name}' started ({image})")


def docker_stop(name: str) -> None:
    subprocess.run(["docker", "stop", name],
                   capture_output=True, check=False)
    print(f"[benchmark] Container '{name}' stopped.")


def wait_for_service(health_url: str, timeout: int = 180) -> None:
    """Poll the health endpoint until it returns 200 or timeout expires."""
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as r:
                if r.status == 200:
                    print(f"\n[benchmark] Service ready at {health_url}")
                    return
        except Exception:
            pass
        print(".", end="", flush=True)
        dots += 1
        time.sleep(3)
    print()
    raise TimeoutError(f"Service did not become ready within {timeout}s")


# ── Audio helpers ─────────────────────────────────────────────────────────────

def decode_audio(raw: dict) -> np.ndarray:
    """
    Decode an audio dict (from datasets with decode=False) into float32 @ 16 kHz.
    raw = {"bytes": <bytes|None>, "path": <str|None>, "sampling_rate": <int|None>}
    """
    audio_bytes = raw.get("bytes")
    audio_path  = raw.get("path")

    if audio_bytes:
        buf = io.BytesIO(audio_bytes)
        arr, sr = sf.read(buf, dtype="float32", always_2d=False)
    elif audio_path and Path(audio_path).exists():
        arr, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    else:
        raise ValueError("Audio sample has neither bytes nor a valid path")

    # Mix down to mono
    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(SAMPLE_RATE, sr)
            arr = resample_poly(arr, SAMPLE_RATE // g, sr // g).astype(np.float32)
        except ImportError:
            log.warning("scipy not available — skipping resample (sr=%d)", sr)

    return arr.astype(np.float32)


# ── Single-sample transcription ───────────────────────────────────────────────

async def transcribe_one(
    sem: asyncio.Semaphore,
    ws_url: str,
    sample: dict,
    idx: int,
    timeout: float,
    preprocessing: dict | None,
) -> SampleResult:
    result = SampleResult(index=idx)
    result.reference = (sample.get("transcription") or sample.get("text") or "").strip()

    async with sem:
        try:
            audio_f32 = decode_audio(sample["audio"])
        except Exception as exc:
            result.error = f"audio decode: {exc}"
            return result

        result.audio_duration_s = round(len(audio_f32) / SAMPLE_RATE, 3)

        try:
            async with websockets.connect(ws_url, max_size=2**23) as ws:
                # ── Config ────────────────────────────────────────────
                cfg: dict[str, Any] = {
                    "type":        "config",
                    "sample_rate": SAMPLE_RATE,
                    "reference":   result.reference,
                }
                if preprocessing:
                    cfg["preprocessing"] = preprocessing
                await ws.send(json.dumps(cfg))
                await asyncio.wait_for(ws.recv(), timeout=10.0)   # ack

                # ── Stream audio ──────────────────────────────────────
                for i in range(0, len(audio_f32), CHUNK_SAMPLES):
                    chunk = audio_f32[i: i + CHUNK_SAMPLES]
                    await ws.send(chunk.tobytes())

                await ws.send(json.dumps({"type": "end"}))

                # ── Collect final result ──────────────────────────────
                deadline = asyncio.get_event_loop().time() + timeout
                while asyncio.get_event_loop().time() < deadline:
                    remaining = deadline - asyncio.get_event_loop().time()
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
                    if msg.get("is_final"):
                        m = msg.get("metrics", {})
                        result.hypothesis      = msg.get("text", "")
                        result.wer             = m.get("wer")
                        result.rtf             = m.get("rtf")
                        result.e2e_latency_ms  = m.get("e2e_latency_ms")
                        result.ttft_ms         = m.get("ttft_ms")
                        cl = m.get("chunk_latency_ms") or {}
                        result.chunk_lat_mean_ms = cl.get("mean")
                        result.chunk_lat_p90_ms  = cl.get("p90")
                        result.chunks_processed  = m.get("chunks_processed", 0)
                        break

        except asyncio.TimeoutError:
            result.error = f"timeout after {timeout}s"
        except Exception as exc:
            result.error = str(exc)

    return result


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_samples(dataset_id: str, split: str, n: int, shuffle: bool, seed: int):
    """
    Stream samples from HuggingFace.
    Uses decode=False to avoid the torchcodec requirement; audio is decoded
    manually with soundfile.
    Returns a list of up to `n` samples.
    """
    print(f"[benchmark] Loading '{dataset_id}' split='{split}' …")
    ds = load_dataset(dataset_id, split=split, streaming=True)
    # Disable the Audio decoder; we decode bytes manually with soundfile
    ds = ds.cast_column("audio", Audio(decode=False))

    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=1_000)

    samples: list[dict] = []
    with tqdm(total=n, desc="Fetching samples", unit="sample") as bar:
        for sample in ds:
            samples.append(sample)
            bar.update(1)
            if len(samples) >= n:
                break

    print(f"[benchmark] Fetched {len(samples)} samples")
    return samples


# ── Aggregation ───────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    def pct(p): return s[max(0, int(n * p / 100) - 1)]
    return {
        "n":    n,
        "mean": round(statistics.mean(s), 4),
        "std":  round(statistics.stdev(s), 4) if n > 1 else 0.0,
        "min":  round(s[0], 4),
        "p50":  round(pct(50), 4),
        "p90":  round(pct(90), 4),
        "p99":  round(pct(99), 4),
        "max":  round(s[-1], 4),
    }


def aggregate(results: list[SampleResult]) -> dict:
    ok  = [r for r in results if r.error is None]
    err = [r for r in results if r.error is not None]

    def vals(attr):
        return [getattr(r, attr) for r in ok if getattr(r, attr) is not None]

    total_audio_s = sum(r.audio_duration_s or 0 for r in ok)

    return {
        "summary": {
            "total_samples":      len(results),
            "successful_samples": len(ok),
            "failed_samples":     len(err),
            "total_audio_s":      round(total_audio_s, 2),
        },
        "wer":              _stats(vals("wer")),
        "rtf":              _stats(vals("rtf")),
        "e2e_latency_ms":   _stats(vals("e2e_latency_ms")),
        "ttft_ms":          _stats(vals("ttft_ms")),
        "chunk_lat_mean_ms": _stats(vals("chunk_lat_mean_ms")),
        "chunk_lat_p90_ms": _stats(vals("chunk_lat_p90_ms")),
        "errors": [{"index": r.index, "error": r.error} for r in err],
    }


# ── Console reporting ─────────────────────────────────────────────────────────

def _row(label: str, stats: dict) -> str:
    if not stats or stats.get("n", 0) == 0:
        return f"  {label:<26}  (no data)"
    return (
        f"  {label:<26}  "
        f"mean={stats['mean']:>9.4f}  "
        f"p50={stats['p50']:>9.4f}  "
        f"p90={stats['p90']:>9.4f}  "
        f"p99={stats['p99']:>9.4f}  "
        f"std={stats['std']:>8.4f}"
    )


def print_report(agg: dict, results: list[SampleResult]) -> None:
    s = agg["summary"]
    print()
    print("═" * 82)
    print("  T-one ASR Benchmark — Results")
    print("═" * 82)
    print(f"  Samples: {s['successful_samples']} ok / {s['failed_samples']} failed "
          f"(total {s['total_samples']})   |   Audio: {s['total_audio_s']:.1f}s")
    print("─" * 82)
    print(f"  {'Metric':<26}  {'mean':>11}  {'p50':>11}  {'p90':>11}  {'p99':>11}  {'std':>10}")
    print("─" * 82)
    print(_row("WER",                   agg["wer"]))
    print(_row("RTF",                   agg["rtf"]))
    print(_row("E2E latency (ms)",       agg["e2e_latency_ms"]))
    print(_row("TTFT (ms)",             agg["ttft_ms"]))
    print(_row("Chunk latency mean(ms)", agg["chunk_lat_mean_ms"]))
    print(_row("Chunk latency p90 (ms)", agg["chunk_lat_p90_ms"]))
    print("─" * 82)

    if agg["errors"]:
        print(f"\n  Failed samples ({len(agg['errors'])}):")
        for e in agg["errors"][:10]:
            print(f"    #{e['index']:>4}  {e['error']}")
        if len(agg["errors"]) > 10:
            print(f"    … and {len(agg['errors']) - 10} more")

    # Show worst WER samples
    ok = sorted(
        [r for r in results if r.error is None and r.wer is not None],
        key=lambda r: r.wer or 0, reverse=True
    )
    if ok:
        print(f"\n  Worst 5 WER samples:")
        for r in ok[:5]:
            ref_short = (r.reference[:60] + "…") if len(r.reference) > 60 else r.reference
            hyp_short = (r.hypothesis[:60] + "…") if len(r.hypothesis) > 60 else r.hypothesis
            print(f"    #{r.index:>4}  WER={r.wer:.3f}  dur={r.audio_duration_s:.1f}s")
            print(f"          REF: {ref_short}")
            print(f"          HYP: {hyp_short}")

    print("═" * 82)


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_benchmark(args: argparse.Namespace) -> None:
    ws_url     = f"ws://{args.host}:{args.port}/ws/transcribe"
    health_url = f"http://{args.host}:{args.port}/health"
    container_name = f"t-one-benchmark-{os.getpid()}"

    # ── Docker lifecycle ───────────────────────────────────────────────
    container_started = False
    if args.docker:
        docker_start(args.docker_image, args.port,
                     args.gpus or None, container_name)
        container_started = True
        try:
            wait_for_service(health_url, timeout=args.startup_timeout)
        except TimeoutError as exc:
            print(f"[benchmark] {exc}")
            docker_stop(container_name)
            sys.exit(1)
    else:
        # Check the service is reachable before we start downloading data
        try:
            wait_for_service(health_url, timeout=10)
        except TimeoutError:
            print(f"[benchmark] Service not reachable at {health_url}")
            print(f"            Start the service first, or use --docker to auto-start it.")
            sys.exit(1)

    # ── Load dataset ───────────────────────────────────────────────────
    samples = load_samples(
        args.dataset, args.split, args.samples,
        shuffle=args.shuffle, seed=args.seed,
    )

    # ── Parse optional preprocessing overrides ─────────────────────────
    preprocessing: dict | None = None
    if args.preprocessing:
        try:
            preprocessing = json.loads(args.preprocessing)
        except json.JSONDecodeError as exc:
            print(f"[benchmark] --preprocessing must be valid JSON: {exc}")
            sys.exit(1)

    # ── Run async workers ──────────────────────────────────────────────
    sem = asyncio.Semaphore(args.workers)
    tasks = [
        transcribe_one(sem, ws_url, sample, idx,
                       timeout=args.timeout, preprocessing=preprocessing)
        for idx, sample in enumerate(samples)
    ]

    print(f"\n[benchmark] Transcribing {len(tasks)} samples "
          f"({args.workers} worker(s)) …\n")

    results: list[SampleResult] = []
    pbar = tqdm(total=len(tasks), unit="sample", dynamic_ncols=True)
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        status = "✗" if r.error else "✓"
        wer_str = f"WER={r.wer:.3f}" if r.wer is not None else "WER=n/a"
        rtf_str = f"RTF={r.rtf:.3f}" if r.rtf is not None else ""
        err_str = f"  err={r.error}" if r.error else ""
        pbar.set_postfix_str(f"#{r.index:>4} {status} {wer_str} {rtf_str}{err_str}",
                             refresh=True)
        pbar.update(1)
    pbar.close()

    # Sort by original index for reproducible output
    results.sort(key=lambda r: r.index)

    # ── Aggregate & report ─────────────────────────────────────────────
    agg = aggregate(results)
    print_report(agg, results)

    # ── Save output ────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        payload = {
            "args": {
                "dataset":    args.dataset,
                "split":      args.split,
                "samples":    args.samples,
                "workers":    args.workers,
                "docker":     args.docker,
                "docker_image": args.docker_image,
                "preprocessing": preprocessing,
            },
            "aggregate":    agg,
            "per_sample":   [asdict(r) for r in results],
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n[benchmark] Results saved → {out_path}")

    if args.csv:
        import csv
        csv_path = Path(args.csv)
        fieldnames = list(asdict(results[0]).keys()) if results else []
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"[benchmark] Per-sample CSV saved → {csv_path}")

    # ── Cleanup ────────────────────────────────────────────────────────
    if container_started and not args.keep_container:
        docker_stop(container_name)


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Benchmark T-one ASR on bond005/sberdevices_golos_100h_farfield",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Service connection ─────────────────────────────────────────────
    g = p.add_argument_group("Service")
    g.add_argument("--host",    default="localhost",
                   help="Host where the ASR service is running")
    g.add_argument("--port",    default=8000, type=int,
                   help="Port of the ASR service")
    g.add_argument("--timeout", default=120.0, type=float,
                   help="Per-sample timeout in seconds")

    # ── Docker ────────────────────────────────────────────────────────
    g = p.add_argument_group("Docker")
    g.add_argument("--docker", action="store_true",
                   help="Auto-start the Docker container before the benchmark")
    g.add_argument("--docker-image", default="t-one-asr:cpu",
                   help="Docker image to start")
    g.add_argument("--gpus",   default=None,
                   help="Value for --gpus flag (e.g. 'all')")
    g.add_argument("--startup-timeout", default=180, type=int,
                   help="Seconds to wait for container health check on start")
    g.add_argument("--keep-container", action="store_true",
                   help="Do not stop the container after the benchmark")

    # ── Dataset ───────────────────────────────────────────────────────
    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset", default=DATASET_ID,
                   help="HuggingFace dataset ID")
    g.add_argument("--split",   default="validation",
                   choices=["train", "validation", "test"],
                   help="Dataset split to use")
    g.add_argument("--samples", default=100, type=int,
                   help="Maximum number of samples to test")
    g.add_argument("--shuffle", action="store_true",
                   help="Shuffle the dataset before taking --samples")
    g.add_argument("--seed",    default=42, type=int,
                   help="Random seed used when --shuffle is set")

    # ── Preprocessing ─────────────────────────────────────────────────
    g = p.add_argument_group("Preprocessing")
    g.add_argument(
        "--preprocessing", default=None,
        help=(
            'JSON string of per-session preprocessing overrides, e.g. '
            '\'{"vad_aggressiveness":3,"denoise":true}\'. '
            'Defaults match the server defaults (all steps on, denoise off).'
        ),
    )

    # ── Concurrency ───────────────────────────────────────────────────
    g = p.add_argument_group("Concurrency")
    g.add_argument("--workers", default=1, type=int,
                   help="Number of concurrent WebSocket connections")

    # ── Output ────────────────────────────────────────────────────────
    g = p.add_argument_group("Output")
    g.add_argument("--output", default=None, metavar="FILE.json",
                   help="Save full results (aggregate + per-sample) to a JSON file")
    g.add_argument("--csv",    default=None, metavar="FILE.csv",
                   help="Save per-sample results to a CSV file")

    args = p.parse_args()

    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\n[benchmark] Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
