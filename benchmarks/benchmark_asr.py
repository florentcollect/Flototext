"""Benchmark Qwen3-ASR vs NVIDIA Canary 1B v2 on French speech.

Pulls a fixed sample of the public Google FLEURS dataset (French, `fr_fr`,
which ships a reference transcription per clip), runs each ASR backend over the
same clips, and reports WER, latency, RTF and peak GPU memory so we can decide
objectively whether Canary should replace Qwen in Flototext.

Why FLEURS via parquet (and not the `datasets` library): `requirements.txt`
pins `huggingface-hub<1`, and recent `datasets` would drag in `hub>=1`. We grab
the auto-converted parquet directly with `huggingface_hub.hf_hub_download`.

Usage (inside the qwen-asr venv):
    python -m benchmarks.benchmark_asr
    python -m benchmarks.benchmark_asr --num-clips 50 --backends qwen canary
    python -m benchmarks.benchmark_asr --lang en   # uses fr_fr by default

Outputs:
    benchmarks/results/asr_bench_<timestamp>.csv   (per-clip detail)
    benchmarks/results/asr_bench_<timestamp>.md    (summary table + verdict)
"""

import argparse
import io
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from flototext.config import config
from flototext.core.asr_backends import create_backend


# --------------------------------------------------------------------------- #
# Language wiring: each backend wants the language expressed differently.
# --------------------------------------------------------------------------- #
LANG_PRESETS = {
    "fr": {"fleurs_config": "fr_fr", "label": "French", "code": "fr"},
    "en": {"fleurs_config": "en_us", "label": "English", "code": "en"},
}

TARGET_SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    clip_id: str
    audio: np.ndarray          # mono float32, 16 kHz, normalized to [-1, 1]
    duration: float            # seconds
    reference: str             # ground-truth transcription (normalized by FLEURS)


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz mono float32 if needed."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, TARGET_SAMPLE_RATE)
        audio = resample_poly(audio, TARGET_SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Match the in-app normalization (peak to [-1, 1])."""
    max_val = float(np.abs(audio).max()) if audio.size else 0.0
    if max_val > 0:
        audio = audio / max_val
    return audio.astype(np.float32)


def load_fleurs_clips(lang: str, num_clips: int, cache_dir: Path) -> List[Clip]:
    """Download FLEURS parquet for `lang` and decode the first `num_clips` rows."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    fleurs_config = LANG_PRESETS[lang]["fleurs_config"]
    repo_id = "google/fleurs"
    revision = "refs/convert/parquet"

    print(f"Locating FLEURS parquet ({fleurs_config}/test) on {repo_id} ...")
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    prefix = f"{fleurs_config}/test"
    parquet_files = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not parquet_files:
        raise RuntimeError(
            f"No parquet found for {prefix} on {repo_id}@{revision}. "
            f"Available sample: {[f for f in files[:10]]}"
        )

    clips: List[Clip] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    for pf in parquet_files:
        if len(clips) >= num_clips:
            break
        print(f"  downloading {pf} ...")
        local = hf_hub_download(
            repo_id, pf, repo_type="dataset", revision=revision, cache_dir=str(cache_dir)
        )
        table = pq.read_table(local)
        cols = table.column_names
        # FLEURS columns: id, num_samples, path, audio{bytes,path}, transcription,
        # raw_transcription, gender, lang_id, language, lang_group_id
        rows = table.to_pylist()
        for row in rows:
            if len(clips) >= num_clips:
                break
            ref = (row.get("transcription") or "").strip()
            audio_field = row.get("audio")
            if not ref or not audio_field:
                continue
            audio_bytes = audio_field.get("bytes") if isinstance(audio_field, dict) else None
            if not audio_bytes:
                continue
            waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            waveform = _normalize_audio(_resample(waveform, sr))
            duration = len(waveform) / TARGET_SAMPLE_RATE
            clip_id = str(row.get("id") or row.get("path") or len(clips))
            clips.append(Clip(clip_id=clip_id, audio=waveform, duration=duration, reference=ref))

    print(f"Loaded {len(clips)} clips ({'cols=' + ','.join(cols)}).")
    return clips


# --------------------------------------------------------------------------- #
# Text normalization for WER
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace; keep accented letters."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# GPU memory helper (whole-GPU usage, comparable across torch/onnxruntime)
# --------------------------------------------------------------------------- #
def gpu_used_gb() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Benchmark core
# --------------------------------------------------------------------------- #
@dataclass
class BackendResult:
    name: str
    load_seconds: float
    hyps: List[str] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    peak_gpu_gb: float = 0.0
    error: Optional[str] = None


def run_backend(name: str, clips: List[Clip], label: str, code: str) -> BackendResult:
    """Load a backend, transcribe every clip, and collect metrics."""
    print(f"\n=== Backend: {name} ===")
    baseline_gpu = gpu_used_gb() or 0.0

    t0 = time.perf_counter()
    backend = create_backend(name)
    try:
        backend.load()
    except Exception as e:
        print(f"  load failed: {e}")
        return BackendResult(name=name, load_seconds=0.0, error=str(e))
    load_seconds = time.perf_counter() - t0
    print(f"  loaded in {load_seconds:.1f}s")

    result = BackendResult(name=name, load_seconds=load_seconds)
    peak = baseline_gpu
    for i, clip in enumerate(clips, 1):
        t = time.perf_counter()
        try:
            text, _lang = backend.transcribe(clip.audio, TARGET_SAMPLE_RATE, label, code)
        except Exception as e:
            text = ""
            print(f"  clip {i} failed: {e}")
        latency = time.perf_counter() - t
        result.hyps.append(text)
        result.latencies.append(latency)
        used = gpu_used_gb()
        if used is not None:
            peak = max(peak, used)
        if i % 10 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)} clips ({latency:.2f}s last)")

    result.peak_gpu_gb = peak
    backend.cleanup()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def wer_for(reference: str, hypothesis: str) -> float:
    import jiwer
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)


def corpus_wer(refs: List[str], hyps: List[str]) -> float:
    import jiwer
    nrefs = [normalize_text(r) for r in refs]
    nhyps = [normalize_text(h) for h in hyps]
    # Drop pairs with an empty reference to avoid jiwer division issues.
    pairs = [(r, h) for r, h in zip(nrefs, nhyps) if r]
    if not pairs:
        return 0.0
    return jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_reports(clips, results, lang, out_dir: Path, stamp: str) -> tuple[Path, Path]:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"asr_bench_{stamp}.csv"
    md_path = out_dir / f"asr_bench_{stamp}.md"

    ran = [r for r in results if r.error is None]

    # ---- CSV: one row per clip ----
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["clip_id", "duration_s", "reference"]
        for r in ran:
            header += [f"hyp_{r.name}", f"wer_{r.name}", f"lat_{r.name}_s"]
        writer.writerow(header)
        for i, clip in enumerate(clips):
            row = [clip.clip_id, f"{clip.duration:.2f}", clip.reference]
            for r in ran:
                hyp = r.hyps[i] if i < len(r.hyps) else ""
                row += [hyp, f"{wer_for(clip.reference, hyp):.4f}",
                        f"{r.latencies[i]:.3f}" if i < len(r.latencies) else ""]
            writer.writerow(row)

    # ---- Summary stats ----
    refs = [c.reference for c in clips]
    durations = [c.duration for c in clips]
    total_audio = sum(durations)

    summary = []
    for r in ran:
        cw = corpus_wer(refs, r.hyps)
        lat_med = statistics.median(r.latencies) if r.latencies else 0.0
        lat_p90 = (statistics.quantiles(r.latencies, n=10)[8]
                   if len(r.latencies) >= 10 else max(r.latencies, default=0.0))
        total_proc = sum(r.latencies)
        rtf = (total_proc / total_audio) if total_audio else 0.0
        summary.append({
            "name": r.name, "wer": cw, "lat_med": lat_med, "lat_p90": lat_p90,
            "rtf": rtf, "vram": r.peak_gpu_gb, "load": r.load_seconds,
        })

    # ---- Markdown report ----
    lines = []
    lines.append(f"# Benchmark ASR — Qwen vs Canary ({lang})")
    lines.append("")
    lines.append(f"- Clips: **{len(clips)}** (FLEURS `{LANG_PRESETS[lang]['fleurs_config']}` test)")
    lines.append(f"- Audio total: **{total_audio:.1f} s**")
    lines.append(f"- Run: `{stamp}`")
    lines.append("")
    lines.append("| Moteur | WER ↓ | Latence médiane ↓ | Latence p90 | RTF ↓ | VRAM crête (GPU) | Chargement |")
    lines.append("|--------|-------|-------------------|-------------|-------|------------------|------------|")
    for s in summary:
        lines.append(
            f"| {s['name']} | {s['wer']*100:.1f}% | {s['lat_med']:.2f}s | "
            f"{s['lat_p90']:.2f}s | {s['rtf']:.3f} | {s['vram']:.2f} GB | {s['load']:.1f}s |"
        )
    lines.append("")

    # Errors, if any
    for r in results:
        if r.error:
            lines.append(f"> ⚠️ Backend `{r.name}` a échoué au chargement : {r.error}")
            lines.append("")

    # Verdict
    if len(summary) >= 2:
        best_wer = min(summary, key=lambda s: s["wer"])
        best_lat = min(summary, key=lambda s: s["lat_med"])
        lines.append("## Verdict")
        lines.append("")
        lines.append(f"- **Précision** : meilleur WER = `{best_wer['name']}` ({best_wer['wer']*100:.1f}%).")
        lines.append(f"- **Vitesse** : meilleure latence médiane = `{best_lat['name']}` ({best_lat['lat_med']:.2f}s).")
        lines.append("")
        lines.append("> Note VRAM : usage GPU global (englobe torch ET onnxruntime), "
                     "à interpréter comme un ordre de grandeur, pas une mesure isolée par moteur.")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Qwen vs Canary on FLEURS.")
    parser.add_argument("--num-clips", type=int, default=50)
    parser.add_argument("--backends", nargs="+", default=["qwen", "canary"],
                        choices=["qwen", "canary"])
    parser.add_argument("--lang", default="fr", choices=list(LANG_PRESETS.keys()))
    args = parser.parse_args(argv)

    bench_dir = Path(__file__).parent
    cache_dir = bench_dir / "data" / "fleurs"
    out_dir = bench_dir / "results"

    preset = LANG_PRESETS[args.lang]
    label, code = preset["label"], preset["code"]

    try:
        clips = load_fleurs_clips(args.lang, args.num_clips, cache_dir)
    except Exception as e:
        print(f"Failed to load FLEURS clips: {e}", file=sys.stderr)
        return 1
    if not clips:
        print("No clips loaded; aborting.", file=sys.stderr)
        return 1

    # Run backends one at a time so GPU memory readings stay meaningful.
    results = [run_backend(name, clips, label, code) for name in args.backends]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path, md_path = write_reports(clips, results, args.lang, out_dir, stamp)

    print("\n" + "=" * 60)
    report = md_path.read_text(encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        # Windows consoles default to cp1252 and choke on arrows/emoji; the
        # .md file keeps the full UTF-8 version regardless.
        enc = sys.stdout.encoding or "ascii"
        print(report.encode(enc, errors="replace").decode(enc))
    print("=" * 60)
    print(f"\nCSV : {csv_path}")
    print(f"MD  : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
