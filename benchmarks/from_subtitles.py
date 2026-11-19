"""Turn a video's audio + its subtitles into takes for benchmarks.own_voice.

A published video with reviewed subtitles is ready-made test material: the
speaker's real voice, real mic, real vocabulary, and a reference text. This
cuts the audio at subtitle boundaries into segments of at most
MAX_SEGMENT_SECONDS (Canary's chunk window, so no engine gets split mid-word)
and writes them in the own_voice layout: take_NNN.wav + manifest.json.

    python -m benchmarks.from_subtitles AUDIO SUBTITLES.srt OUT_DIR
    python -m benchmarks.own_voice bench --dir OUT_DIR

AUDIO can be any format ffmpeg reads (webm, m4a, mp4, wav...).
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

SAMPLE_RATE = 16000
MAX_SEGMENT_SECONDS = 20.0
# Cues further apart than this are a pause: a natural place to cut.
PAUSE_SECONDS = 0.3
# Margin kept around each segment so the first and last syllables survive.
PAD_SECONDS = 0.15

_TIME_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(content: str) -> List[Cue]:
    """Parse SRT text into cues; tags like <i> are dropped."""
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", content.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        for i, line in enumerate(lines):
            match = _TIME_RE.search(line)
            if not match:
                continue
            g = match.groups()
            text = " ".join(lines[i + 1:])
            text = re.sub(r"<[^>]+>", "", text).strip()
            if text:
                cues.append(Cue(_seconds(*g[:4]), _seconds(*g[4:]), text))
            break
    return cues


def group_cues(cues: List[Cue], max_seconds: float = MAX_SEGMENT_SECONDS,
               pause: float = PAUSE_SECONDS) -> List[List[Cue]]:
    """Group consecutive cues into segments no longer than ``max_seconds``.

    A segment closes at the last pause before it would overflow, so cuts fall
    between sentences rather than across a word. A single cue longer than the
    limit stays alone: engines chunk it themselves.
    """
    segments: List[List[Cue]] = []
    current: List[Cue] = []
    for cue in cues:
        if current and cue.end - current[0].start > max_seconds:
            # Close at the last pause inside the current group, if any.
            split = len(current)
            for j in range(len(current) - 1, 0, -1):
                if current[j].start - current[j - 1].end >= pause:
                    split = j
                    break
            segments.append(current[:split])
            current = current[split:]
            if current and cue.end - current[0].start > max_seconds:
                segments.append(current)
                current = []
        current.append(cue)
    if current:
        segments.append(current)
    return segments


def load_audio(path: Path) -> np.ndarray:
    """Decode any ffmpeg-readable file to 16 kHz mono float32."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav)],
            check=True,
        )
        audio, _ = sf.read(wav, dtype="float32")
    return audio


def build_takes(audio: np.ndarray, segments: List[List[Cue]], out_dir: Path) -> list:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for group in segments:
        start = max(0.0, group[0].start - PAD_SECONDS)
        end = min(len(audio) / SAMPLE_RATE, group[-1].end + PAD_SECONDS)
        chunk = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        if chunk.size == 0:
            continue
        name = f"take_{len(manifest) + 1:03d}.wav"
        sf.write(out_dir / name, chunk, SAMPLE_RATE, subtype="PCM_16")
        manifest.append({
            "file": name,
            "reference": " ".join(c.text for c in group),
            "duration": round(len(chunk) / SAMPLE_RATE, 2),
            "start": round(start, 2),
            "source": "subtitles",
        })
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Découpe un audio selon ses sous-titres.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("subtitles", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)

    cues = parse_srt(args.subtitles.read_text(encoding="utf-8-sig"))
    if not cues:
        print("Aucun sous-titre lisible.")
        return 1
    segments = group_cues(cues)
    manifest = build_takes(load_audio(args.audio), segments, args.out_dir)
    total = sum(entry["duration"] for entry in manifest)
    print(f"{len(cues)} sous-titres -> {len(manifest)} segments, {total:.0f} s d'audio, dans {args.out_dir}")
    print(f"Ensuite : python -m benchmarks.own_voice bench --dir {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
