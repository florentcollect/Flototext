"""Benchmark the ASR backends on the user's own voice, mic and vocabulary.

FLEURS measures read news in a studio. Flototext dictates RPG content into a
Shure MV7 behind a MOTIV denoiser, so the decision between engines has to be
checked on that audio too. The app only stores the text of past dictations,
not the sound, so the past dictations become the script: the user reads them
back, and each take is kept with its sentence as the reference.

Three steps, run inside the .venv:

    python -m benchmarks.own_voice prepare          # pick sentences from the app DB
    (edit benchmarks/data/own_voice/phrases.txt to fix any transcription error)
    python -m benchmarks.own_voice record           # read each sentence into the mic
    python -m benchmarks.own_voice bench            # canary vs parakeet on those takes

Everything lands in benchmarks/data/own_voice/ and benchmarks/results/, both
gitignored: the recordings and the sentences are personal and stay local.

The references come from past dictations, which were already corrected by
data/custom_words.json. The report therefore gives two WERs per engine: raw
engine output, and after the dictionary, which is what actually gets typed.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from flototext.config import config

OWN_VOICE_DIR = Path(__file__).parent / "data" / "own_voice"
PHRASES_FILE = OWN_VOICE_DIR / "phrases.txt"
MANIFEST_FILE = OWN_VOICE_DIR / "manifest.json"
RESULTS_DIR = Path(__file__).parent / "results"
DB_PATH = config.data_dir / "transcriptions.db"

LANGUAGE_LABEL = "French"
LANGUAGE_CODE = "fr"


# --------------------------------------------------------------------------- #
# prepare: pick sentences from past dictations
# --------------------------------------------------------------------------- #
def select_phrases(texts: List[str], count: int, min_words: int, max_words: int) -> List[str]:
    """Keep readable, distinct dictations and spread the pick across the list.

    Dictations are taken in the order given (oldest first). Very short ones say
    little about accuracy, very long ones are tiring to read in one breath, and
    duplicates would count the same sentence twice. Taking every k-th survivor
    instead of the first ``count`` keeps several days of vocabulary.
    """
    seen = set()
    kept = []
    for text in texts:
        clean = " ".join((text or "").split())
        words = len(clean.split())
        if not (min_words <= words <= max_words):
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(clean)

    if len(kept) <= count:
        return kept
    step = len(kept) / count
    return [kept[int(i * step)] for i in range(count)]


def cmd_prepare(args) -> int:
    if PHRASES_FILE.exists() and not args.force:
        print(f"{PHRASES_FILE} existe déjà (tes corrections y sont peut-être). "
              f"Relance avec --force pour l'écraser.")
        return 1
    if not DB_PATH.exists():
        print(f"Base introuvable : {DB_PATH}")
        return 1

    # Read-only: the app may be running and writing to it.
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        texts = [row[0] for row in conn.execute(
            "SELECT text FROM transcriptions ORDER BY created_at")]
    finally:
        conn.close()

    phrases = select_phrases(texts, args.count, args.min_words, args.max_words)
    if not phrases:
        print("Aucune dictée ne correspond aux critères de longueur.")
        return 1

    OWN_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    PHRASES_FILE.write_text("\n".join(phrases) + "\n", encoding="utf-8")
    print(f"{len(phrases)} phrases (sur {len(texts)} dictées) écrites dans :\n  {PHRASES_FILE}")
    print("Relis-les et corrige les erreurs de transcription AVANT d'enregistrer :")
    print("chaque ligne sert de référence exacte. Supprime celles que tu ne veux pas lire.")
    return 0


# --------------------------------------------------------------------------- #
# record: read each sentence into the pinned microphone
# --------------------------------------------------------------------------- #
def load_phrases() -> List[str]:
    if not PHRASES_FILE.exists():
        return []
    lines = PHRASES_FILE.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_manifest() -> list:
    if not MANIFEST_FILE.exists():
        return []
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def save_manifest(entries: list) -> None:
    MANIFEST_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_record(args) -> int:
    import soundfile as sf
    from flototext.core.audio_recorder import AudioRecorder, is_silent, peak_window_rms

    phrases = load_phrases()
    if not phrases:
        print(f"Pas de phrases : lance d'abord `python -m benchmarks.own_voice prepare`.")
        return 1

    manifest = load_manifest()
    done = {entry["reference"] for entry in manifest}
    todo = [p for p in phrases if p not in done]
    if not todo:
        print(f"Les {len(phrases)} phrases sont déjà enregistrées. Lance `bench`.")
        return 0

    recorder = AudioRecorder()
    threshold = config.audio.silence_rms_threshold
    print(f"Micro épinglé : {config.audio.input_device or '(défaut système)'}")
    print("N'appuie pas sur F2 pendant la session. Pour chaque phrase :")
    print("  Entrée = démarrer, Entrée = arrêter | s = passer | q = quitter (rien n'est perdu)\n")

    for i, reference in enumerate(todo, 1):
        while True:
            print(f"[{len(done) + i}/{len(phrases)}] {reference}")
            choice = input("  Entrée pour démarrer > ").strip().lower()
            if choice == "q":
                print(f"Arrêt. {len(manifest)} prises enregistrées, la suite reprendra ici.")
                return 0
            if choice == "s":
                break

            if not recorder.start_recording():
                print("  Impossible d'ouvrir le micro.")
                return 1
            input("  ● enregistrement... Entrée pour arrêter > ")
            result = recorder.stop_recording()
            if result is None or result.audio_data.size == 0:
                print("  Rien capté, on recommence.")
                continue

            audio = result.audio_data.astype(np.float32)
            level = peak_window_rms(audio, recorder.sample_rate)
            device = recorder.describe_input_device()
            if is_silent(audio, threshold):
                print(f"  Silence (pic {level:.4f} < {threshold}) sur « {device} ». On recommence.")
                continue

            print(f"  {result.duration:.1f} s, pic {level:.4f}, micro « {device} »")
            again = input("  Garder ? (Entrée = oui, r = refaire) > ").strip().lower()
            if again == "r":
                continue

            name = f"take_{len(manifest) + 1:03d}.wav"
            sf.write(OWN_VOICE_DIR / name, audio, recorder.sample_rate, subtype="PCM_16")
            manifest.append({"file": name, "reference": reference,
                             "duration": round(result.duration, 2), "device": device})
            save_manifest(manifest)
            break

    recorder.cleanup()
    print(f"\nTerminé : {len(manifest)} prises. Lance `python -m benchmarks.own_voice bench`.")
    return 0


# --------------------------------------------------------------------------- #
# bench: run the engines on the takes
# --------------------------------------------------------------------------- #
def load_clips(manifest: list, directory: Path):
    """Turn recorded takes into benchmark clips, normalized the way the app does."""
    import soundfile as sf
    from benchmarks.benchmark_asr import Clip, _normalize_audio, _resample

    clips = []
    for entry in manifest:
        path = directory / entry["file"]
        if not path.exists():
            print(f"  prise manquante, ignorée : {path.name}")
            continue
        waveform, sr = sf.read(path, dtype="float32")
        waveform = _normalize_audio(_resample(waveform, sr))
        clips.append(Clip(clip_id=entry["file"], audio=waveform,
                          duration=len(waveform) / 16000, reference=entry["reference"]))
    return clips


def cmd_bench(args) -> int:
    import csv
    import statistics
    from benchmarks.benchmark_asr import corpus_wer, run_backend, wer_for
    from flototext.core.text_corrector import TextCorrector

    clips = load_clips(load_manifest(), OWN_VOICE_DIR)
    if not clips:
        print("Aucune prise : lance d'abord `record`.")
        return 1

    corrector = TextCorrector()
    results = [run_backend(name, clips, LANGUAGE_LABEL, LANGUAGE_CODE) for name in args.backends]
    ran = [r for r in results if r.error is None]
    refs = [c.reference for c in clips]
    corrected = {r.name: [corrector.correct(h) for h in r.hyps] for r in ran}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"own_voice_{stamp}.csv"
    md_path = RESULTS_DIR / f"own_voice_{stamp}.md"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["take", "duration_s", "reference"]
        for r in ran:
            header += [f"brut_{r.name}", f"wer_brut_{r.name}",
                       f"dico_{r.name}", f"wer_dico_{r.name}", f"lat_{r.name}_s"]
        writer.writerow(header)
        for i, clip in enumerate(clips):
            row = [clip.clip_id, f"{clip.duration:.2f}", clip.reference]
            for r in ran:
                raw, fixed = r.hyps[i], corrected[r.name][i]
                row += [raw, f"{wer_for(clip.reference, raw):.4f}",
                        fixed, f"{wer_for(clip.reference, fixed):.4f}", f"{r.latencies[i]:.3f}"]
            writer.writerow(row)

    total_audio = sum(c.duration for c in clips)
    lines = [
        "# Benchmark ASR — ta voix (MV7)",
        "",
        f"- Prises : **{len(clips)}**, audio total **{total_audio:.1f} s**",
        f"- Run : `{stamp}`",
        "- WER brut = sortie du moteur ; WER dico = après `custom_words.json` (ce qui est tapé).",
        "",
        "| Moteur | WER brut ↓ | WER dico ↓ | Latence médiane ↓ | Chargement |",
        "|--------|-----------|-----------|-------------------|------------|",
    ]
    for r in ran:
        lines.append(
            f"| {r.name} | {corpus_wer(refs, r.hyps) * 100:.1f}% | "
            f"{corpus_wer(refs, corrected[r.name]) * 100:.1f}% | "
            f"{statistics.median(r.latencies):.2f}s | {r.load_seconds:.1f}s |")
    for r in results:
        if r.error:
            lines += ["", f"> Backend `{r.name}` a échoué au chargement : {r.error}"]
    lines += ["", "> Les références viennent de dictées déjà passées par le dictionnaire,",
              "> écrit d'après les erreurs de Canary : le WER dico avantage Canary."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nCSV : {csv_path}\nMD  : {md_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bench ASR sur ta propre voix.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="choisir des phrases dans les dictées passées")
    p.add_argument("--count", type=int, default=25)
    p.add_argument("--min-words", type=int, default=6)
    p.add_argument("--max-words", type=int, default=40)
    p.add_argument("--force", action="store_true", help="écraser phrases.txt")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("record", help="lire les phrases au micro")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("bench", help="comparer les moteurs sur les prises")
    p.add_argument("--backends", nargs="+", default=["canary", "parakeet"],
                   choices=["qwen", "canary", "parakeet"])
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
