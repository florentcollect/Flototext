"""Pont de transcription Hermes -> Flototext.

Surveille le dossier d'echange partage avec le conteneur Hermes, transcrit
chaque fichier audio depose avec le moteur ASR de Flototext, applique le
dictionnaire de correction, et ecrit le texte resultant.

    F:\\Projets-IA\\Hermes\\hermes-data\\stt-bridge\\in\\    audio depose par Hermes
    F:\\Projets-IA\\Hermes\\hermes-data\\stt-bridge\\out\\   texte rendu a Hermes

Cote conteneur, ces dossiers sont /opt/data/stt-bridge/ : c'est un bind mount,
donc les deux mondes lisent les memes fichiers sans reseau ni port ouvert.

Lancement :
    F:\\Flototext\\.venv\\Scripts\\python.exe F:\\Flototext\\hermes_bridge.py

Le modele est charge une fois au demarrage et reste en memoire : la
transcription d'un vocal WhatsApp prend alors quelques secondes, pas le temps
d'un chargement complet.

Note : ce processus charge sa PROPRE instance du modele ASR. Si l'application
Flototext tourne en parallele, comptez deux fois la VRAM.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

BRIDGE = Path(r"F:\Projets-IA\Hermes\hermes-data\stt-bridge")
IN_DIR = BRIDGE / "in"
OUT_DIR = BRIDGE / "out"
LOG_FILE = BRIDGE / "watcher.log"

AUDIO_SUFFIXES = {".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".flac", ".webm", ".aac"}
POLL_SECONDS = 0.5
TARGET_SAMPLE_RATE = 16000

# Flototext est importable depuis la racine du projet.
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("hermes-bridge")


def charger_audio(chemin: Path):
    """Decode n'importe quel format vers du mono 16 kHz float32.

    librosa (present dans le venv Flototext) s'appuie sur soundfile puis
    audioread ; les .ogg/.opus de WhatsApp passent par le second.
    """
    import librosa

    audio, sr = librosa.load(str(chemin), sr=TARGET_SAMPLE_RATE, mono=True)
    return audio, sr


def init_moteur():
    """Charge le moteur ASR et le correcteur, exactement comme l'app."""
    from flototext.core.transcriber import Transcriber
    from flototext.core.text_corrector import TextCorrector

    log.info("Chargement du moteur ASR Flototext...")
    transcriber = Transcriber()

    # load_model_async() rend la main immediatement ; on attend que le modele
    # soit reellement pret avant de traiter quoi que ce soit.
    transcriber.load_model_async()
    for _ in range(600):  # 5 minutes maximum
        if getattr(transcriber, "_model_loaded", False):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("le modele ASR n'a pas fini de charger apres 5 minutes")

    log.info("Moteur ASR pret.")

    corrector = TextCorrector()
    log.info("Dictionnaire de correction charge.")
    return transcriber, corrector


DICO = Path(__file__).resolve().parent / "data" / "custom_words.json"
_dico_mtime = 0.0


def recharger_dictionnaire_si_modifie(corrector) -> None:
    """Relit le dictionnaire quand le fichier a change.

    Sans ca, toute correction ajoutee apres le lancement du pont resterait
    ignoree jusqu'au prochain redemarrage — le piege classique quand on
    enrichit le dictionnaire au fil des transcriptions ratees.
    """
    global _dico_mtime
    try:
        mtime = DICO.stat().st_mtime
    except OSError:
        return
    if mtime <= _dico_mtime:
        return
    if _dico_mtime:  # pas au tout premier passage
        try:
            corrector.reload()
            log.info("Dictionnaire recharge (%s corrections)", len(getattr(corrector, "corrections", {}) or {}))
        except Exception:  # noqa: BLE001
            log.exception("Echec du rechargement du dictionnaire")
    _dico_mtime = mtime


def traiter(chemin: Path, transcriber, corrector) -> None:
    recharger_dictionnaire_si_modifie(corrector)
    job = chemin.stem
    sortie = OUT_DIR / f"{job}.txt"
    erreur = OUT_DIR / f"{job}.err"
    debut = time.monotonic()

    try:
        audio, sr = charger_audio(chemin)
        duree = len(audio) / sr if sr else 0.0
        log.info("%s : %.1f s d'audio", job, duree)

        resultat = transcriber.transcribe(audio, sr)
        if not getattr(resultat, "success", False):
            raise RuntimeError(getattr(resultat, "error", None) or "transcription echouee")

        texte = (resultat.text or "").strip()
        if texte:
            texte = corrector.correct(texte)

        # Ecriture atomique : Hermes attend l'apparition du .txt et le lit
        # aussitot, il ne doit jamais tomber sur un fichier incomplet.
        tmp = sortie.with_suffix(".txt.part")
        tmp.write_text(texte, encoding="utf-8")
        tmp.replace(sortie)

        log.info("%s : OK en %.1f s -> %r", job, time.monotonic() - debut, texte[:80])

    except Exception as exc:  # noqa: BLE001 - on rapporte tout a Hermes
        log.exception("%s : echec", job)
        tmp = erreur.with_suffix(".err.part")
        tmp.write_text(str(exc), encoding="utf-8")
        tmp.replace(erreur)

    finally:
        try:
            chemin.unlink()
        except OSError:
            log.warning("%s : impossible de supprimer l'audio traite", job)


def main() -> int:
    for d in (IN_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    try:
        transcriber, corrector = init_moteur()
    except Exception:
        log.exception("Impossible d'initialiser le moteur ASR")
        return 1

    log.info("En veille sur %s", IN_DIR)

    try:
        while True:
            fichiers = sorted(
                p for p in IN_DIR.iterdir()
                if p.is_file()
                and p.suffix.lower() in AUDIO_SUFFIXES
                and not p.name.startswith(".")
            )
            for fichier in fichiers:
                traiter(fichier, transcriber, corrector)
            if not fichiers:
                time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        log.info("Arret demande.")
        return 0
    finally:
        try:
            transcriber.cleanup()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
