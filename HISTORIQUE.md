# Historique et etat fonctionnel de Flototext

Derniere mise a jour: 2026-06-27

## Etat actuel

Flototext est une application Windows de dictee push-to-talk. L'utilisateur maintient `F2`, parle, relache la touche, puis le texte transcrit est corrige, sauvegarde et insere automatiquement a l'emplacement du curseur.

## Fonctionnalites disponibles

- Enregistrement audio au micro avec declenchement par touche globale `F2`.
- Transcription locale multi-backend commutable a chaud (menu tray « Modele ASR »).
  - Moteur par defaut : **NVIDIA Canary 1B v2** (`nemo-canary-1b-v2` via `onnx-asr` / ONNX Runtime GPU).
  - Alternative disponible : `Qwen/Qwen3-ASR-1.7B` (via `qwen-asr`), conservee comme option mais plus active par defaut.
  - Choix du backend persiste dans `data/settings.json` (section `model.backend`).
- Chargement du modele ASR en arriere-plan.
- Benchmark ASR comparatif (FLEURS fr_fr : WER, latence, RTF) via `python -m benchmarks.benchmark_asr`.
- Mode dry-run sans GPU ni modele avec `FLOTOTEXT_DRY_RUN=1`.
- Insertion automatique du texte par presse-papiers et collage.
- Repli vers copie presse-papiers si le collage echoue.
- Historique SQLite des transcriptions dans `data/transcriptions.db`.
- Retention automatique des transcriptions sur 7 jours.
- Menu system tray avec etats visuels: chargement, pret, enregistrement, traitement, erreur.
- Notifications Windows configurables.
- Sons configurables.
- Mute automatique du son systeme pendant l'enregistrement, configurable.
- Restauration du son systeme meme si l'option de mute est desactivee pendant un mute actif.
- Copie de la derniere transcription depuis le menu system tray.
- Dictionnaire personnalise dans `data/custom_words.json`.
- Editeur visuel Tkinter pour ajouter, modifier et supprimer les corrections du dictionnaire.
- Localisation via fichiers JSON dans `data/locales/`.
- Changement de langue depuis le menu system tray.
- Persistance des reglages utilisateur dans `data/settings.json`: langue, sons, notifications, mute pendant enregistrement.
- Normalisation de nombres francais dictes en toutes lettres.
- Tests unitaires couvrant les composants principaux.

## Corrections et ameliorations recentes

### 2026-06-14

Commit pousse : `f7bfeaa feat: enrichit le dictionnaire ASR (JDR, mecarun, cool)`

- Enrichissement de `data/custom_words.json` avec des corrections orientees vocabulaire JDR
  (variantes de « JDR », « mecarun »/« Mécarun », « rôliste »/« rôlistes », « cool », « midjourney »…).
- Prise en compte des transcriptions phonetiques erronees frequentes (ex. « mais qu'arronne ? » -> `mecarun`).
- `.gitignore` : exclusion de `wiki_temp/` et des fichiers `*.code-workspace`.

### 2026-06-08

Commit pousse : `675ab50 feat: backend ASR commutable Canary + benchmark FLEURS`

- Abstraction `BaseASRBackend` (`flototext/core/asr_backends.py`) avec deux implementations :
  `QwenBackend` et `CanaryOnnxBackend`.
- `Transcriber` delegue desormais au backend choisi et bascule a chaud via `reload_backend()`.
- Menu tray « Modele ASR » pour permuter Qwen / Canary ; choix persiste dans `data/settings.json`.
- Canary 1B v2 charge via `onnx-asr` ; provider CUDA force et cablage des DLL CUDA de torch
  (`_ensure_cuda_dlls`), sinon ONNX Runtime retombe silencieusement sur CPU.
- Chunking audio interne (Canary limite a ~25 s par appel).
- Script de benchmark `benchmarks/benchmark_asr.py` (FLEURS fr_fr) : WER, latence, RTF.
  Resultat : Canary 4,6 % WER vs Qwen 4,9 %, environ 5x plus rapide.
- **F2 passe sur Canary par defaut** (`settings.json` -> `model.backend = "canary"`). Qwen reste accessible
  via le menu tray mais n'est plus le moteur actif.
- Tests ajoutes : interface backend, chunking, persistance de la config (en dry-run).

Verification :

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

### 2026-04-30

Commit pousse : `a25ff47 fix: prevent dictionary editor crash on second open`

- Correction du crash de l'editeur de dictionnaire a la deuxieme ouverture.
- Cause : Tkinter n'est pas thread-safe ; creer un nouveau `tk.Tk()` dans un thread daemon a
  chaque ouverture corrompait l'etat Tcl/Tk.
- Solution : un unique thread GUI persistant possede une seule racine Tk cachee ; l'editeur est
  reconstruit comme `Toplevel` planifie via `root.after()`, pour que tous les appels Tk se fassent
  sur le meme thread.

### 2026-04-24

Commit pousse: `0549d56 fix: preserve standalone un and une`

- Correction de la normalisation de `un` et `une`.
- `un` et `une` seuls restent en lettres pour preserver le texte courant.
- Exemple conserve: `un ou une option`.
- Les nombres composes restent convertis.
- Exemple converti: `vingt et un jours` -> `21 jours`.
- Ajout de tests de regression dans `tests/test_number_normalizer.py`.

Verification:

```powershell
python -m unittest tests.test_number_normalizer
python -m unittest discover -s tests
```

Resultat: 16 tests OK.

### 2026-04-23

Commit pousse: `b05ba8a fix: normalize dictated numbers and add dry-run tests`

- Ajout de `flototext/core/number_normalizer.py`.
- Integration de la normalisation des nombres dans `TextCorrector.correct()`.
- Conversions prises en charge pour les cas courants:
  - `deux-cent` -> `200`
  - `deux cents` -> `200`
  - `vingt et un` -> `21`
  - `quatre-vingt-dix-neuf` -> `99`
  - `deux cent cinquante trois euros` -> `253 euros`
- Ajout du mode dry-run via `FLOTOTEXT_DRY_RUN=1`.
- Correction de la gestion des erreurs de transcription pour eviter une double notification.
- Correction du mute audio: `unmute()` peut restaurer l'audio meme si l'option est desactivee entre-temps.
- Correction des remplacements du dictionnaire finissant par ponctuation, par exemple `gitpo.`.
- Ajout de tests pour audio muter, audio recorder, database, normalisation de nombres, correcteur de texte et transcripteur.

Verification:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q flototext
```

Resultat: tests OK au moment du commit.

### Changements anterieurs visibles dans l'historique Git

- `2b0d993 Add visual dictionary editor with tkinter GUI`
  - Ajout de l'editeur graphique du dictionnaire personnalise.
- `782b8eb Persist user settings (language, sounds, notifications, mute) across sessions`
  - Ajout de la sauvegarde et du rechargement des reglages utilisateur.
- `024ecb6 Add MIT License`
  - Ajout de la licence MIT.

## Etat des tests

La commande de reference actuelle est (a lancer depuis le venv du projet) :

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

Dernier resultat connu: 25 tests passes.

Note: `python -m unittest discover` sans `-s tests` ne decouvre pas les tests dans cette configuration.
Note: les tests des backends ASR tournent en dry-run et ne chargent pas les vrais modeles (pytest absent du venv, utiliser `unittest`).

## Limites connues

- La normalisation des nombres francais couvre les formes courantes, pas toute la grammaire francaise.
- Les decimaux, numeros de telephone, codes postaux, dates et heures ne sont pas encore normalises explicitement.
- Les modeles ASR demandent CUDA ; Canary (defaut) est plus leger et rapide que Qwen, qui requiert environ 4 Go de VRAM.
- Le backend Canary (ONNX Runtime) retombe silencieusement sur CPU si les DLL CUDA de torch ne sont pas cablees (gere par `_ensure_cuda_dlls`).
- Le mode dry-run sert aux tests et au demarrage sans modele, pas a une vraie transcription.
- L'application cible Windows; les integrations tray, hotkey, presse-papiers et audio sont pensees pour cet environnement.

## Fichiers locaux hors historique Git

Au moment de cette note, ces fichiers existent localement mais n'ont pas ete inclus dans les commits precedents:

- `data/custom_words.json`
- `data/settings.json`
- `flototext/Flototext.code-workspace`
- `wiki_temp/`

