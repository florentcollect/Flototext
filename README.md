# Flototext

Application de reconnaissance vocale Windows avec transcription en temps réel.

## Fonctionnalités

- **Activation par F2** : Appuyez et maintenez F2 pour enregistrer, relâchez pour transcrire
- **Transcription IA** : Utilise Qwen3-ASR-1.7B (support français et 52 langues)
- **Collage automatique** : Le texte transcrit est automatiquement collé à la position du curseur
- **Historique** : Toutes les transcriptions sont sauvegardées (7 jours)
- **Feedback visuel** : Icône système colorée + notifications Windows

## Prérequis

- Windows 10/11
- Python 3.10+
- GPU NVIDIA avec CUDA (RTX série recommandée)
- ~4 GB VRAM disponible

## Installation

1. Cloner le projet :
```bash
git clone https://github.com/florentcollect/Flototext.git
cd Flototext
```

2. Créer un environnement virtuel (recommandé) :
```bash
python -m venv venv
venv\Scripts\activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Pour le support GPU avec PyTorch CUDA :
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

5. Double-cliquer sur **`install.bat`** (en administrateur)
   - Configure le démarrage automatique avec Windows
   - Enregistre l'application dans les paramètres Windows

Pour désinstaller : Paramètres Windows → Applications → Flototext → Désinstaller

## Utilisation

1. Lancer manuellement (si besoin) :
   - Double-cliquer sur **`start.bat`**
   - Ou : `python -m flototext.main`

2. L'icône apparaît dans la barre système (près de l'horloge)

3. **Enregistrement** :
   - Appuyez et **maintenez** F2 pour commencer l'enregistrement
   - Parlez en français
   - **Relâchez** F2 pour arrêter et transcrire

4. Le texte transcrit sera automatiquement collé là où se trouve votre curseur

## États de l'icône

| Couleur | État |
|---------|------|
| Orange | Chargement du modèle |
| Vert | Prêt |
| Rouge | Enregistrement en cours |
| Jaune | Traitement de la transcription |
| Gris | Erreur |

## Menu de l'icône système

Clic droit sur l'icône pour accéder aux options :
- **Copier dernière transcription** : Récupérer la dernière transcription dans le presse-papiers
- **Sons** : Activer/désactiver les sons de feedback
- **Notifications** : Activer/désactiver les notifications Windows
- **Quitter** : Fermer l'application

## Base de données

Les transcriptions sont stockées dans `data/transcriptions.db` et conservées pendant 7 jours.

Consulter l'historique :
```bash
sqlite3 data/transcriptions.db "SELECT * FROM transcriptions ORDER BY created_at DESC LIMIT 10"
```

## Structure du projet

```
Flototext/
├── flototext/
│   ├── __init__.py
│   ├── main.py                 # Point d'entrée
│   ├── config.py               # Configuration
│   ├── core/
│   │   ├── hotkey_manager.py   # Détection F2
│   │   ├── audio_recorder.py   # Capture microphone
│   │   ├── transcriber.py      # Modèle Qwen3-ASR
│   │   └── text_inserter.py    # Collage clipboard
│   ├── storage/
│   │   ├── database.py         # Opérations SQLite
│   │   └── models.py           # Modèles de données
│   └── ui/
│       ├── tray_app.py         # Icône système
│       ├── notifications.py    # Notifications toast
│       └── sounds.py           # Feedback audio
├── data/
│   └── transcriptions.db       # Base de données
├── assets/
│   └── icon.ico
├── install.bat                 # Installation Windows
├── uninstall.bat               # Désinstallation
├── start.bat                   # Lancement manuel
├── requirements.txt
└── README.md
```

## Configuration

Modifier `flototext/config.py` pour personnaliser :
- `hotkey.trigger_key` : Touche de déclenchement (défaut: "f2")
- `audio.sample_rate` : Taux d'échantillonnage (défaut: 16000)
- `model.model_name` : Modèle ASR à utiliser
- `ui.play_sounds` : Sons activés par défaut
- `ui.show_notifications` : Notifications activées par défaut

## Dépannage

### Le modèle ne se charge pas
- Vérifiez que CUDA est installé : `python -c "import torch; print(torch.cuda.is_available())"`
- Assurez-vous d'avoir assez de VRAM (~4 GB)

### Pas de son lors de l'enregistrement
- Vérifiez que le microphone par défaut est correctement configuré dans Windows
- Testez avec un autre logiciel d'enregistrement

### Le texte ne se colle pas
- Assurez-vous qu'un champ de texte est actif (curseur clignotant)
- Le texte est aussi copié dans le presse-papiers (Ctrl+V manuellement)

## Licence

MIT License
