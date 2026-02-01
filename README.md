# Flototext

Application de reconnaissance vocale Windows avec transcription en temps réel.

## Fonctionnalités

- **Activation par F2** : Appuyez et maintenez F2 pour enregistrer, relâchez pour transcrire
- **Transcription IA** : Utilise Qwen3-ASR-1.7B (support français et 52 langues)
- **Collage automatique** : Le texte transcrit est automatiquement collé à la position du curseur
- **Historique** : Toutes les transcriptions sont sauvegardées dans une base SQLite
- **Feedback visuel et audio** : Icône système colorée + sons de confirmation

## Prérequis

- Windows 10/11
- Python 3.10+
- GPU NVIDIA avec CUDA (RTX série recommandée)
- ~4 GB VRAM disponible

## Installation

1. Cloner ou télécharger le projet :
```bash
cd F:\Flototext
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

## Utilisation

1. Lancer l'application :
```bash
python -m flototext.main
```

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
- **Sounds** : Activer/désactiver les sons de feedback
- **Notifications** : Activer/désactiver les notifications Windows
- **Quit** : Fermer l'application

## Base de données

Les transcriptions sont stockées dans `data/transcriptions.db`.

Consulter l'historique :
```bash
sqlite3 data/transcriptions.db "SELECT * FROM transcriptions ORDER BY created_at DESC LIMIT 10"
```

## Structure du projet

```
F:\Flototext\
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
