# Flototext

Windows voice recognition application with real-time transcription.

## Features

- **Push-to-talk with F2**: Hold F2 to record, release to transcribe
- **Switchable ASR engines**: NVIDIA Canary 1B v2 (default, via ONNX Runtime) or Qwen3-ASR-1.7B, swappable at runtime from the tray menu
- **Auto-paste**: Transcribed text is automatically pasted at cursor position
- **Custom Dictionary**: Define word corrections for technical terms, names, etc.
- **Auto-mute**: System audio is muted during recording to prevent interference
- **History**: All transcriptions are saved (7-day retention)
- **Visual feedback**: Color-coded system tray icon + Windows notifications

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with CUDA (RTX series recommended)
- ~4 GB VRAM available

## Installation

1. Clone the project:
```bash
git clone https://github.com/florentcollect/Flototext.git
cd Flototext
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For GPU support with PyTorch CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

5. Double-click **`install.bat`** (as administrator)
   - Configures auto-start with Windows
   - Registers the application in Windows Settings

To uninstall: Windows Settings → Apps → Flototext → Uninstall

## Usage

1. Launch manually (if needed):
   - Double-click **`start-flototext.bat`**
   - Or: `python -m flototext.main`

2. The icon appears in the system tray (near the clock)

3. **Recording**:
   - Press and **hold** F2 to start recording
   - Speak (French by default; Qwen also supports 50+ languages)
   - **Release** F2 to stop and transcribe

4. The transcribed text will be automatically pasted at your cursor position

## Tray Icon States

| Color | State |
|-------|-------|
| Orange | Loading model |
| Green | Ready |
| Red | Recording |
| Yellow | Processing transcription |
| Gray | Error |

## System Tray Menu

Right-click on the icon to access options:
- **Copy last transcription**: Copy the last transcription to clipboard
- **ASR model**: Switch between Canary (default) and Qwen3-ASR; the model reloads on the fly and the choice is saved
- **Edit dictionary**: Open the custom words dictionary file
- **Sounds**: Enable/disable audio feedback
- **Notifications**: Enable/disable Windows notifications
- **Mute during recording**: Enable/disable system audio muting while recording
- **Quit**: Close the application

## ASR Engines

Flototext ships two switchable speech-to-text backends behind a common interface
(`flototext/core/asr_backends.py`). Switch them live from the tray menu (**ASR model**);
the choice is persisted in `data/settings.json` under `model.backend`.

| Backend | Model | Library | Notes |
|---------|-------|---------|-------|
| **Canary** (default) | `nemo-canary-1b-v2` | `onnx-asr` (ONNX Runtime GPU) | Lighter & ~5x faster, best WER on FLEURS fr_fr (4.6%). Expects a short language code (`fr`). |
| Qwen | `Qwen/Qwen3-ASR-1.7B` | `qwen-asr` (transformers 4.57.6) | Multilingual (50+ languages). Expects a full language label (`French`). ~4 GB VRAM. |

Benchmark them yourself (FLEURS fr_fr — WER, latency, RTF):
```bash
python -m benchmarks.benchmark_asr --backends canary qwen
```

## Custom Dictionary

Create custom word corrections for terms that are frequently misrecognized (technical jargon, names, etc.).

The dictionary file is located at `data/custom_words.json`:

```json
{
  "corrections": {
    "clode": "Claude",
    "anthropique": "Anthropic",
    "pie torche": "PyTorch"
  }
}
```

- **Keys**: What the ASR model outputs (lowercase)
- **Values**: The correct spelling you want

Access via tray menu: **Edit dictionary** (opens the file in your default JSON editor)

## Database

Transcriptions are stored in `data/transcriptions.db` and kept for 7 days.

View history:
```bash
sqlite3 data/transcriptions.db "SELECT * FROM transcriptions ORDER BY created_at DESC LIMIT 10"
```

## Project Structure

```
Flototext/
├── flototext/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py               # Configuration
│   ├── core/
│   │   ├── hotkey_manager.py   # F2 key detection
│   │   ├── audio_recorder.py   # Microphone capture
│   │   ├── asr_backends.py     # Canary / Qwen ASR backends (switchable)
│   │   ├── transcriber.py      # Delegates to the selected backend
│   │   ├── text_inserter.py    # Clipboard paste
│   │   ├── text_corrector.py   # Custom word corrections
│   │   └── audio_muter.py      # System audio muting
│   ├── storage/
│   │   ├── database.py         # SQLite operations
│   │   └── models.py           # Data models
│   └── ui/
│       ├── tray_app.py         # System tray icon
│       ├── notifications.py    # Toast notifications
│       └── sounds.py           # Audio feedback
├── benchmarks/
│   └── benchmark_asr.py        # FLEURS fr_fr WER / latency / RTF comparison
├── data/
│   ├── transcriptions.db       # Database
│   ├── settings.json           # User settings (incl. model.backend)
│   └── custom_words.json       # Custom word dictionary
├── assets/
│   └── icon.ico
├── install.bat                 # Windows installer
├── uninstall.bat               # Uninstaller
├── start-flototext.bat                   # Manual launch
├── requirements.txt
└── README.md
```

## Configuration

Edit `flototext/config.py` to customize:
- `hotkey.trigger_key`: Trigger key (default: "f2")
- `audio.sample_rate`: Sample rate (default: 16000)
- `model.backend`: Active ASR backend, `"canary"` (default) or `"qwen"` (persisted in `data/settings.json`)
- `model.canary_model_name`: Canary model name (default: `nemo-canary-1b-v2`)
- `model.model_name`: Qwen model id (default: `Qwen/Qwen3-ASR-1.7B`)
- `ui.play_sounds`: Sounds enabled by default
- `ui.show_notifications`: Notifications enabled by default
- `ui.mute_during_recording`: Mute system audio while recording (default: True)

## Troubleshooting

### Model won't load
- Check CUDA is installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Make sure you have enough VRAM (~4 GB)

### No audio during recording
- Check that the default microphone is properly configured in Windows
- Test with another recording application

### Text doesn't paste
- Make sure a text field is active (blinking cursor)
- Text is also copied to clipboard (use Ctrl+V manually)

## Support

If Flototext saves you time, consider supporting its development:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/I2I61TF2LS)

## License

MIT License
