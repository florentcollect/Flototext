"""Configuration for Flototext application."""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    """Audio recording configuration."""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    min_duration: float = 0.5  # Minimum recording duration in seconds
    max_duration: float = 300.0  # Maximum recording duration (5 minutes)
    # Substring of the input device name to pin (e.g. "Shure MV7"). None follows
    # the Windows default input, which other apps are free to steal.
    input_device: Optional[str] = None
    # Below this RMS the capture holds no speech. Feeding silence to the ASR
    # yields confident garbage ("<unk>", stray digits) that passes for a working
    # transcription, so refuse it and say why instead. Sits under a measured
    # idle noise floor (~0.002 RMS on a Shure MV7+) to never reject a quiet voice;
    # the fault it catches is a dead stream, which reads as exactly 0.0.
    silence_rms_threshold: float = 0.0015


@dataclass
class ModelConfig:
    """ASR model configuration."""
    backend: str = "qwen"  # Active ASR engine: "qwen" or "canary"
    model_name: str = "Qwen/Qwen3-ASR-1.7B"  # Qwen backend model id
    canary_model_name: str = "nemo-canary-1b-v2"  # Canary backend (onnx-asr) model name
    device: str = "cuda:0"
    dtype: str = "bfloat16"  # Optimal for RTX 4090
    max_new_tokens: int = 512
    # Release the model after this long without a transcription, handing its
    # VRAM back to the system. The ONNX arena only grows while the model is
    # loaded, so unloading between sessions is the only thing that truly
    # reclaims it. The next hotkey press reloads (~15 s, overlapped with the
    # recording itself). 0 disables the idle unload.
    idle_unload_seconds: int = 900  # 15 minutes
    # VRAM budget in GB for this process. The ONNX arena grows with use and a
    # per-session gpu_mem_limit cannot cap the process (Canary opens two
    # sessions, and the encoder's 3.06 GB of weights set the floor). So instead
    # of capping, we watch: once the process is over budget, the model is
    # released at the first lull rather than waiting for the full idle delay.
    # A shared env allocator would give a hard cap, but it keeps its memory when
    # sessions are dropped - measured - which would defeat the idle unload.
    # 0 disables the budget check.
    vram_budget_gb: float = 5.0
    # How long without a transcription counts as a lull, when over budget.
    over_budget_idle_seconds: int = 120
    dry_run: bool = field(default_factory=lambda: os.getenv("FLOTOTEXT_DRY_RUN", "").lower() in {"1", "true", "yes", "on"})
    dry_run_text: str = "test dry-run deux-cent euros"


@dataclass
class HotkeyConfig:
    """Hotkey configuration."""
    trigger_key: str = "f2"


@dataclass
class UIConfig:
    """UI configuration."""
    app_name: str = "Flototext"
    language: str = "en"  # Language code (en, fr, etc.)
    show_notifications: bool = True
    play_sounds: bool = False
    mute_during_recording: bool = True  # Mute system audio while recording


@dataclass
class Config:
    """Main application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return self.base_dir / "data"

    @property
    def assets_dir(self) -> Path:
        """Get the assets directory path."""
        return self.base_dir / "assets"

    @property
    def database_path(self) -> Path:
        """Get the database file path."""
        return self.data_dir / "transcriptions.db"

    @property
    def icon_path(self) -> Path:
        """Get the icon file path."""
        return self.assets_dir / "icon.ico"

    @property
    def settings_path(self) -> Path:
        """Get the user settings file path."""
        return self.data_dir / "settings.json"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def load_settings(self) -> None:
        """Load user settings from disk, overriding defaults."""
        if not self.settings_path.exists():
            return
        try:
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ui = data.get("ui", {})
            if "language" in ui:
                self.ui.language = ui["language"]
            if "play_sounds" in ui:
                self.ui.play_sounds = ui["play_sounds"]
            if "show_notifications" in ui:
                self.ui.show_notifications = ui["show_notifications"]
            if "mute_during_recording" in ui:
                self.ui.mute_during_recording = ui["mute_during_recording"]
            # Audio section is optional; legacy settings without it keep the default.
            audio = data.get("audio", {})
            if "input_device" in audio:
                self.audio.input_device = audio["input_device"] or None
            # Model section is optional; legacy settings without it keep the default.
            model = data.get("model", {})
            if "backend" in model:
                self.model.backend = model["backend"]
            if "idle_unload_seconds" in model:
                self.model.idle_unload_seconds = int(model["idle_unload_seconds"])
            if "vram_budget_gb" in model:
                self.model.vram_budget_gb = float(model["vram_budget_gb"])
            if "over_budget_idle_seconds" in model:
                self.model.over_budget_idle_seconds = int(model["over_budget_idle_seconds"])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load settings: {e}")

    def save_settings(self) -> None:
        """Save user settings to disk."""
        data = {
            "ui": {
                "language": self.ui.language,
                "play_sounds": self.ui.play_sounds,
                "show_notifications": self.ui.show_notifications,
                "mute_during_recording": self.ui.mute_during_recording,
            },
            "audio": {
                "input_device": self.audio.input_device,
            },
            "model": {
                "backend": self.model.backend,
                "idle_unload_seconds": self.model.idle_unload_seconds,
                "vram_budget_gb": self.model.vram_budget_gb,
                "over_budget_idle_seconds": self.model.over_budget_idle_seconds,
            }
        }
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save settings: {e}")


# Global configuration instance
config = Config()
config.load_settings()
