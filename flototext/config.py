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


@dataclass
class ModelConfig:
    """Qwen3-ASR model configuration."""
    model_name: str = "Qwen/Qwen3-ASR-1.7B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"  # Optimal for RTX 4090
    max_new_tokens: int = 512


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
