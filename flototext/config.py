"""Configuration for Flototext application."""

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
    language: str = "French"


@dataclass
class HotkeyConfig:
    """Hotkey configuration."""
    trigger_key: str = "f2"


@dataclass
class UIConfig:
    """UI configuration."""
    app_name: str = "Flototext"
    show_notifications: bool = True
    play_sounds: bool = True


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

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()
