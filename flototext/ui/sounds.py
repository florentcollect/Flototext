"""Sound feedback module using winsound."""

import winsound
import threading
from typing import Optional

from ..config import config


class SoundManager:
    """Manages audio feedback sounds."""

    # Frequency constants for beeps (Hz)
    FREQ_LOW = 400
    FREQ_MID = 600
    FREQ_HIGH = 800

    # Duration constants (ms)
    DURATION_SHORT = 100
    DURATION_MEDIUM = 150
    DURATION_LONG = 200

    def __init__(self, enabled: bool = None):
        """Initialize the sound manager.

        Args:
            enabled: Whether sounds are enabled (default from config).
        """
        self.enabled = enabled if enabled is not None else config.ui.play_sounds

    def _play_beep_async(self, frequency: int, duration: int) -> None:
        """Play a beep sound asynchronously.

        Args:
            frequency: Frequency in Hz.
            duration: Duration in milliseconds.
        """
        if not self.enabled:
            return

        def play():
            try:
                winsound.Beep(frequency, duration)
            except Exception as e:
                print(f"Error playing sound: {e}")

        thread = threading.Thread(target=play, daemon=True)
        thread.start()

    def play_start_recording(self) -> None:
        """Play sound when recording starts (rising tone)."""
        if not self.enabled:
            return

        def play():
            try:
                winsound.Beep(self.FREQ_LOW, self.DURATION_SHORT)
                winsound.Beep(self.FREQ_HIGH, self.DURATION_SHORT)
            except Exception as e:
                print(f"Error playing start sound: {e}")

        thread = threading.Thread(target=play, daemon=True)
        thread.start()

    def play_stop_recording(self) -> None:
        """Play sound when recording stops (falling tone)."""
        if not self.enabled:
            return

        def play():
            try:
                winsound.Beep(self.FREQ_HIGH, self.DURATION_SHORT)
                winsound.Beep(self.FREQ_LOW, self.DURATION_SHORT)
            except Exception as e:
                print(f"Error playing stop sound: {e}")

        thread = threading.Thread(target=play, daemon=True)
        thread.start()

    def play_success(self) -> None:
        """Play success sound (pleasant double beep)."""
        if not self.enabled:
            return

        def play():
            try:
                winsound.Beep(self.FREQ_MID, self.DURATION_SHORT)
                winsound.Beep(self.FREQ_HIGH, self.DURATION_MEDIUM)
            except Exception as e:
                print(f"Error playing success sound: {e}")

        thread = threading.Thread(target=play, daemon=True)
        thread.start()

    def play_error(self) -> None:
        """Play error sound (low buzz)."""
        if not self.enabled:
            return

        def play():
            try:
                winsound.Beep(200, self.DURATION_LONG)
                winsound.Beep(200, self.DURATION_LONG)
            except Exception as e:
                print(f"Error playing error sound: {e}")

        thread = threading.Thread(target=play, daemon=True)
        thread.start()

    def play_ready(self) -> None:
        """Play ready/model loaded sound (triple ascending beep)."""
        if not self.enabled:
            return

        def play():
            try:
                winsound.Beep(self.FREQ_LOW, self.DURATION_SHORT)
                winsound.Beep(self.FREQ_MID, self.DURATION_SHORT)
                winsound.Beep(self.FREQ_HIGH, self.DURATION_MEDIUM)
            except Exception as e:
                print(f"Error playing ready sound: {e}")

        thread = threading.Thread(target=play, daemon=True)
        thread.start()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable sounds.

        Args:
            enabled: Whether to enable sounds.
        """
        self.enabled = enabled
