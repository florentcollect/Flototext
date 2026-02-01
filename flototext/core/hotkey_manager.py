"""Hotkey management using pynput."""

import threading
from typing import Callable, Optional
from pynput import keyboard

from ..config import config


class HotkeyManager:
    """Manages global hotkey detection for push-to-talk."""

    def __init__(
        self,
        on_key_press: Optional[Callable] = None,
        on_key_release: Optional[Callable] = None
    ):
        """Initialize the hotkey manager.

        Args:
            on_key_press: Callback when trigger key is pressed.
            on_key_release: Callback when trigger key is released.
        """
        self.on_key_press = on_key_press
        self.on_key_release = on_key_release

        self._listener: Optional[keyboard.Listener] = None
        self._is_pressed = False
        self._lock = threading.Lock()
        self._enabled = True

        # Map trigger key string to pynput key
        self._trigger_key = self._parse_key(config.hotkey.trigger_key)

    def _parse_key(self, key_string: str) -> keyboard.Key:
        """Parse a key string to a pynput key.

        Args:
            key_string: Key name (e.g., 'f2', 'ctrl', 'shift').

        Returns:
            Corresponding pynput key.
        """
        key_string = key_string.lower()

        # Function keys
        if key_string.startswith('f') and key_string[1:].isdigit():
            key_num = int(key_string[1:])
            return getattr(keyboard.Key, f'f{key_num}')

        # Special keys mapping
        special_keys = {
            'ctrl': keyboard.Key.ctrl,
            'alt': keyboard.Key.alt,
            'shift': keyboard.Key.shift,
            'space': keyboard.Key.space,
            'enter': keyboard.Key.enter,
            'tab': keyboard.Key.tab,
            'escape': keyboard.Key.esc,
            'esc': keyboard.Key.esc,
        }

        return special_keys.get(key_string, keyboard.Key.f2)

    def _on_press(self, key) -> None:
        """Handle key press events."""
        if not self._enabled:
            return

        try:
            with self._lock:
                if key == self._trigger_key and not self._is_pressed:
                    self._is_pressed = True
                    if self.on_key_press:
                        # Run callback in separate thread to avoid blocking
                        threading.Thread(
                            target=self.on_key_press,
                            daemon=True
                        ).start()
        except Exception as e:
            print(f"Error in key press handler: {e}")

    def _on_release(self, key) -> None:
        """Handle key release events."""
        if not self._enabled:
            return

        try:
            with self._lock:
                if key == self._trigger_key and self._is_pressed:
                    self._is_pressed = False
                    if self.on_key_release:
                        # Run callback in separate thread to avoid blocking
                        threading.Thread(
                            target=self.on_key_release,
                            daemon=True
                        ).start()
        except Exception as e:
            print(f"Error in key release handler: {e}")

    def start(self) -> None:
        """Start listening for hotkeys."""
        if self._listener is not None:
            return

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()
        print(f"Hotkey listener started (trigger: {config.hotkey.trigger_key.upper()})")

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            print("Hotkey listener stopped")

    def enable(self) -> None:
        """Enable hotkey detection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable hotkey detection."""
        self._enabled = False

    @property
    def is_running(self) -> bool:
        """Check if listener is running."""
        return self._listener is not None and self._listener.is_alive()

    @property
    def is_enabled(self) -> bool:
        """Check if hotkey detection is enabled."""
        return self._enabled
