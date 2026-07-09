"""Hotkey management using pynput."""

import ctypes
import sys
import threading
from typing import Callable, Optional
from pynput import keyboard

from ..config import config


# Windows virtual-key codes, used to poll the real state of the trigger key.
_SPECIAL_VK = {
    'ctrl': 0x11,
    'alt': 0x12,
    'shift': 0x10,
    'space': 0x20,
    'enter': 0x0D,
    'tab': 0x09,
    'escape': 0x1B,
    'esc': 0x1B,
}
_VK_F1 = 0x70


class HotkeyManager:
    """Manages global hotkey detection for push-to-talk."""

    # A missed release is confirmed over two polls so a key sampled between
    # scan-code repeats is never mistaken for a released one.
    _WATCHDOG_INTERVAL = 0.25
    _WATCHDOG_CONFIRMATIONS = 2

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

        # Watchdog state. pynput drops the release event whenever its low-level
        # hook is bypassed -- most often while an elevated window holds focus.
        # _is_pressed then stays True forever, the recorder never stops and every
        # later press is swallowed by the `not self._is_pressed` guard. Polling
        # the physical key state lets us recover instead of wedging.
        self._trigger_vk = self._parse_vk(config.hotkey.trigger_key)
        self._can_poll_key = sys.platform == 'win32' and self._trigger_vk is not None
        self._watchdog_stop = threading.Event()
        self._watchdog: Optional[threading.Thread] = None
        self._listener_lock = threading.Lock()

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

    def _parse_vk(self, key_string: str) -> Optional[int]:
        """Parse a key string to a Windows virtual-key code.

        Args:
            key_string: Key name (e.g., 'f2', 'ctrl', 'shift').

        Returns:
            The virtual-key code, or None if the key cannot be polled.
        """
        key_string = key_string.lower()

        if key_string.startswith('f') and key_string[1:].isdigit():
            key_num = int(key_string[1:])
            if 1 <= key_num <= 24:
                return _VK_F1 + key_num - 1
            return None

        return _SPECIAL_VK.get(key_string)

    def _dispatch(self, callback: Optional[Callable]) -> None:
        """Run a callback off the listener thread so it never blocks input."""
        if callback:
            threading.Thread(target=callback, daemon=True).start()

    def _on_press(self, key) -> None:
        """Handle key press events."""
        if not self._enabled:
            return

        try:
            fire = False
            with self._lock:
                if key == self._trigger_key and not self._is_pressed:
                    self._is_pressed = True
                    fire = True
            if fire:
                self._dispatch(self.on_key_press)
        except Exception as e:
            print(f"Error in key press handler: {e}")

    def _on_release(self, key) -> None:
        """Handle key release events."""
        if not self._enabled:
            return

        try:
            if key == self._trigger_key:
                self._release()
        except Exception as e:
            print(f"Error in key release handler: {e}")

    def _press(self) -> bool:
        """Mark the trigger key pressed and fire the callback exactly once.

        Returns:
            True if this call performed the transition.
        """
        with self._lock:
            if self._is_pressed:
                return False
            self._is_pressed = True

        self._dispatch(self.on_key_press)
        return True

    def _release(self) -> bool:
        """Mark the trigger key released and fire the callback exactly once.

        Returns:
            True if this call performed the transition.
        """
        with self._lock:
            if not self._is_pressed:
                return False
            self._is_pressed = False

        self._dispatch(self.on_key_release)
        return True

    def _key_physically_down(self) -> bool:
        """Check whether the trigger key is currently held down."""
        # GetAsyncKeyState sets the high-order bit while the key is down.
        state = ctypes.windll.user32.GetAsyncKeyState(self._trigger_vk)
        return bool(state & 0x8000)

    def _restart_listener(self) -> None:
        """Rebuild the pynput listener after Windows silently unhooked it.

        A listener whose low-level hook was removed (LowLevelHooksTimeout) keeps
        pumping messages, so `is_alive()` still reports True while no key event
        ever arrives again. Rebuilding is the only way back.
        """
        with self._listener_lock:
            old = self._listener
            self._listener = None
            if old is not None:
                try:
                    old.stop()
                except Exception as e:
                    print(f"Error stopping stale listener: {e}")

            listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            listener.start()
            self._listener = listener
            print("Hotkey listener rebuilt (events were no longer arriving)")

    def _watchdog_loop(self) -> None:
        """Repair hotkey state that pynput failed to deliver.

        Two independent faults produce an identical symptom -- a stuck flag and a
        recording that never ends -- and the log cannot tell them apart, so both
        are handled: a dropped release event, and a listener that stopped
        receiving events entirely.
        """
        stuck_down = 0
        unseen_down = 0

        while not self._watchdog_stop.wait(self._WATCHDOG_INTERVAL):
            try:
                if not self._enabled:
                    stuck_down = unseen_down = 0
                    continue

                down = self._key_physically_down()

                # The key is up but we still believe it is held: the release
                # event was dropped. Ending the press unblocks every later press.
                if self._is_pressed and not down:
                    stuck_down += 1
                    if stuck_down >= self._WATCHDOG_CONFIRMATIONS:
                        stuck_down = 0
                        if self._release():
                            print("Missed key release recovered by watchdog")
                    continue
                stuck_down = 0

                # The key is held but no press ever reached us: the listener has
                # gone deaf. Rebuild it, then honour the press the user is making.
                if down and not self._is_pressed:
                    unseen_down += 1
                    if unseen_down >= self._WATCHDOG_CONFIRMATIONS:
                        unseen_down = 0
                        self._restart_listener()
                        self._press()
                    continue
                unseen_down = 0
            except Exception as e:
                print(f"Error in hotkey watchdog: {e}")
                stuck_down = unseen_down = 0

    def start(self) -> None:
        """Start listening for hotkeys."""
        with self._listener_lock:
            if self._listener is not None:
                return

            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self._listener.start()

        if self._can_poll_key:
            self._watchdog_stop.clear()
            self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog.start()

        print(f"Hotkey listener started (trigger: {config.hotkey.trigger_key.upper()})")

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        self._watchdog_stop.set()
        self._watchdog = None

        with self._listener_lock:
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
