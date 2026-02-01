"""Audio muting module to silence system sounds during recording."""

import threading
from typing import Optional

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    HAS_PYCAW = True
except ImportError:
    HAS_PYCAW = False


class AudioMuter:
    """Mutes system audio during recording to prevent interference."""

    def __init__(self, enabled: bool = True):
        """Initialize the audio muter.

        Args:
            enabled: Whether muting is enabled.
        """
        self.enabled = enabled
        self._volume_interface: Optional[IAudioEndpointVolume] = None
        self._was_muted: bool = False
        self._previous_volume: float = 1.0
        self._lock = threading.Lock()
        self._is_muted_by_us = False

        if HAS_PYCAW:
            self._init_audio_interface()
        else:
            print("Warning: pycaw not available, audio muting disabled")

    def _init_audio_interface(self) -> None:
        """Initialize the Windows audio interface."""
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            self._volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            print(f"Error initializing audio interface: {e}")
            self._volume_interface = None

    def mute(self) -> bool:
        """Mute system audio.

        Returns:
            True if muted successfully.
        """
        if not self.enabled or not HAS_PYCAW or not self._volume_interface:
            return False

        with self._lock:
            if self._is_muted_by_us:
                return True  # Already muted by us

            try:
                # Save current mute state
                self._was_muted = bool(self._volume_interface.GetMute())

                # Only mute if not already muted
                if not self._was_muted:
                    self._volume_interface.SetMute(1, None)
                    self._is_muted_by_us = True
                    return True
                return True
            except Exception as e:
                print(f"Error muting audio: {e}")
                return False

    def unmute(self) -> bool:
        """Restore audio to previous state.

        Returns:
            True if unmuted successfully.
        """
        if not self.enabled or not HAS_PYCAW or not self._volume_interface:
            return False

        with self._lock:
            if not self._is_muted_by_us:
                return True  # We didn't mute it

            try:
                # Only unmute if we muted it and it wasn't muted before
                if not self._was_muted:
                    self._volume_interface.SetMute(0, None)
                self._is_muted_by_us = False
                return True
            except Exception as e:
                print(f"Error unmuting audio: {e}")
                return False

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable audio muting.

        Args:
            enabled: Whether to enable muting.
        """
        self.enabled = enabled
        # If disabling while muted, unmute first
        if not enabled and self._is_muted_by_us:
            self.unmute()

    @property
    def is_available(self) -> bool:
        """Check if audio muting is available."""
        return HAS_PYCAW and self._volume_interface is not None
