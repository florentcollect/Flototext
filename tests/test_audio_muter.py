import threading
import unittest
from unittest.mock import patch

from flototext.core.audio_muter import AudioMuter


class FakeVolumeInterface:
    def __init__(self):
        self.muted = False
        self.set_mute_calls = []

    def GetMute(self):
        return int(self.muted)

    def SetMute(self, value, _event_context):
        self.muted = bool(value)
        self.set_mute_calls.append(value)


class AudioMuterTests(unittest.TestCase):
    def _make_muter(self):
        muter = AudioMuter.__new__(AudioMuter)
        muter.enabled = True
        muter._volume_interface = FakeVolumeInterface()
        muter._was_muted = False
        muter._lock = threading.Lock()
        muter._is_muted_by_us = False
        return muter

    @patch("flototext.core.audio_muter.HAS_PYCAW", True)
    def test_disabling_while_muted_restores_audio(self):
        muter = self._make_muter()

        self.assertTrue(muter.mute())
        self.assertTrue(muter._volume_interface.muted)

        muter.set_enabled(False)

        self.assertFalse(muter._volume_interface.muted)
        self.assertFalse(muter._is_muted_by_us)
        self.assertFalse(muter.enabled)
        self.assertEqual(muter._volume_interface.set_mute_calls, [1, 0])

    @patch("flototext.core.audio_muter.HAS_PYCAW", True)
    def test_unmute_restores_audio_even_after_disabled(self):
        muter = self._make_muter()
        self.assertTrue(muter.mute())

        muter.enabled = False

        self.assertTrue(muter.unmute())
        self.assertFalse(muter._volume_interface.muted)
        self.assertFalse(muter._is_muted_by_us)


if __name__ == "__main__":
    unittest.main()
