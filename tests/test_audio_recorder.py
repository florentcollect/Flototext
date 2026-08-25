import unittest
from unittest.mock import patch

import numpy as np

from flototext.config import config
from flototext.core.audio_recorder import AudioRecorder, is_silent, peak_window_rms


class FakeInputStream:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        FakeInputStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class AudioRecorderTests(unittest.TestCase):
    def setUp(self):
        FakeInputStream.instances = []

    @patch("flototext.core.audio_recorder.time.time", side_effect=[10.0, 11.0])
    @patch("flototext.core.audio_recorder.sd.InputStream", FakeInputStream)
    def test_records_concatenates_and_flattens_audio(self, _time_mock):
        started = []
        stopped = []
        recorder = AudioRecorder(on_start=lambda: started.append(True), on_stop=lambda: stopped.append(True))

        self.assertTrue(recorder.start_recording())
        stream = FakeInputStream.instances[0]
        stream.kwargs["callback"](np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None)
        stream.kwargs["callback"](np.array([[0.3]], dtype=np.float32), 1, None, None)

        result = recorder.stop_recording()

        self.assertEqual(started, [True])
        self.assertEqual(stopped, [True])
        self.assertTrue(stream.started)
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.duration, 1.0)
        np.testing.assert_array_equal(result.audio_data, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    @patch("flototext.core.audio_recorder.time.time", side_effect=[10.0, 11.0])
    @patch("flototext.core.audio_recorder.sd.InputStream", FakeInputStream)
    def test_recording_is_capped_at_max_duration(self, _time_mock):
        recorder = AudioRecorder()
        recorder._max_samples = 3  # Simulate a tiny max_duration

        self.assertTrue(recorder.start_recording())
        stream = FakeInputStream.instances[0]
        stream.kwargs["callback"](np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None)
        stream.kwargs["callback"](np.array([[0.3], [0.4]], dtype=np.float32), 2, None, None)
        # Beyond the cap: must be dropped
        stream.kwargs["callback"](np.array([[0.5]], dtype=np.float32), 1, None, None)

        result = recorder.stop_recording()

        np.testing.assert_array_equal(
            result.audio_data, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        )

    @patch("flototext.core.audio_recorder.time.time", side_effect=[10.0, 10.1])
    @patch("flototext.core.audio_recorder.sd.InputStream", FakeInputStream)
    def test_short_recording_is_invalid(self, _time_mock):
        recorder = AudioRecorder()

        self.assertTrue(recorder.start_recording())
        stream = FakeInputStream.instances[0]
        stream.kwargs["callback"](np.array([[0.1]], dtype=np.float32), 1, None, None)

        result = recorder.stop_recording()

        self.assertFalse(result.is_valid)


class SilenceDetectionTests(unittest.TestCase):
    """Silence must be refused: the ASR turns a noise floor into plausible text."""

    THRESHOLD = 0.0015

    def test_dead_stream_is_silent(self):
        # A microphone Windows reassigned elsewhere delivers exact zeros.
        self.assertTrue(is_silent(np.zeros(16000, dtype=np.float32), self.THRESHOLD))

    def test_empty_buffer_is_silent(self):
        self.assertTrue(is_silent(np.array([], dtype=np.float32), self.THRESHOLD))

    def test_noise_floor_is_silent(self):
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(16000) * 0.0003).astype(np.float32)
        self.assertTrue(is_silent(noise, self.THRESHOLD))

    def test_speech_level_audio_is_not_silent(self):
        rng = np.random.default_rng(1)
        speech = (rng.standard_normal(16000) * 0.05).astype(np.float32)
        self.assertFalse(is_silent(speech, self.THRESHOLD))

    def test_quiet_voice_above_threshold_survives(self):
        # Measured idle noise floor on a Shure MV7+ is ~0.002 RMS: a real but
        # quiet voice sits above it and must never be rejected.
        quiet = np.full(16000, 0.002, dtype=np.float32)
        self.assertFalse(is_silent(quiet, self.THRESHOLD))

    def _speech_surrounded_by_silence(self, speech_seconds, total_seconds, level):
        """Build a capture holding one phrase inside the usual leading/trailing gap."""
        rng = np.random.default_rng(2)
        capture = (rng.standard_normal(int(total_seconds * 16000)) * 0.00003)
        start = (len(capture) - int(speech_seconds * 16000)) // 2
        speech = rng.standard_normal(int(speech_seconds * 16000)) * level
        capture[start:start + len(speech)] = speech
        return capture.astype(np.float32)

    def test_short_phrase_in_a_long_take_survives(self):
        # The failure that dropped real dictations: one second of speech inside a
        # ten-second hold averages below the threshold even though it was spoken.
        capture = self._speech_surrounded_by_silence(1.0, 10.0, level=0.0035)
        overall = float(np.sqrt(np.mean(capture.astype(np.float64) ** 2)))
        self.assertLess(overall, self.THRESHOLD, "test case no longer reproduces the bug")
        self.assertFalse(is_silent(capture, self.THRESHOLD))

    def test_quiet_phrase_at_measured_level_survives(self):
        # Levels measured on the user's under-gained MV7+: speech peaks at
        # ~0.0025 RMS while the whole take averages ~0.0008.
        capture = self._speech_surrounded_by_silence(1.5, 6.0, level=0.0025)
        self.assertFalse(is_silent(capture, self.THRESHOLD))

    def test_long_dead_stream_stays_silent(self):
        # Windowing must not turn a dead five-minute stream into speech.
        self.assertTrue(is_silent(np.zeros(300 * 16000, dtype=np.float32), self.THRESHOLD))

    def test_peak_window_rms_finds_the_loudest_window(self):
        capture = self._speech_surrounded_by_silence(1.0, 10.0, level=0.05)
        self.assertGreater(peak_window_rms(capture), 0.03)

    def test_peak_window_rms_of_empty_capture_is_zero(self):
        self.assertEqual(peak_window_rms(np.array([], dtype=np.float32)), 0.0)

    def test_non_finite_capture_is_silent(self):
        # A driver fault can deliver NaN; it must read as silence, not speech.
        broken = np.full(16000, np.nan, dtype=np.float32)
        self.assertTrue(is_silent(broken, self.THRESHOLD))


class InputDeviceResolutionTests(unittest.TestCase):
    """Pinning the microphone by name survives Windows changing the default."""

    DEVICES = [
        {"name": "Mappeur de sons Microsoft - Input", "max_input_channels": 2},
        {"name": "MOTIV Mix Virtual Output (Shure Virtual Audio)", "max_input_channels": 2},
        {"name": "Speakers (Realtek)", "max_input_channels": 0},
        {"name": "Microphone (3- Shure MV7+)", "max_input_channels": 1},
    ]

    def setUp(self):
        self._saved = config.audio.input_device
        self.addCleanup(lambda: setattr(config.audio, "input_device", self._saved))
        with patch("flototext.core.audio_recorder.sd"):
            self.recorder = AudioRecorder()

    def _resolve(self, wanted):
        config.audio.input_device = wanted
        with patch("flototext.core.audio_recorder.sd.query_devices", return_value=self.DEVICES):
            return self.recorder._resolve_input_device()

    def test_unset_device_follows_system_default(self):
        self.assertIsNone(self._resolve(None))

    def test_name_is_matched_case_insensitively(self):
        self.assertEqual(self._resolve("shure mv7"), 3)

    def test_output_only_device_is_never_selected(self):
        self.assertEqual(self._resolve("Realtek"), None)

    def test_unknown_device_falls_back_to_default_rather_than_failing(self):
        self.assertIsNone(self._resolve("Nonexistent Mic"))

    def test_query_failure_falls_back_to_default(self):
        config.audio.input_device = "Shure MV7"
        with patch("flototext.core.audio_recorder.sd.query_devices", side_effect=OSError("boom")):
            self.assertIsNone(self.recorder._resolve_input_device())


if __name__ == "__main__":
    unittest.main()
