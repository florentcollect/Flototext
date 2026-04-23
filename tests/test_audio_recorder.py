import unittest
from unittest.mock import patch

import numpy as np

from flototext.core.audio_recorder import AudioRecorder


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

    @patch("flototext.core.audio_recorder.time.time", side_effect=[10.0, 10.1])
    @patch("flototext.core.audio_recorder.sd.InputStream", FakeInputStream)
    def test_short_recording_is_invalid(self, _time_mock):
        recorder = AudioRecorder()

        self.assertTrue(recorder.start_recording())
        stream = FakeInputStream.instances[0]
        stream.kwargs["callback"](np.array([[0.1]], dtype=np.float32), 1, None, None)

        result = recorder.stop_recording()

        self.assertFalse(result.is_valid)


if __name__ == "__main__":
    unittest.main()
