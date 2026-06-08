import unittest

import numpy as np

from flototext.core.asr_backends import (
    BaseASRBackend,
    QwenBackend,
    CanaryOnnxBackend,
    create_backend,
)


class FactoryTests(unittest.TestCase):
    def test_create_backend_qwen_is_default(self):
        self.assertIsInstance(create_backend("qwen"), QwenBackend)
        self.assertIsInstance(create_backend("unknown"), QwenBackend)

    def test_create_backend_canary(self):
        self.assertIsInstance(create_backend("canary"), CanaryOnnxBackend)

    def test_backends_implement_interface(self):
        for backend in (QwenBackend(), CanaryOnnxBackend()):
            self.assertIsInstance(backend, BaseASRBackend)
            self.assertTrue(hasattr(backend, "load"))
            self.assertTrue(callable(backend.transcribe))
            self.assertTrue(callable(backend.cleanup))
            # cleanup must be safe before load
            backend.cleanup()


class FakeCanaryModel:
    """Records each recognize() call so we can assert the chunking behaviour."""

    def __init__(self):
        self.calls = []

    def recognize(self, audio, sample_rate, language):
        self.calls.append(len(audio))
        return "seg"


class CanaryChunkingTests(unittest.TestCase):
    def test_short_audio_is_single_call(self):
        backend = CanaryOnnxBackend()
        backend._model = FakeCanaryModel()
        audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 s < 20 s limit

        text, code = backend.transcribe(audio, 16000, "French", "fr")

        self.assertEqual(len(backend._model.calls), 1)
        self.assertEqual(text, "seg")
        self.assertEqual(code, "fr")

    def test_long_audio_is_chunked_and_joined(self):
        backend = CanaryOnnxBackend()
        backend._model = FakeCanaryModel()
        # 50 s with a 20 s window -> 3 chunks (20 + 20 + 10).
        audio = np.zeros(16000 * 50, dtype=np.float32)

        text, _ = backend.transcribe(audio, 16000, "French", "fr")

        self.assertEqual(len(backend._model.calls), 3)
        self.assertEqual(text, "seg seg seg")


if __name__ == "__main__":
    unittest.main()
