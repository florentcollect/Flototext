import types
import unittest
from unittest.mock import patch

import numpy as np

from flototext.core import asr_backends
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


class CanaryLoadOptionsTests(unittest.TestCase):
    """Guards the VRAM fix: left to its defaults, ONNX Runtime reserved 8.7 GB
    for 3.8 GB of weights. These options are easy to drop by accident."""

    def _load(self, cuda=True):
        recorded = {}

        def fake_load_model(name, **kwargs):
            recorded.update(kwargs)
            return FakeCanaryModel()

        available = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
        fake_ort = types.SimpleNamespace(get_available_providers=lambda: available)

        with patch.dict("sys.modules", {
            "onnx_asr": types.SimpleNamespace(load_model=fake_load_model),
            "onnxruntime": fake_ort,
        }), patch.object(asr_backends, "_ensure_cuda_dlls", lambda: None), patch("builtins.print"):
            backend = CanaryOnnxBackend()
            backend.load()
        return recorded

    def test_cuda_provider_gets_memory_options(self):
        options = dict(self._load()["providers"][0][1])

        # kNextPowerOfTwo (the default) doubles the arena and never gives it back.
        self.assertEqual(options["arena_extend_strategy"], "kSameAsRequested")
        # EXHAUSTIVE benchmarks every cuDNN algorithm, allocating each workspace.
        self.assertEqual(options["cudnn_conv_algo_search"], "HEURISTIC")
        # "1" (the default) offers cuDNN all free VRAM as convolution workspace.
        self.assertEqual(options["cudnn_conv_use_max_workspace"], "0")
        self.assertEqual(options["gpu_mem_limit"], 6 * 1024 ** 3)
        # Already the default: setting it would be noise.
        self.assertNotIn("do_copy_in_default_stream", options)

    def test_cpu_fallback_is_kept(self):
        self.assertEqual(self._load()["providers"][1], "CPUExecutionProvider")

    def test_resampler_stays_on_cpu(self):
        # onnx-asr builds one InferenceSession per source sample rate. We always
        # feed 16 kHz, so all seven would sit on the GPU doing nothing.
        for cuda in (True, False):
            with self.subTest(cuda=cuda):
                self.assertEqual(
                    self._load(cuda=cuda)["resampler_config"],
                    {"providers": ["CPUExecutionProvider"]},
                )

    def test_without_cuda_no_providers_are_forced(self):
        self.assertNotIn("providers", self._load(cuda=False))


if __name__ == "__main__":
    unittest.main()
