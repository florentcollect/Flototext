import types
import unittest
from unittest.mock import patch

import numpy as np

from flototext.core import asr_backends
from flototext.core.asr_backends import (
    BaseASRBackend,
    QwenBackend,
    CanaryOnnxBackend,
    ParakeetOnnxBackend,
    create_backend,
)


class FactoryTests(unittest.TestCase):
    def test_create_backend_qwen_is_default(self):
        self.assertIsInstance(create_backend("qwen"), QwenBackend)
        self.assertIsInstance(create_backend("unknown"), QwenBackend)

    def test_create_backend_canary(self):
        self.assertIsInstance(create_backend("canary"), CanaryOnnxBackend)

    def test_create_backend_parakeet(self):
        self.assertIsInstance(create_backend("parakeet"), ParakeetOnnxBackend)

    def test_backends_implement_interface(self):
        for backend in (QwenBackend(), CanaryOnnxBackend(), ParakeetOnnxBackend()):
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


class ChunkBalanceTests(unittest.TestCase):
    def test_no_tiny_tail_reaches_the_engine(self):
        # 20 s + 100 samples: a fixed 20 s stride sent a 100-sample tail, under
        # the 320 samples onnx-asr's feature extractor needs.
        backend = CanaryOnnxBackend()
        backend._model = FakeCanaryModel()
        audio = np.zeros(16000 * 20 + 100, dtype=np.float32)

        backend.transcribe(audio, 16000, "French", "fr")

        self.assertEqual(sum(backend._model.calls), len(audio))
        self.assertGreaterEqual(min(backend._model.calls), 16000 * 10)
        self.assertLessEqual(max(backend._model.calls), 16000 * 20)


class UnkTokenTests(unittest.TestCase):
    def test_unk_tokens_are_not_pasted(self):
        class UnkModel:
            def recognize(self, audio, sample_rate, language):
                return "Fletch<unk> Friends, répéter<unk> il<unk> vient"

        backend = CanaryOnnxBackend()
        backend._model = UnkModel()
        text, _ = backend.transcribe(np.zeros(16000, dtype=np.float32), 16000, "French", "fr")
        self.assertEqual(text, "Fletch Friends, répéter il vient")


class FakeParakeetModel:
    """Parakeet v3 detects the language itself: recognize() takes none."""

    def __init__(self):
        self.calls = []

    def recognize(self, audio, sample_rate):
        self.calls.append(len(audio))
        return " seg "


class ParakeetTests(unittest.TestCase):
    def test_no_language_argument_and_chunking(self):
        backend = ParakeetOnnxBackend()
        backend._model = FakeParakeetModel()
        audio = np.zeros(16000 * 50, dtype=np.float32)

        text, code = backend.transcribe(audio, 16000, "French", "fr")

        self.assertEqual(len(backend._model.calls), 3)
        self.assertEqual(text, "seg seg seg")
        self.assertEqual(code, "fr")

    def test_loads_parakeet_model_with_same_memory_caps(self):
        recorded = {}

        def fake_load_model(name, **kwargs):
            recorded["name"] = name
            recorded.update(kwargs)
            return FakeParakeetModel()

        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        with patch.dict("sys.modules", {
            "onnx_asr": types.SimpleNamespace(load_model=fake_load_model),
            "onnxruntime": fake_ort,
        }), patch.object(asr_backends, "_ensure_cuda_dlls", lambda: None), patch("builtins.print"):
            ParakeetOnnxBackend().load()

        self.assertEqual(recorded["name"], "nemo-parakeet-tdt-0.6b-v3")
        self.assertEqual(recorded["providers"][0][1], CanaryOnnxBackend.CUDA_PROVIDER_OPTIONS)
        self.assertEqual(recorded["resampler_config"], {"providers": ["CPUExecutionProvider"]})


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
        self.assertEqual(options["gpu_mem_limit"], 4 * 1024 ** 3)
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


class FakeSession:
    """An InferenceSession stand-in that records the run options it gets."""

    def __init__(self, inputs=("audio_signal", "length")):
        self.runs = []
        self._inputs = [types.SimpleNamespace(name=n) for n in inputs]

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, input_feed, run_options=None):
        self.runs.append((input_feed, run_options))
        return ["out"]


class FakeRunOptions:
    def __init__(self):
        self.entries = {}

    def add_run_config_entry(self, key, value):
        self.entries[key] = value


def mems(steps):
    return np.zeros((2, 1, steps, 4), dtype=np.float32)


class ArenaShrinkageTests(unittest.TestCase):
    """Guards the fix for the arena filling up: 20 refusals out of 149
    back-to-back clips without it, none with it (2026-11-19)."""

    def _install(self, asr):
        backend = CanaryOnnxBackend()
        backend._model = types.SimpleNamespace(asr=asr)
        fake_ort = types.SimpleNamespace(RunOptions=FakeRunOptions)
        with patch.dict("sys.modules", {"onnxruntime": fake_ort}), patch("builtins.print"):
            backend._install_arena_shrinkage()
        return backend

    def test_every_encoder_run_asks_for_a_shrink(self):
        encoder = FakeSession()
        asr = types.SimpleNamespace(_encoder=encoder)
        self._install(asr)

        asr._encoder.run(["out"], {"audio_signal": 1})

        options = encoder.runs[0][1]
        self.assertEqual(options.entries, {"memory.enable_memory_arena_shrinkage": "gpu:0"})

    def test_decoder_shrinks_once_after_the_dictation_not_every_token(self):
        decoder = FakeSession(inputs=("input_ids", "decoder_mems"))
        asr = types.SimpleNamespace(_encoder=FakeSession(), _decoder=decoder)
        backend = self._install(asr)

        first = {"input_ids": 1, "decoder_mems": mems(0)}
        asr._decoder.run(["logits"], first)
        asr._decoder.run(["logits"], {"input_ids": 2, "decoder_mems": mems(1)})
        self.assertTrue(all(options is None for _, options in decoder.runs))

        backend._release_working_memory()
        replay, options = decoder.runs[-1]
        self.assertIs(replay, first)
        self.assertIn("memory.enable_memory_arena_shrinkage", options.entries)

        backend._release_working_memory()  # nothing kept: no second replay
        self.assertEqual(len(decoder.runs), 3)

    def test_transcribe_releases_even_when_recognition_fails(self):
        backend = CanaryOnnxBackend()
        released = []
        backend._releasing_decoder = types.SimpleNamespace(release=lambda: released.append(1))

        class Broken:
            def recognize(self, audio, sample_rate, language):
                raise RuntimeError("BFCArena")

        backend._model = Broken()
        with self.assertRaises(RuntimeError):
            backend.transcribe(np.zeros(16000, dtype=np.float32), 16000, "French", "fr")
        self.assertEqual(released, [1])

    def test_a_failed_release_does_not_cost_the_text(self):
        backend = CanaryOnnxBackend()
        backend._model = FakeCanaryModel()

        def boom():
            raise RuntimeError("cudaFree")

        backend._releasing_decoder = types.SimpleNamespace(release=boom)
        with patch("builtins.print"):
            text, _ = backend.transcribe(np.zeros(16000, dtype=np.float32), 16000, "French", "fr")
        self.assertEqual(text, "seg")

    def test_parakeet_joint_decoder_is_left_alone(self):
        joint = FakeSession(inputs=("encoder_outputs", "targets"))
        asr = types.SimpleNamespace(_encoder=FakeSession(), _decoder_joint=joint)
        backend = self._install(asr)
        self.assertIsNone(backend._releasing_decoder)
        self.assertIs(asr._decoder_joint, joint)

    def test_unknown_model_layout_is_skipped_quietly(self):
        backend = self._install(None)  # no asr attribute to wrap
        self.assertIsNone(backend._releasing_decoder)


if __name__ == "__main__":
    unittest.main()
