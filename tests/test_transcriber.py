import types
import unittest
from unittest.mock import patch

import numpy as np

from flototext.core.transcriber import Transcriber


class FailingBackend:
    name = "failing"

    def transcribe(self, *_args, **_kwargs):
        raise RuntimeError("boom")

    def cleanup(self):
        pass


class TranscriberTests(unittest.TestCase):
    def test_dry_run_loads_without_model_and_returns_sample_text(self):
        loaded = []
        transcriber = Transcriber(on_model_loaded=lambda: loaded.append(True), dry_run=True)

        with patch("builtins.print"):
            transcriber._load_model()
        result = transcriber.transcribe(np.array([0.1], dtype=np.float32))

        self.assertTrue(transcriber.is_ready)
        self.assertEqual(loaded, [True])
        self.assertTrue(result.success)
        self.assertIn("deux-cent", result.text)

    def test_transcribe_returns_error_without_invoking_error_callback(self):
        errors = []
        transcriber = Transcriber(on_error=errors.append, dry_run=False)
        transcriber._model_loaded = True
        transcriber._backend = FailingBackend()

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                OutOfMemoryError=type("FakeOOM", (Exception,), {}),
                is_available=lambda: False,
                empty_cache=lambda: None,
            )
        )

        with patch.dict("sys.modules", {"torch": fake_torch}), patch("builtins.print"):
            result = transcriber.transcribe(np.array([0.1], dtype=np.float32))

        self.assertFalse(result.success)
        self.assertIn("boom", result.error)
        self.assertEqual(errors, [])

    def test_load_model_selects_backend_from_config(self):
        # Default config selects the Qwen backend without actually loading it.
        from flototext.core import transcriber as transcriber_module
        from flototext.config import config

        captured = {}

        class DummyBackend:
            name = "dummy"

            def load(self):
                captured["loaded"] = True

            def cleanup(self):
                pass

        def fake_create_backend(name):
            captured["requested"] = name
            return DummyBackend()

        transcriber = Transcriber(dry_run=False)
        with patch.object(transcriber_module, "create_backend", fake_create_backend), \
                patch("builtins.print"):
            transcriber._load_model()

        self.assertEqual(captured["requested"], config.model.backend)
        self.assertTrue(captured["loaded"])
        self.assertTrue(transcriber.is_ready)
        self.assertEqual(transcriber.backend_name, "dummy")


if __name__ == "__main__":
    unittest.main()
