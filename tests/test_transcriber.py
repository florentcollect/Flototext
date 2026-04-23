import types
import unittest
from unittest.mock import patch

import numpy as np

from flototext.core.transcriber import Transcriber


class FailingModel:
    def transcribe(self, **_kwargs):
        raise RuntimeError("boom")


class FakeOutOfMemoryError(Exception):
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
        transcriber = Transcriber(on_error=errors.append)
        transcriber._model_loaded = True
        transcriber._model = FailingModel()

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                OutOfMemoryError=FakeOutOfMemoryError,
                is_available=lambda: False,
                empty_cache=lambda: None,
            )
        )

        with patch.dict("sys.modules", {"torch": fake_torch}), patch("builtins.print"):
            result = transcriber.transcribe(np.array([0.1], dtype=np.float32))

        self.assertFalse(result.success)
        self.assertIn("boom", result.error)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
