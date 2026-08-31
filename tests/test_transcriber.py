import threading
import time
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


class SlowBackend:
    """Records when transcriptions overlap, so we can prove they never do."""

    name = "slow"

    def __init__(self, delay=0.05):
        self.delay = delay
        self.actifs = 0
        self.max_actifs = 0
        self.cleaned = False
        self._lock = threading.Lock()

    def transcribe(self, *_args, **_kwargs):
        with self._lock:
            self.actifs += 1
            self.max_actifs = max(self.max_actifs, self.actifs)
        time.sleep(self.delay)
        with self._lock:
            self.actifs -= 1
        return "ok", "fr"

    def cleanup(self):
        self.cleaned = True


def _pret(transcriber, backend):
    """Put a transcriber in the state a successful load would leave it in."""
    transcriber._backend = backend
    transcriber._model_loaded = True
    transcriber._loading = False
    transcriber._last_used = time.monotonic()
    return transcriber


class IdleUnloadTests(unittest.TestCase):
    """The ONNX arena never shrinks, so releasing the model between working
    sessions is the only thing that hands its VRAM back."""

    def test_unloads_once_idle_long_enough(self):
        backend = SlowBackend()
        t = _pret(Transcriber(dry_run=False), backend)
        t._last_used = time.monotonic() - 60

        with patch("builtins.print"):
            self.assertTrue(t.unload_if_idle(idle_seconds=30))

        self.assertFalse(t.is_ready)
        self.assertTrue(t.was_unloaded_when_idle)
        self.assertTrue(backend.cleaned)

    def test_keeps_model_while_still_recent(self):
        t = _pret(Transcriber(dry_run=False), SlowBackend())

        self.assertFalse(t.unload_if_idle(idle_seconds=30))
        self.assertTrue(t.is_ready)

    def test_zero_disables_the_unload(self):
        t = _pret(Transcriber(dry_run=False), SlowBackend())
        t._last_used = time.monotonic() - 10_000

        self.assertFalse(t.unload_if_idle(idle_seconds=0))
        self.assertTrue(t.is_ready)

    def test_transcribing_refreshes_the_idle_clock(self):
        t = _pret(Transcriber(dry_run=False), SlowBackend(delay=0))
        t._last_used = time.monotonic() - 60

        t.transcribe(np.zeros(16, dtype=np.float32))

        self.assertFalse(t.unload_if_idle(idle_seconds=30))

    def test_never_unloads_under_a_running_transcription(self):
        """The watchdog fires on a timer and must skip a busy engine rather
        than pull the backend out from under it."""
        backend = SlowBackend(delay=0.3)
        t = _pret(Transcriber(dry_run=False), backend)
        t._last_used = time.monotonic() - 60
        resultats = []

        fil = threading.Thread(
            target=lambda: resultats.append(t.transcribe(np.zeros(16, dtype=np.float32)))
        )
        fil.start()
        time.sleep(0.1)  # let the transcription take the engine lock
        with patch("builtins.print"):
            unloaded = t.unload_if_idle(idle_seconds=30)
        fil.join()

        self.assertFalse(unloaded)
        self.assertTrue(resultats[0].success)

    def test_concurrent_transcriptions_are_serialised(self):
        backend = SlowBackend(delay=0.05)
        t = _pret(Transcriber(dry_run=False), backend)

        fils = [threading.Thread(target=lambda: t.transcribe(np.zeros(16, dtype=np.float32)))
                for _ in range(4)]
        for f in fils:
            f.start()
        for f in fils:
            f.join()

        self.assertEqual(backend.max_actifs, 1)

    def test_ensure_loaded_returns_true_when_already_loaded(self):
        t = _pret(Transcriber(dry_run=False), SlowBackend())
        self.assertTrue(t.ensure_loaded(timeout=0.5))

    def test_ensure_loaded_reloads_after_an_idle_unload(self):
        t = Transcriber(dry_run=True)
        with patch("builtins.print"):
            t._load_model()
            t._last_used = time.monotonic() - 60
            self.assertTrue(t.unload_if_idle(idle_seconds=30))
            self.assertTrue(t.ensure_loaded(timeout=5))

        self.assertTrue(t.is_ready)
        # A completed reload clears the idle marker.
        self.assertFalse(t.was_unloaded_when_idle)

    def test_ensure_loaded_gives_up_when_loading_fails(self):
        t = Transcriber(dry_run=False)
        with patch("flototext.core.transcriber.create_backend",
                   side_effect=RuntimeError("pas de GPU")), patch("builtins.print"):
            self.assertFalse(t.ensure_loaded(timeout=5))
