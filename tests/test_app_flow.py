"""FlototextApp orchestration, driven with fakes (no model, mic or tray).

Every lost dictation found in the log went through main.py, which had no test.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from flototext.main import FlototextApp
from flototext.core.transcriber import TranscriptionResult
from flototext.ui.tray_app import AppState


class FakeTranscriber:
    def __init__(self, ready=True, loading=False, unloaded=False, results=None):
        self.is_ready = ready
        self.is_loading = loading
        self.was_unloaded_when_idle = unloaded
        self.load_calls = 0
        self.reload_calls = 0
        self.results = list(results or [])

    def load_model_async(self):
        self.load_calls += 1

    def ensure_loaded(self):
        return self.is_ready

    def reload_backend(self):
        self.reload_calls += 1

    def transcribe(self, audio, sample_rate):
        return self.results.pop(0)


class FakeTray:
    def __init__(self):
        self.states = []

    def set_state(self, state):
        self.states.append(state)

    def increment_transcription_count(self):
        pass

    def refresh_ui(self):
        pass


def make_app(transcriber):
    app = FlototextApp.__new__(FlototextApp)
    app._transcriber = transcriber
    app._tray_app = FakeTray()
    app._sound_manager = MagicMock()
    app._notification_manager = MagicMock()
    app._audio_recorder = MagicMock(is_recording=False)
    app._audio_muter = MagicMock()
    app._text_corrector = types.SimpleNamespace(correct=lambda text: text)
    app._text_inserter = MagicMock()
    app._text_inserter.insert_text.return_value = True
    app._database = MagicMock()
    app._running = True
    app._processing = False
    app._failed_capture = None
    return app


SPEECH = np.full(16000, 0.3, dtype=np.float32)


def ok(text="bonjour"):
    return TranscriptionResult(text=text, language="French", success=True)


def failed(error="BFCArena failed"):
    return TranscriptionResult(text="", language="French", success=False, error=error)


class SyncThread:
    """Runs a would-be background thread inline, so tests stay deterministic."""

    def __init__(self, target=None, daemon=None, args=()):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


class HotkeyPressTests(unittest.TestCase):
    def test_press_after_a_failed_load_retries_the_load(self):
        app = make_app(FakeTranscriber(ready=False))
        with patch("builtins.print"):
            app._on_hotkey_press()
        self.assertEqual(app._transcriber.load_calls, 1)
        app._audio_recorder.start_recording.assert_not_called()

    def test_press_during_a_load_does_not_record(self):
        # Creating the ONNX sessions freezes the audio callback (2026-09-13).
        app = make_app(FakeTranscriber(ready=False, loading=True))
        with patch("builtins.print"):
            app._on_hotkey_press()
        app._audio_recorder.start_recording.assert_not_called()
        self.assertEqual(app._transcriber.load_calls, 0)

    def test_press_after_an_idle_unload_records(self):
        app = make_app(FakeTranscriber(ready=False, unloaded=True))
        with patch("builtins.print"):
            app._on_hotkey_press()
        app._audio_recorder.start_recording.assert_called_once()
        self.assertEqual(app._transcriber.load_calls, 0)

    def test_microphone_that_will_not_open_is_reported(self):
        app = make_app(FakeTranscriber())
        app._audio_recorder.start_recording.return_value = False
        with patch("builtins.print"):
            app._on_hotkey_press()
        app._notification_manager.notify_error.assert_called_once()
        app._sound_manager.play_error.assert_called_once()
        app._audio_muter.mute.assert_not_called()


class FailedDictationTests(unittest.TestCase):
    def test_failed_capture_is_kept_and_retry_pastes_it(self):
        app = make_app(FakeTranscriber(results=[failed(), ok("texte retrouvé")]))
        with patch("builtins.print"), patch("flototext.main.threading.Thread", SyncThread), \
                patch("flototext.main.time.sleep"):
            app._process_transcription(SPEECH, 1.0)
            self.assertIsNotNone(app._failed_capture)
            app._text_inserter.insert_text.assert_not_called()

            app._on_retry_last()

        app._text_inserter.insert_text.assert_called_once_with("texte retrouvé")
        self.assertIsNone(app._failed_capture)
        self.assertFalse(app._processing)

    def test_icon_stays_grey_when_the_model_is_missing(self):
        app = make_app(FakeTranscriber(ready=False))
        with patch("builtins.print"), patch("flototext.main.threading.Thread", SyncThread), \
                patch("flototext.main.time.sleep"):
            app._process_transcription(SPEECH, 1.0)
        self.assertEqual(app._tray_app.states[-1], AppState.ERROR)
        self.assertIsNotNone(app._failed_capture)

    def test_history_failure_does_not_cost_the_dictation(self):
        app = make_app(FakeTranscriber(results=[ok("à coller")]))
        app._database.save_transcription.side_effect = OSError("database is locked")
        with patch("builtins.print"):
            app._process_transcription(SPEECH, 1.0)  # must not raise
        app._text_inserter.insert_text.assert_called_once_with("à coller")
        self.assertEqual(app._tray_app.states[-1], AppState.IDLE)


class LoadingStateTests(unittest.TestCase):
    def test_engine_switch_is_refused_while_loading(self):
        app = make_app(FakeTranscriber(ready=False, loading=True))
        with patch("builtins.print"), patch("flototext.main.config") as cfg:
            cfg.model.backend = "canary"
            app._on_change_asr_backend("parakeet")
            self.assertEqual(cfg.model.backend, "canary")
        self.assertEqual(app._transcriber.reload_calls, 0)

    def test_model_loaded_during_a_transcription_does_not_turn_green(self):
        app = make_app(FakeTranscriber())
        app._processing = True
        with patch("builtins.print"):
            app._on_model_loaded()
        self.assertNotIn(AppState.IDLE, app._tray_app.states)


if __name__ == "__main__":
    unittest.main()
