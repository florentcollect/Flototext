"""Failure notices must reach the user; routine ones must not spam."""

import unittest

from flototext.ui.notifications import NotificationManager


class NotificationTests(unittest.TestCase):
    def _manager(self, enabled=True):
        shown = []
        manager = NotificationManager(enabled=enabled,
                                      notifier=lambda message, title: shown.append(message))
        return manager, shown

    def test_failures_are_shown(self):
        manager, shown = self._manager()
        manager.notify_no_audio()
        manager.notify_error("modèle absent")
        manager.notify_clipboard_only("texte collé")
        self.assertEqual(len(shown), 3)
        self.assertIn("modèle absent", shown)

    def test_routine_notices_stay_quiet(self):
        manager, shown = self._manager()
        manager.notify_model_loading()
        manager.notify_model_loaded()
        manager.notify_transcription_complete("bonjour", 1)
        manager.notify_recording_too_short()
        self.assertEqual(shown, [])

    def test_disabled_shows_nothing(self):
        manager, shown = self._manager(enabled=False)
        manager.notify_no_audio()
        self.assertEqual(shown, [])


if __name__ == "__main__":
    unittest.main()
