"""TrayApp must survive being driven from several threads at once."""

import threading
import time
import unittest
from unittest.mock import patch

from flototext.ui.tray_app import AppState, TrayApp


class RacyIcon:
    """Stands in for pystray.Icon, whose setters are not thread-safe.

    Each setter holds a short critical section; two threads inside at once is
    exactly what made pystray destroy one icon handle twice.
    """

    HAS_NOTIFICATION = True

    def __init__(self):
        self._inside = 0
        self.overlaps = 0
        self.notified = []
        self.fail_with = None

    def _enter(self):
        if self.fail_with:
            raise self.fail_with
        self._inside += 1
        if self._inside > 1:
            self.overlaps += 1
        time.sleep(0.005)
        self._inside -= 1

    def __setattr__(self, name, value):
        if name in ("icon", "title", "menu"):
            self._enter()
        object.__setattr__(self, name, value)

    def notify(self, message, title):
        self.notified.append((title, message))


class TrayThreadSafetyTests(unittest.TestCase):
    def _tray(self):
        tray = TrayApp()
        tray._icon = RacyIcon()
        return tray

    def test_concurrent_state_changes_never_overlap(self):
        tray = self._tray()
        states = [AppState.IDLE, AppState.PROCESSING, AppState.LOADING, AppState.ERROR]
        threads = [threading.Thread(target=lambda s=s: [tray.set_state(s) for _ in range(5)])
                   for s in states]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(tray._icon.overlaps, 0)

    def test_a_failing_icon_update_does_not_raise(self):
        tray = self._tray()
        tray._icon.fail_with = OSError(1402, "Invalid cursor handle")
        with patch("builtins.print"):
            tray.set_state(AppState.PROCESSING)  # must not raise
        self.assertEqual(tray._state, AppState.PROCESSING)

    def test_notify_goes_through_the_icon(self):
        tray = self._tray()
        tray.notify("Aucun son capté", "Flototext")
        self.assertEqual(tray._icon.notified, [("Flototext", "Aucun son capté")])


if __name__ == "__main__":
    unittest.main()
