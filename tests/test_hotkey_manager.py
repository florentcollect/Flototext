import threading
import time
import unittest

from flototext.core.hotkey_manager import HotkeyManager


class WatchdogTest(unittest.TestCase):
    """The watchdog repairs the two faults that both strand the push-to-talk key.

    Windows can drop pynput's release event (the flag stays pressed and every
    later press is swallowed), and it can unhook a slow listener without killing
    its thread (no event ever arrives again). The log cannot tell them apart, so
    both must be recovered from.
    """

    def _manager(self, key_down):
        self.presses = []
        self.releases = []
        self.rebuilds = 0

        manager = HotkeyManager(
            on_key_press=lambda: self.presses.append(1),
            on_key_release=lambda: self.releases.append(1),
        )
        manager._WATCHDOG_INTERVAL = 0.02
        manager._key_physically_down = key_down
        # Never install a real keyboard hook from a test.
        def fake_restart():
            self.rebuilds += 1
        manager._restart_listener = fake_restart
        return manager

    def _run_watchdog(self, manager, seconds=0.3):
        manager._watchdog_stop.clear()
        thread = threading.Thread(target=manager._watchdog_loop, daemon=True)
        thread.start()
        time.sleep(seconds)
        manager._watchdog_stop.set()
        thread.join(timeout=1)

    def test_dropped_release_is_recovered(self):
        manager = self._manager(lambda: False)
        manager._is_pressed = True

        self._run_watchdog(manager)

        self.assertEqual(self.releases, [1], "release callback must fire exactly once")
        self.assertFalse(manager._is_pressed)
        self.assertEqual(self.rebuilds, 0, "a dropped release must not rebuild the listener")

    def test_deaf_listener_is_rebuilt_and_press_honoured(self):
        manager = self._manager(lambda: True)

        self._run_watchdog(manager)

        self.assertEqual(self.rebuilds, 1, "listener must be rebuilt exactly once, not on every poll")
        self.assertEqual(self.presses, [1], "the held key must produce exactly one press")
        self.assertTrue(manager._is_pressed)

    def test_idle_key_triggers_nothing(self):
        manager = self._manager(lambda: False)

        self._run_watchdog(manager)

        self.assertEqual((self.presses, self.releases, self.rebuilds), ([], [], 0))

    def test_disabled_manager_is_inert(self):
        manager = self._manager(lambda: True)
        manager.disable()

        self._run_watchdog(manager)

        self.assertEqual(self.presses, [])
        self.assertEqual(self.rebuilds, 0)

    def test_press_and_release_are_idempotent(self):
        manager = self._manager(lambda: False)

        self.assertTrue(manager._press())
        self.assertFalse(manager._press(), "a second press without a release is a no-op")
        self.assertTrue(manager._release())
        self.assertFalse(manager._release(), "a late real release after recovery is a no-op")

        self.assertEqual(self.presses, [1])
        self.assertEqual(self.releases, [1])

    def test_f2_maps_to_its_virtual_key_code(self):
        manager = self._manager(lambda: False)
        self.assertEqual(manager._parse_vk("f2"), 0x71)
        self.assertEqual(manager._parse_vk("f1"), 0x70)
        self.assertIsNone(manager._parse_vk("f99"), "an unpollable key disables the watchdog")


if __name__ == "__main__":
    unittest.main()
