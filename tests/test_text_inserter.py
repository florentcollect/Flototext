"""Pasting must not leave the clipboard emptier than it found it."""

import unittest
from unittest.mock import patch

from flototext.core.text_inserter import TextInserter


class ClipboardRestoreTests(unittest.TestCase):
    def _insert(self, previous):
        copies = []
        with patch("flototext.core.text_inserter.pyperclip.paste", return_value=previous), \
                patch("flototext.core.text_inserter.pyperclip.copy", side_effect=copies.append), \
                patch("flototext.core.text_inserter.pyautogui.hotkey"), \
                patch("flototext.core.text_inserter.time.sleep"):
            self.assertTrue(TextInserter().insert_text("texte dicté"))
        return copies

    def test_text_clipboard_is_restored(self):
        self.assertEqual(self._insert("ancien texte"), ["texte dicté", "ancien texte"])

    def test_non_text_clipboard_is_not_replaced_by_nothing(self):
        # An image reads as "": restoring it used to empty the clipboard.
        self.assertEqual(self._insert(""), ["texte dicté"])


if __name__ == "__main__":
    unittest.main()
