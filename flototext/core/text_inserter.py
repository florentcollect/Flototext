"""Text insertion module using clipboard and keyboard simulation."""

import time
import pyperclip
import pyautogui
from typing import Optional


class TextInserter:
    """Inserts text at the current cursor position."""

    def __init__(self):
        """Initialize the text inserter."""
        # Configure pyautogui for safety
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05

        self._previous_clipboard: Optional[str] = None

    def insert_text(self, text: str, restore_clipboard: bool = True) -> bool:
        """Insert text at the current cursor position.

        Uses clipboard copy + Ctrl+V paste method for reliability.

        Args:
            text: The text to insert.
            restore_clipboard: Whether to restore previous clipboard content.

        Returns:
            True if text was inserted successfully.
        """
        if not text:
            return False

        try:
            # Save current clipboard content
            if restore_clipboard:
                try:
                    self._previous_clipboard = pyperclip.paste()
                except Exception:
                    self._previous_clipboard = None

            # Copy text to clipboard
            pyperclip.copy(text)

            # Small delay to ensure clipboard is ready
            time.sleep(0.05)

            # Simulate Ctrl+V to paste
            pyautogui.hotkey('ctrl', 'v')

            # Small delay to ensure paste completes
            time.sleep(0.1)

            # Restore previous clipboard content
            if restore_clipboard and self._previous_clipboard is not None:
                time.sleep(0.1)
                try:
                    pyperclip.copy(self._previous_clipboard)
                except Exception:
                    pass

            return True

        except Exception as e:
            print(f"Error inserting text: {e}")
            return False

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to clipboard without pasting.

        Args:
            text: The text to copy.

        Returns:
            True if text was copied successfully.
        """
        if not text:
            return False

        try:
            pyperclip.copy(text)
            return True
        except Exception as e:
            print(f"Error copying to clipboard: {e}")
            return False

    def get_clipboard_content(self) -> Optional[str]:
        """Get current clipboard content.

        Returns:
            Clipboard content or None if error.
        """
        try:
            return pyperclip.paste()
        except Exception:
            return None
