"""User notifications, shown as balloons from the tray icon.

win10toast was disabled long ago (WNDPROC errors on modern Windows), which
left every notification a silent no-op: a capture rejected as silent, a failed
model load or a paste that fell back to the clipboard only changed the icon's
colour. pystray's own Icon.notify() has none of win10toast's problems, so the
tray icon now carries them.

Only the notices that report something going wrong are shown. Routine ones
(model loading, a successful dictation, a tap too short to record) stay quiet:
they happen dozens of times a day and the icon colour already says it.
"""

from typing import Callable, Optional

from ..config import config
from ..core.localization import localization


class NotificationManager:
    """Routes user-facing notices to the tray icon."""

    def __init__(self, enabled: bool = None,
                 notifier: Optional[Callable[[str, str], None]] = None):
        """Initialize the notification manager.

        Args:
            enabled: Whether notifications are enabled (default from config).
            notifier: ``notifier(message, title)``; usually ``TrayApp.notify``.
        """
        self.enabled = enabled if enabled is not None else config.ui.show_notifications
        self._notifier = notifier

    def set_notifier(self, notifier: Optional[Callable[[str, str], None]]) -> None:
        """Attach the function that displays a notice (the tray icon's)."""
        self._notifier = notifier

    def _show(self, title: str, message: str) -> None:
        """Display a notice that the user needs to see."""
        if not self.enabled or self._notifier is None:
            return
        try:
            self._notifier(message, title)
        except Exception as e:
            print(f"Error showing notification: {e}")

    def _show_toast_async(self, title: str, message: str, duration: int = 3,
                          threaded: bool = True) -> None:
        """Routine notice: deliberately not displayed (see module docstring)."""

    def notify_ready(self) -> None:
        """Show notification that app is ready."""
        self._show_toast_async(
            title=config.ui.app_name,
            message=localization.get("notifications.app_ready"),
            duration=3
        )

    def notify_model_loading(self) -> None:
        """Show notification that model is loading."""
        self._show_toast_async(
            title=config.ui.app_name,
            message=localization.get("notifications.model_loading"),
            duration=3
        )

    def notify_model_loaded(self) -> None:
        """Show notification that model has loaded."""
        self._show_toast_async(
            title=config.ui.app_name,
            message=localization.get("notifications.model_loaded"),
            duration=3
        )

    def notify_transcription_complete(self, text: str, word_count: int) -> None:
        """Show notification for completed transcription.

        Args:
            text: The transcribed text (will be truncated).
            word_count: Number of words transcribed.
        """
        # Truncate text for notification
        display_text = text[:100] + "..." if len(text) > 100 else text
        words_label = localization.get("notifications.words_count", count=word_count)
        self._show_toast_async(
            title=f"{config.ui.app_name} - {words_label}",
            message=display_text,
            duration=3
        )

    def notify_error(self, error_message: str) -> None:
        """Show error notification.

        Args:
            error_message: The error message to display.
        """
        error_title = localization.get("notifications.error_title")
        self._show(f"{config.ui.app_name} - {error_title}", error_message)

    def notify_recording_too_short(self) -> None:
        """Show notification that recording was too short."""
        self._show_toast_async(
            title=config.ui.app_name,
            message=localization.get("notifications.recording_too_short"),
            duration=3
        )

    def notify_no_audio(self) -> None:
        """Show notification that the microphone captured no sound."""
        self._show(config.ui.app_name, localization.get("notifications.no_audio"))

    def notify_clipboard_only(self, text: str) -> None:
        """Show notification when text was copied to clipboard only.

        Args:
            text: The transcribed text (will be truncated).
        """
        display_text = text[:80] + "..." if len(text) > 80 else text
        clipboard_title = localization.get("notifications.copied_to_clipboard")
        self._show(f"{config.ui.app_name} - {clipboard_title}", display_text)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable notifications.

        Args:
            enabled: Whether to enable notifications.
        """
        self.enabled = enabled
