"""Windows toast notifications module."""

import threading
from typing import Optional

from ..config import config
from ..core.localization import localization

# Disable win10toast due to WNDPROC errors on modern Windows
# Use simple print notifications instead
HAS_TOAST = False
ToastNotifier = None


class NotificationManager:
    """Manages Windows toast notifications."""

    def __init__(self, enabled: bool = None):
        """Initialize the notification manager.

        Args:
            enabled: Whether notifications are enabled (default from config).
        """
        self.enabled = enabled if enabled is not None else config.ui.show_notifications
        self._toaster: Optional[ToastNotifier] = None

        if HAS_TOAST and self.enabled:
            try:
                self._toaster = ToastNotifier()
            except Exception as e:
                print(f"Failed to initialize toast notifier: {e}")
                self._toaster = None

    def _show_toast_async(
        self,
        title: str,
        message: str,
        duration: int = 3,
        threaded: bool = True
    ) -> None:
        """Show a toast notification asynchronously.

        Args:
            title: Notification title.
            message: Notification message.
            duration: Duration in seconds.
            threaded: Whether to run in a separate thread.
        """
        if not self.enabled or not self._toaster:
            return

        def show():
            try:
                self._toaster.show_toast(
                    title=title,
                    msg=message,
                    duration=duration,
                    threaded=False  # Already in a thread
                )
            except Exception as e:
                print(f"Error showing notification: {e}")

        if threaded:
            thread = threading.Thread(target=show, daemon=True)
            thread.start()
        else:
            show()

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
        self._show_toast_async(
            title=f"{config.ui.app_name} - {error_title}",
            message=error_message,
            duration=5
        )

    def notify_recording_too_short(self) -> None:
        """Show notification that recording was too short."""
        self._show_toast_async(
            title=config.ui.app_name,
            message=localization.get("notifications.recording_too_short"),
            duration=3
        )

    def notify_clipboard_only(self, text: str) -> None:
        """Show notification when text was copied to clipboard only.

        Args:
            text: The transcribed text (will be truncated).
        """
        display_text = text[:80] + "..." if len(text) > 80 else text
        clipboard_title = localization.get("notifications.copied_to_clipboard")
        self._show_toast_async(
            title=f"{config.ui.app_name} - {clipboard_title}",
            message=display_text,
            duration=3
        )

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable notifications.

        Args:
            enabled: Whether to enable notifications.
        """
        self.enabled = enabled
