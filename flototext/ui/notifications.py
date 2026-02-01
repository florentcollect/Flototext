"""Windows toast notifications module."""

import threading
from typing import Optional

from ..config import config

# Try to import win10toast, provide fallback if not available
try:
    from win10toast import ToastNotifier
    HAS_TOAST = True
except ImportError:
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
            message="Application ready. Press F2 to record.",
            duration=3
        )

    def notify_model_loading(self) -> None:
        """Show notification that model is loading."""
        self._show_toast_async(
            title=config.ui.app_name,
            message="Loading speech recognition model...",
            duration=3
        )

    def notify_model_loaded(self) -> None:
        """Show notification that model has loaded."""
        self._show_toast_async(
            title=config.ui.app_name,
            message="Model loaded. Ready to transcribe!",
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
        self._show_toast_async(
            title=f"{config.ui.app_name} - {word_count} words",
            message=display_text,
            duration=3
        )

    def notify_error(self, error_message: str) -> None:
        """Show error notification.

        Args:
            error_message: The error message to display.
        """
        self._show_toast_async(
            title=f"{config.ui.app_name} - Error",
            message=error_message,
            duration=5
        )

    def notify_recording_too_short(self) -> None:
        """Show notification that recording was too short."""
        self._show_toast_async(
            title=config.ui.app_name,
            message="Recording too short. Please speak longer.",
            duration=3
        )

    def notify_clipboard_only(self, text: str) -> None:
        """Show notification when text was copied to clipboard only.

        Args:
            text: The transcribed text (will be truncated).
        """
        display_text = text[:80] + "..." if len(text) > 80 else text
        self._show_toast_async(
            title=f"{config.ui.app_name} - Copied to Clipboard",
            message=display_text,
            duration=3
        )

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable notifications.

        Args:
            enabled: Whether to enable notifications.
        """
        self.enabled = enabled
