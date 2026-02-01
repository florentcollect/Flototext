"""Main entry point for Flototext application."""

import os
import sys
import time
import threading
import signal
from datetime import datetime

from .config import config
from .core.hotkey_manager import HotkeyManager
from .core.audio_recorder import AudioRecorder
from .core.transcriber import Transcriber
from .core.text_inserter import TextInserter
from .core.text_corrector import TextCorrector
from .core.audio_muter import AudioMuter
from .storage.database import Database
from .storage.models import Transcription
from .ui.tray_app import TrayApp, AppState
from .ui.notifications import NotificationManager
from .ui.sounds import SoundManager


class FlototextApp:
    """Main Flototext application orchestrator."""

    def __init__(self):
        """Initialize the application."""
        # Ensure directories exist
        config.ensure_directories()

        # Initialize components
        self._database = Database(config.database_path)
        self._sound_manager = SoundManager()
        self._notification_manager = NotificationManager()
        self._text_inserter = TextInserter()
        self._text_corrector = TextCorrector()
        self._audio_muter = AudioMuter(enabled=config.ui.mute_during_recording)

        # Initialize transcriber with callbacks
        self._transcriber = Transcriber(
            on_model_loaded=self._on_model_loaded,
            on_error=self._on_transcription_error
        )

        # Initialize audio recorder with callbacks
        self._audio_recorder = AudioRecorder(
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop
        )

        # Initialize tray app with callbacks
        self._tray_app = TrayApp(
            on_quit=self._on_quit,
            on_toggle_sounds=self._on_toggle_sounds,
            on_toggle_notifications=self._on_toggle_notifications,
            on_toggle_mute=self._on_toggle_mute,
            on_copy_last=self._on_copy_last,
            on_edit_dictionary=self._on_edit_dictionary
        )

        # Initialize hotkey manager with callbacks
        self._hotkey_manager = HotkeyManager(
            on_key_press=self._on_hotkey_press,
            on_key_release=self._on_hotkey_release
        )

        # State
        self._running = False
        self._processing = False
        self._shutdown_event = threading.Event()

    def _on_model_loaded(self) -> None:
        """Handle model loaded event."""
        print("Model loaded successfully")
        self._tray_app.set_state(AppState.IDLE)
        self._sound_manager.play_ready()
        self._notification_manager.notify_model_loaded()

    def _on_transcription_error(self, error: str) -> None:
        """Handle transcription error.

        Args:
            error: Error message.
        """
        print(f"Transcription error: {error}")
        self._tray_app.set_state(AppState.ERROR)
        self._sound_manager.play_error()
        self._notification_manager.notify_error(error)

        # Return to idle state after a moment
        def reset_state():
            time.sleep(2)
            if self._running and not self._audio_recorder.is_recording:
                self._tray_app.set_state(AppState.IDLE)

        threading.Thread(target=reset_state, daemon=True).start()

    def _on_recording_start(self) -> None:
        """Handle recording start event."""
        print("Recording started")

    def _on_recording_stop(self) -> None:
        """Handle recording stop event."""
        print("Recording stopped")

    def _on_hotkey_press(self) -> None:
        """Handle hotkey press (start recording)."""
        if not self._transcriber.is_ready:
            print("Model not ready yet")
            self._sound_manager.play_error()
            return

        if self._processing:
            print("Already processing a transcription")
            return

        if self._audio_recorder.start_recording():
            self._tray_app.set_state(AppState.RECORDING)
            self._sound_manager.play_start_recording()
            # Mute system audio to prevent interference
            self._audio_muter.mute()

    def _on_hotkey_release(self) -> None:
        """Handle hotkey release (stop recording and transcribe)."""
        if not self._audio_recorder.is_recording:
            return

        result = self._audio_recorder.stop_recording()
        # Restore system audio
        self._audio_muter.unmute()
        self._sound_manager.play_stop_recording()

        if result is None:
            self._on_transcription_error("Failed to stop recording")
            return

        if not result.is_valid:
            print(f"Recording too short: {result.duration:.2f}s")
            self._tray_app.set_state(AppState.IDLE)
            self._notification_manager.notify_recording_too_short()
            return

        # Process transcription in background
        self._processing = True
        self._tray_app.set_state(AppState.PROCESSING)

        def process():
            try:
                self._process_transcription(result.audio_data, result.duration)
            finally:
                self._processing = False

        threading.Thread(target=process, daemon=True).start()

    def _process_transcription(self, audio_data, duration: float) -> None:
        """Process audio transcription.

        Args:
            audio_data: Audio data as numpy array.
            duration: Recording duration in seconds.
        """
        # Transcribe
        result = self._transcriber.transcribe(audio_data, config.audio.sample_rate)

        if not result.success:
            self._on_transcription_error(result.error or "Unknown error")
            return

        if not result.text.strip():
            print("Empty transcription")
            self._tray_app.set_state(AppState.IDLE)
            return

        # Apply custom word corrections
        text = self._text_corrector.correct(result.text.strip())
        word_count = len(text.split())

        # Save to database
        transcription = Transcription(
            text=text,
            language=result.language,
            duration_seconds=duration,
            created_at=datetime.now(),
            word_count=word_count
        )
        self._database.save_transcription(transcription)

        # Insert text at cursor position
        if self._text_inserter.insert_text(text):
            print(f"Transcribed and pasted: {text[:50]}...")
            self._tray_app.increment_transcription_count()
            self._sound_manager.play_success()
            self._notification_manager.notify_transcription_complete(text, word_count)
        else:
            # Fallback: just copy to clipboard
            self._text_inserter.copy_to_clipboard(text)
            print(f"Transcribed and copied to clipboard: {text[:50]}...")
            self._sound_manager.play_success()
            self._notification_manager.notify_clipboard_only(text)

        self._tray_app.set_state(AppState.IDLE)

    def _on_toggle_sounds(self, enabled: bool) -> None:
        """Handle sound toggle.

        Args:
            enabled: Whether sounds are enabled.
        """
        self._sound_manager.set_enabled(enabled)
        print(f"Sounds {'enabled' if enabled else 'disabled'}")

    def _on_toggle_notifications(self, enabled: bool) -> None:
        """Handle notification toggle.

        Args:
            enabled: Whether notifications are enabled.
        """
        self._notification_manager.set_enabled(enabled)
        print(f"Notifications {'enabled' if enabled else 'disabled'}")

    def _on_toggle_mute(self, enabled: bool) -> None:
        """Handle mute during recording toggle.

        Args:
            enabled: Whether muting is enabled.
        """
        self._audio_muter.set_enabled(enabled)
        print(f"Mute during recording {'enabled' if enabled else 'disabled'}")

    def _on_copy_last(self) -> None:
        """Copy last transcription to clipboard."""
        last = self._database.get_last_transcription()
        if last and last.text:
            self._text_inserter.copy_to_clipboard(last.text)
            self._notification_manager.notify_clipboard_only(last.text)
            print(f"Copied to clipboard: {last.text[:50]}...")
        else:
            self._notification_manager.notify_error("Aucune transcription disponible")
            print("No transcription available")

    def _on_edit_dictionary(self) -> None:
        """Open the custom words dictionary file for editing."""
        dictionary_path = self._text_corrector.dictionary_file
        print(f"Opening dictionary: {dictionary_path}")
        try:
            os.startfile(str(dictionary_path))
        except Exception as e:
            print(f"Error opening dictionary: {e}")
            self._notification_manager.notify_error(f"Cannot open dictionary: {e}")

    def _on_quit(self) -> None:
        """Handle quit request."""
        print("Quit requested")
        self.stop()

    def _signal_handler(self, signum, frame) -> None:
        """Handle system signals.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop()

    def start(self) -> None:
        """Start the application."""
        print(f"Starting {config.ui.app_name}...")
        print(f"Press {config.hotkey.trigger_key.upper()} to record")

        self._running = True

        # Clean up old transcriptions (keep only last 7 days)
        deleted = self._database.delete_old_transcriptions(days=7)
        if deleted > 0:
            print(f"Cleaned up {deleted} old transcription(s)")

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start tray app
        self._tray_app.start()
        self._tray_app.set_state(AppState.LOADING)

        # Start hotkey listener
        self._hotkey_manager.start()

        # Load model in background
        print("Loading ASR model in background...")
        self._notification_manager.notify_model_loading()
        self._transcriber.load_model_async()

        # Wait for shutdown
        try:
            while self._running and not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            pass

        self._cleanup()

    def stop(self) -> None:
        """Stop the application."""
        self._running = False
        self._shutdown_event.set()

    def _cleanup(self) -> None:
        """Clean up resources."""
        print("Cleaning up...")

        self._hotkey_manager.stop()
        self._audio_recorder.cleanup()
        self._transcriber.cleanup()
        self._database.close()
        self._tray_app.stop()

        print("Goodbye!")


def main():
    """Main entry point."""
    app = FlototextApp()
    app.start()


if __name__ == "__main__":
    main()
