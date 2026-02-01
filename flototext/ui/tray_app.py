"""System tray application using pystray."""

import os
import threading
from typing import Callable, Optional
from enum import Enum
from PIL import Image, ImageDraw

try:
    import pystray
    from pystray import MenuItem, Menu
    HAS_PYSTRAY = True
except ImportError:
    HAS_PYSTRAY = False
    pystray = None

from ..config import config
from ..core.localization import localization


class AppState(Enum):
    """Application state for tray icon."""
    LOADING = "loading"
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


class TrayApp:
    """System tray application with status icon."""

    # Color definitions for different states
    COLORS = {
        AppState.LOADING: (255, 165, 0),    # Orange
        AppState.IDLE: (0, 200, 0),          # Green
        AppState.RECORDING: (255, 0, 0),     # Red
        AppState.PROCESSING: (255, 255, 0),  # Yellow
        AppState.ERROR: (128, 128, 128),     # Gray
    }

    # State to localization key mapping
    STATE_KEYS = {
        AppState.LOADING: "loading",
        AppState.IDLE: "ready",
        AppState.RECORDING: "recording",
        AppState.PROCESSING: "processing",
        AppState.ERROR: "error",
    }

    def __init__(
        self,
        on_quit: Optional[Callable] = None,
        on_toggle_sounds: Optional[Callable[[bool], None]] = None,
        on_toggle_notifications: Optional[Callable[[bool], None]] = None,
        on_toggle_mute: Optional[Callable[[bool], None]] = None,
        on_copy_last: Optional[Callable] = None,
        on_edit_dictionary: Optional[Callable] = None,
        on_change_language: Optional[Callable[[str], None]] = None
    ):
        """Initialize the tray application.

        Args:
            on_quit: Callback when quit is selected.
            on_toggle_sounds: Callback when sounds are toggled.
            on_toggle_notifications: Callback when notifications are toggled.
            on_toggle_mute: Callback when mute during recording is toggled.
            on_copy_last: Callback to copy last transcription to clipboard.
            on_edit_dictionary: Callback to edit custom words dictionary.
            on_change_language: Callback when language is changed.
        """
        self.on_quit = on_quit
        self.on_toggle_sounds = on_toggle_sounds
        self.on_toggle_notifications = on_toggle_notifications
        self.on_toggle_mute = on_toggle_mute
        self.on_copy_last = on_copy_last
        self.on_edit_dictionary = on_edit_dictionary
        self.on_change_language = on_change_language

        self._icon: Optional[pystray.Icon] = None
        self._state = AppState.LOADING
        self._sounds_enabled = config.ui.play_sounds
        self._notifications_enabled = config.ui.show_notifications
        self._mute_enabled = config.ui.mute_during_recording
        self._transcription_count = 0

        if not HAS_PYSTRAY:
            print("Warning: pystray not available, tray icon disabled")

    def _create_icon_image(self, state: AppState, size: int = 64) -> Image.Image:
        """Create an icon image for the given state.

        Args:
            state: Current application state.
            size: Icon size in pixels.

        Returns:
            PIL Image for the icon.
        """
        # Create a new image with transparency
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Get color for current state
        color = self.COLORS.get(state, self.COLORS[AppState.IDLE])

        # Draw a filled circle
        margin = size // 8
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=color,
            outline=(255, 255, 255)
        )

        # Add microphone shape in center
        mic_color = (255, 255, 255) if state != AppState.PROCESSING else (0, 0, 0)
        center_x, center_y = size // 2, size // 2
        mic_width = size // 6
        mic_height = size // 3

        # Microphone body (rectangle with rounded top)
        draw.rectangle(
            [center_x - mic_width, center_y - mic_height // 2,
             center_x + mic_width, center_y + mic_height // 3],
            fill=mic_color
        )

        # Microphone stand (small rectangle at bottom)
        stand_width = mic_width // 2
        draw.rectangle(
            [center_x - stand_width, center_y + mic_height // 3,
             center_x + stand_width, center_y + mic_height // 2],
            fill=mic_color
        )

        return image

    def _get_tooltip(self) -> str:
        """Get tooltip text based on current state."""
        state_key = self.STATE_KEYS.get(self._state, "ready")
        state_text = localization.get(f"tooltip.{state_key}")
        base_text = f"{config.ui.app_name} - {state_text}"

        if self._transcription_count > 0:
            base_text += f"\n{localization.get('menu.transcriptions', count=self._transcription_count)}"

        return base_text

    def _get_state_display(self) -> str:
        """Get localized state display text."""
        state_key = self.STATE_KEYS.get(self._state, "ready")
        return localization.get(f"states.{state_key}")

    def _create_language_menu(self) -> Menu:
        """Create the language submenu."""
        languages = localization.get_available_languages()
        items = []
        for lang in languages:
            code = lang["code"]
            name = lang["name"]
            items.append(
                MenuItem(
                    name,
                    lambda _, c=code: self._change_language(c),
                    checked=lambda item, c=code: localization.current_language == c
                )
            )
        return Menu(*items)

    def _create_menu(self) -> Menu:
        """Create the context menu for the tray icon."""
        return Menu(
            MenuItem(
                localization.get("menu.status", state=self._get_state_display()),
                lambda: None,
                enabled=False
            ),
            MenuItem(
                localization.get("menu.transcriptions", count=self._transcription_count),
                lambda: None,
                enabled=False
            ),
            Menu.SEPARATOR,
            MenuItem(
                localization.get("menu.copy_last"),
                self._copy_last
            ),
            MenuItem(
                localization.get("menu.edit_dictionary"),
                self._edit_dictionary
            ),
            Menu.SEPARATOR,
            MenuItem(
                localization.get("menu.sounds"),
                self._toggle_sounds,
                checked=lambda item: self._sounds_enabled
            ),
            MenuItem(
                localization.get("menu.notifications"),
                self._toggle_notifications,
                checked=lambda item: self._notifications_enabled
            ),
            MenuItem(
                localization.get("menu.mute_recording"),
                self._toggle_mute,
                checked=lambda item: self._mute_enabled
            ),
            Menu.SEPARATOR,
            MenuItem(
                localization.get("menu.language"),
                self._create_language_menu()
            ),
            Menu.SEPARATOR,
            MenuItem(
                localization.get("menu.quit"),
                self._quit
            )
        )

    def _copy_last(self) -> None:
        """Copy last transcription to clipboard."""
        if self.on_copy_last:
            self.on_copy_last()

    def _edit_dictionary(self) -> None:
        """Open the custom words dictionary for editing."""
        if self.on_edit_dictionary:
            self.on_edit_dictionary()

    def _toggle_sounds(self) -> None:
        """Toggle sound feedback."""
        self._sounds_enabled = not self._sounds_enabled
        if self.on_toggle_sounds:
            self.on_toggle_sounds(self._sounds_enabled)
        self._update_menu()

    def _toggle_notifications(self) -> None:
        """Toggle notifications."""
        self._notifications_enabled = not self._notifications_enabled
        if self.on_toggle_notifications:
            self.on_toggle_notifications(self._notifications_enabled)
        self._update_menu()

    def _toggle_mute(self) -> None:
        """Toggle mute during recording."""
        self._mute_enabled = not self._mute_enabled
        if self.on_toggle_mute:
            self.on_toggle_mute(self._mute_enabled)
        self._update_menu()

    def _change_language(self, language_code: str) -> None:
        """Change the application language."""
        if localization.set_language(language_code):
            if self.on_change_language:
                self.on_change_language(language_code)
            self._update_menu()
            # Update tooltip
            if self._icon:
                self._icon.title = self._get_tooltip()

    def _quit(self) -> None:
        """Handle quit action."""
        if self.on_quit:
            self.on_quit()
        self.stop()

    def _update_menu(self) -> None:
        """Update the tray icon menu."""
        if self._icon:
            self._icon.menu = self._create_menu()

    def set_state(self, state: AppState) -> None:
        """Set the application state and update icon.

        Args:
            state: New application state.
        """
        self._state = state
        if self._icon:
            self._icon.icon = self._create_icon_image(state)
            self._icon.title = self._get_tooltip()
            self._update_menu()

    def increment_transcription_count(self) -> None:
        """Increment the transcription counter."""
        self._transcription_count += 1
        self._update_menu()

    def refresh_ui(self) -> None:
        """Refresh the UI after language change."""
        if self._icon:
            self._icon.title = self._get_tooltip()
            self._update_menu()

    def start(self) -> None:
        """Start the tray application in a background thread."""
        if not HAS_PYSTRAY:
            return

        def run():
            self._icon = pystray.Icon(
                name=config.ui.app_name,
                icon=self._create_icon_image(self._state),
                title=self._get_tooltip(),
                menu=self._create_menu()
            )
            self._icon.run()

        thread = threading.Thread(target=run, daemon=False)
        thread.start()

    def stop(self) -> None:
        """Stop the tray application."""
        if self._icon:
            self._icon.stop()
            self._icon = None
