"""Localization module for Flototext application."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from ..config import config


class Localization:
    """Manages application localization and translations."""

    _instance: Optional['Localization'] = None

    def __new__(cls) -> 'Localization':
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the localization manager."""
        if self._initialized:
            return

        self._locales_dir = config.data_dir / "locales"
        self._current_language = config.ui.language
        self._translations: Dict[str, Any] = {}
        self._fallback_translations: Dict[str, Any] = {}
        self._on_language_changed: List[Callable[[str], None]] = []

        self._load_translations()
        self._initialized = True

    @property
    def current_language(self) -> str:
        """Get the current language code."""
        return self._current_language

    @property
    def asr_language(self) -> str:
        """Get the ASR language name for the current language."""
        return self._translations.get("asr_language", "English")

    @property
    def language_name(self) -> str:
        """Get the display name of the current language."""
        return self._translations.get("language_name", self._current_language)

    def get_available_languages(self) -> List[Dict[str, str]]:
        """Get list of available languages with their codes and names.

        Returns:
            List of dicts with 'code' and 'name' keys.
        """
        languages = []
        if self._locales_dir.exists():
            for locale_file in self._locales_dir.glob("*.json"):
                code = locale_file.stem
                try:
                    with open(locale_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        name = data.get("language_name", code)
                        languages.append({"code": code, "name": name})
                except (json.JSONDecodeError, IOError):
                    languages.append({"code": code, "name": code})
        return sorted(languages, key=lambda x: x["name"])

    def set_language(self, language_code: str) -> bool:
        """Change the current language.

        Args:
            language_code: The language code to switch to (e.g., 'fr', 'en').

        Returns:
            True if language was changed successfully, False otherwise.
        """
        locale_file = self._locales_dir / f"{language_code}.json"
        if not locale_file.exists():
            print(f"Language file not found: {locale_file}")
            return False

        self._current_language = language_code
        config.ui.language = language_code
        self._load_translations()

        # Notify listeners
        for callback in self._on_language_changed:
            try:
                callback(language_code)
            except Exception as e:
                print(f"Error in language change callback: {e}")

        return True

    def on_language_changed(self, callback: Callable[[str], None]) -> None:
        """Register a callback for language changes.

        Args:
            callback: Function to call when language changes, receives new language code.
        """
        if callback not in self._on_language_changed:
            self._on_language_changed.append(callback)

    def get(self, key: str, **kwargs) -> str:
        """Get a translated string by key with optional interpolation.

        Args:
            key: Dot-separated key path (e.g., 'menu.quit', 'notifications.app_ready').
            **kwargs: Values to interpolate into the string.

        Returns:
            The translated string, or the key if not found.
        """
        # Navigate the nested dict using the key path
        value = self._get_nested(self._translations, key)

        # Fallback to English if not found
        if value is None:
            value = self._get_nested(self._fallback_translations, key)

        # Return key if still not found
        if value is None:
            return key

        # Interpolate kwargs
        if kwargs:
            try:
                return value.format(**kwargs)
            except KeyError:
                return value

        return value

    def _get_nested(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """Get a nested value from a dict using dot notation.

        Args:
            data: The dictionary to search.
            key: Dot-separated key path.

        Returns:
            The value if found, None otherwise.
        """
        keys = key.split('.')
        current = data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None

        return current if isinstance(current, str) else None

    def _load_translations(self) -> None:
        """Load translations for the current language."""
        # Load current language
        locale_file = self._locales_dir / f"{self._current_language}.json"
        if locale_file.exists():
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    self._translations = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading translations: {e}")
                self._translations = {}
        else:
            print(f"Locale file not found: {locale_file}")
            self._translations = {}

        # Load English as fallback
        if self._current_language != "en":
            fallback_file = self._locales_dir / "en.json"
            if fallback_file.exists():
                try:
                    with open(fallback_file, 'r', encoding='utf-8') as f:
                        self._fallback_translations = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self._fallback_translations = {}
            else:
                self._fallback_translations = {}
        else:
            self._fallback_translations = {}


# Global localization instance
localization = Localization()
