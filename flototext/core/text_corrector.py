"""Text correction module with custom word dictionary."""

import json
import re
from pathlib import Path
from typing import Dict, Optional

from ..config import config


class TextCorrector:
    """Applies custom word corrections to transcribed text."""

    def __init__(self, dictionary_path: Optional[Path] = None):
        """Initialize the text corrector.

        Args:
            dictionary_path: Path to the custom words JSON file.
        """
        self.dictionary_path = dictionary_path or config.data_dir / "custom_words.json"
        self._corrections: Dict[str, str] = {}
        self._pattern: Optional[re.Pattern] = None
        self._load_dictionary()

    def _load_dictionary(self) -> None:
        """Load corrections from the JSON file."""
        if not self.dictionary_path.exists():
            self._create_default_dictionary()
            return

        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._corrections = data.get('corrections', {})
                self._build_pattern()
                print(f"Loaded {len(self._corrections)} custom word corrections")
        except Exception as e:
            print(f"Error loading custom words dictionary: {e}")
            self._corrections = {}

    def _create_default_dictionary(self) -> None:
        """Create a default dictionary file."""
        default_data = {
            "corrections": {},
            "_comment": "Add your custom word corrections here. Keys are what the ASR outputs, values are the correct spelling."
        }
        try:
            self.dictionary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error creating default dictionary: {e}")

    def _build_pattern(self) -> None:
        """Build regex pattern for efficient replacement."""
        if not self._corrections:
            self._pattern = None
            return

        # Sort by length (longest first) to avoid partial replacements
        sorted_keys = sorted(self._corrections.keys(), key=len, reverse=True)
        # Escape special regex characters and join with |
        escaped = [re.escape(k) for k in sorted_keys]
        # Case-insensitive word boundary matching
        self._pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE
        )

    def correct(self, text: str) -> str:
        """Apply corrections to the text.

        Args:
            text: The transcribed text to correct.

        Returns:
            Corrected text with custom word replacements applied.
        """
        if not text or not self._pattern or not self._corrections:
            return text

        def replace_match(match: re.Match) -> str:
            """Replace matched text preserving case when possible."""
            matched = match.group(0)
            # Find the correction (case-insensitive lookup)
            lower_matched = matched.lower()
            for key, value in self._corrections.items():
                if key.lower() == lower_matched:
                    # Preserve original case pattern if single word
                    if matched.isupper():
                        return value.upper()
                    elif matched[0].isupper() and len(matched) > 1:
                        return value[0].upper() + value[1:] if len(value) > 1 else value.upper()
                    return value
            return matched

        return self._pattern.sub(replace_match, text)

    def reload(self) -> None:
        """Reload the dictionary from file."""
        self._load_dictionary()

    def add_correction(self, wrong: str, correct: str) -> bool:
        """Add a new correction to the dictionary.

        Args:
            wrong: The incorrectly transcribed word/phrase.
            correct: The correct spelling.

        Returns:
            True if added successfully.
        """
        try:
            self._corrections[wrong.lower()] = correct
            self._save_dictionary()
            self._build_pattern()
            return True
        except Exception as e:
            print(f"Error adding correction: {e}")
            return False

    def remove_correction(self, wrong: str) -> bool:
        """Remove a correction from the dictionary.

        Args:
            wrong: The key to remove.

        Returns:
            True if removed successfully.
        """
        try:
            if wrong.lower() in self._corrections:
                del self._corrections[wrong.lower()]
                self._save_dictionary()
                self._build_pattern()
                return True
            return False
        except Exception as e:
            print(f"Error removing correction: {e}")
            return False

    def _save_dictionary(self) -> None:
        """Save corrections to the JSON file."""
        data = {
            "corrections": self._corrections,
            "_comment": "Add your custom word corrections here. Keys are what the ASR outputs, values are the correct spelling."
        }
        with open(self.dictionary_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_corrections(self) -> Dict[str, str]:
        """Get all current corrections.

        Returns:
            Dictionary of corrections.
        """
        return self._corrections.copy()

    @property
    def dictionary_file(self) -> Path:
        """Get the path to the dictionary file."""
        return self.dictionary_path
