"""Data models for Flototext."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Transcription:
    """Represents a transcription record."""
    id: Optional[int] = None
    text: str = ""
    language: str = "French"
    duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    word_count: int = 0

    def __post_init__(self):
        """Calculate word count if not provided."""
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())

    @classmethod
    def from_row(cls, row: tuple) -> "Transcription":
        """Create a Transcription from a database row."""
        return cls(
            id=row[0],
            text=row[1],
            language=row[2],
            duration_seconds=row[3],
            created_at=datetime.fromisoformat(row[4]) if isinstance(row[4], str) else row[4],
            word_count=row[5]
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat(),
            "word_count": self.word_count
        }
