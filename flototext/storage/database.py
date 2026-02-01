"""SQLite database operations for Flototext."""

import sqlite3
import threading
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from .models import Transcription


class Database:
    """SQLite database manager for transcriptions."""

    def __init__(self, db_path: Path):
        """Initialize database connection.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _cursor(self):
        """Context manager for database cursor."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_database(self) -> None:
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    language VARCHAR(50) DEFAULT 'French',
                    duration_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    word_count INTEGER
                )
            """)

            # Create index for faster queries by date
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at
                ON transcriptions(created_at DESC)
            """)

    def save_transcription(self, transcription: Transcription) -> int:
        """Save a transcription to the database.

        Args:
            transcription: The transcription to save.

        Returns:
            The ID of the saved transcription.
        """
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO transcriptions (text, language, duration_seconds, created_at, word_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                transcription.text,
                transcription.language,
                transcription.duration_seconds,
                transcription.created_at.isoformat(),
                transcription.word_count
            ))
            return cursor.lastrowid

    def get_transcription(self, transcription_id: int) -> Optional[Transcription]:
        """Get a transcription by ID.

        Args:
            transcription_id: The ID of the transcription.

        Returns:
            The transcription or None if not found.
        """
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT id, text, language, duration_seconds, created_at, word_count "
                "FROM transcriptions WHERE id = ?",
                (transcription_id,)
            )
            row = cursor.fetchone()
            if row:
                return Transcription.from_row(tuple(row))
            return None

    def get_recent_transcriptions(self, limit: int = 10) -> List[Transcription]:
        """Get recent transcriptions.

        Args:
            limit: Maximum number of transcriptions to return.

        Returns:
            List of recent transcriptions.
        """
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT id, text, language, duration_seconds, created_at, word_count "
                "FROM transcriptions ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            return [Transcription.from_row(tuple(row)) for row in cursor.fetchall()]

    def get_transcription_count(self) -> int:
        """Get total number of transcriptions.

        Returns:
            Total count of transcriptions.
        """
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM transcriptions")
            return cursor.fetchone()[0]

    def delete_transcription(self, transcription_id: int) -> bool:
        """Delete a transcription by ID.

        Args:
            transcription_id: The ID of the transcription to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM transcriptions WHERE id = ?", (transcription_id,))
            return cursor.rowcount > 0

    def delete_old_transcriptions(self, days: int = 7) -> int:
        """Delete transcriptions older than specified days.

        Args:
            days: Number of days to keep transcriptions.

        Returns:
            Number of deleted transcriptions.
        """
        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM transcriptions WHERE created_at < datetime('now', ?)",
                (f'-{days} days',)
            )
            return cursor.rowcount

    def get_last_transcription(self) -> Optional[Transcription]:
        """Get the most recent transcription.

        Returns:
            The last transcription or None if none exist.
        """
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT id, text, language, duration_seconds, created_at, word_count "
                "FROM transcriptions ORDER BY created_at DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                return Transcription.from_row(tuple(row))
            return None

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
