import unittest
from datetime import datetime, timedelta
from pathlib import Path

from flototext.storage.database import Database
from flototext.storage.models import Transcription


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db_path = Path(__file__).parent / "artifacts" / "database_test.sqlite"
        if self.db_path.exists():
            self.db_path.unlink()
        self.database = Database(self.db_path)

    def tearDown(self):
        self.database.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_saves_and_reads_transcription(self):
        transcription = Transcription(
            text="bonjour 200",
            language="French",
            duration_seconds=1.5,
            created_at=datetime(2026, 4, 23, 12, 0, 0),
        )

        transcription_id = self.database.save_transcription(transcription)
        saved = self.database.get_transcription(transcription_id)

        self.assertEqual(saved.text, "bonjour 200")
        self.assertEqual(saved.word_count, 2)
        self.assertEqual(self.database.get_transcription_count(), 1)

    def test_deletes_old_transcriptions(self):
        old = Transcription(text="old", created_at=datetime.now() - timedelta(days=10))
        recent = Transcription(text="recent", created_at=datetime.now())
        self.database.save_transcription(old)
        self.database.save_transcription(recent)

        deleted = self.database.delete_old_transcriptions(days=7)

        self.assertEqual(deleted, 1)
        self.assertEqual(self.database.get_last_transcription().text, "recent")


if __name__ == "__main__":
    unittest.main()
