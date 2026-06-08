import json
import tempfile
import unittest
from pathlib import Path

from flototext.config import Config


class ConfigBackendPersistenceTests(unittest.TestCase):
    def _fresh_config(self, tmp: str) -> Config:
        # base_dir/data is where settings.json lives -> isolate from the real file.
        cfg = Config(base_dir=Path(tmp))
        cfg.ensure_directories()
        return cfg

    def test_default_backend_is_qwen(self):
        self.assertEqual(Config().model.backend, "qwen")

    def test_save_and_load_roundtrip_persists_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._fresh_config(tmp)
            cfg.model.backend = "canary"
            cfg.save_settings()

            # File actually contains the model section.
            data = json.loads(cfg.settings_path.read_text(encoding="utf-8"))
            self.assertEqual(data["model"]["backend"], "canary")

            # A fresh config reading the same file restores the choice.
            reloaded = self._fresh_config(tmp)
            reloaded.load_settings()
            self.assertEqual(reloaded.model.backend, "canary")

    def test_legacy_settings_without_model_section_keep_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._fresh_config(tmp)
            # Simulate an old settings.json that only has the ui section.
            cfg.settings_path.write_text(
                json.dumps({"ui": {"language": "fr"}}), encoding="utf-8"
            )
            cfg.load_settings()
            self.assertEqual(cfg.ui.language, "fr")
            self.assertEqual(cfg.model.backend, "qwen")


if __name__ == "__main__":
    unittest.main()
