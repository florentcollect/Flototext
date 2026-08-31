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


class VramBudgetSettingsTests(unittest.TestCase):
    """The budget and the idle delay must survive a save/load round trip:
    save_settings rewrites the whole file, so a key it forgets is a key lost at
    the next backend switch from the tray."""

    def _fresh_config(self, tmp: str) -> Config:
        cfg = Config(base_dir=Path(tmp))
        cfg.ensure_directories()
        return cfg

    def test_round_trip_keeps_memory_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._fresh_config(tmp)
            cfg.model.idle_unload_seconds = 600
            cfg.model.vram_budget_gb = 5.0
            cfg.model.over_budget_idle_seconds = 90
            cfg.save_settings()

            reloaded = self._fresh_config(tmp)
            reloaded.load_settings()

            self.assertEqual(reloaded.model.idle_unload_seconds, 600)
            self.assertEqual(reloaded.model.vram_budget_gb, 5.0)
            self.assertEqual(reloaded.model.over_budget_idle_seconds, 90)

    def test_legacy_settings_without_the_keys_keep_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._fresh_config(tmp)
            cfg.settings_path.write_text(
                json.dumps({"model": {"backend": "canary"}}), encoding="utf-8")

            reloaded = self._fresh_config(tmp)
            reloaded.load_settings()

            self.assertEqual(reloaded.model.backend, "canary")
            self.assertEqual(reloaded.model.idle_unload_seconds, 900)
            self.assertEqual(reloaded.model.vram_budget_gb, 5.0)
