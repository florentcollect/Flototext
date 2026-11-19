"""Tests for the own-voice benchmark helpers (no microphone, no model)."""

import tempfile
import unittest.mock
import unittest
from pathlib import Path

import numpy as np

from benchmarks.own_voice import load_clips, select_phrases


class SelectPhrasesTests(unittest.TestCase):
    def test_filters_by_length_and_duplicates(self):
        texts = [
            "trop court",
            "Une phrase de longueur tout à fait correcte ici.",
            "une phrase de longueur tout à fait correcte ici.",  # duplicate, other case
            "  Deux   espaces   en trop   dans   cette phrase  là.  ",
            None,
        ]
        kept = select_phrases(texts, count=10, min_words=4, max_words=20)
        self.assertEqual(kept, [
            "Une phrase de longueur tout à fait correcte ici.",
            "Deux espaces en trop dans cette phrase là.",
        ])

    def test_spreads_the_pick_across_the_history(self):
        texts = [f"phrase numéro {i} pour le test" for i in range(100)]
        kept = select_phrases(texts, count=4, min_words=1, max_words=50)
        self.assertEqual(len(kept), 4)
        # Not just the first four dictations: the pick reaches the recent ones.
        self.assertIn("phrase numéro 75 pour le test", kept)


class LoadClipsTests(unittest.TestCase):
    def test_takes_become_normalized_clips_and_missing_files_are_skipped(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            audio = (np.sin(np.linspace(0, 100, 16000)) * 0.1).astype(np.float32)
            sf.write(directory / "take_001.wav", audio, 16000, subtype="PCM_16")
            manifest = [
                {"file": "take_001.wav", "reference": "bonjour"},
                {"file": "take_002.wav", "reference": "absente"},
            ]
            with unittest.mock.patch("builtins.print"):
                clips = load_clips(manifest, directory)

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].reference, "bonjour")
        self.assertAlmostEqual(clips[0].duration, 1.0, places=2)
        # Peak-normalized like the app does before transcription.
        self.assertAlmostEqual(float(np.abs(clips[0].audio).max()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()


class BenchReportTests(unittest.TestCase):
    def test_engine_failures_are_counted_not_hidden_in_the_wer(self):
        # An arena failure leaves an empty text that the WER books as deleted
        # words: Canary read 12.5 % on the video takes for a real 6.3 %.
        from benchmarks import benchmark_asr
        from benchmarks.benchmark_asr import Clip, failure_note, run_backend

        class FlakyBackend:
            calls = 0

            def load(self):
                pass

            def transcribe(self, audio, sr, label, code):
                FlakyBackend.calls += 1
                if FlakyBackend.calls == 2:
                    raise RuntimeError("BFCArena::AllocateRawInternal")
                return "bonjour", "fr"

            def cleanup(self):
                pass

        clips = [Clip(clip_id=str(i), audio=np.zeros(16000, dtype=np.float32),
                      duration=1.0, reference="bonjour") for i in range(3)]
        with unittest.mock.patch.object(benchmark_asr, "create_backend", lambda name: FlakyBackend()), \
                unittest.mock.patch.object(benchmark_asr, "gpu_used_gb", lambda: None), \
                unittest.mock.patch("builtins.print"):
            result = run_backend("canary", clips, "French", "fr")

        self.assertEqual(result.failures, 1)
        self.assertEqual(result.hyps, ["bonjour", "", "bonjour"])
        self.assertIn("1 clip", failure_note(result))

    def test_subtitle_references_get_their_own_caveat(self):
        from benchmarks.own_voice import reference_caveat
        subtitles = " ".join(reference_caveat([{"file": "a.wav", "source": "subtitles"}]))
        dictations = " ".join(reference_caveat([{"file": "a.wav"}]))
        self.assertIn("sous-titres", subtitles)
        self.assertNotIn("sous-titres", dictations)
