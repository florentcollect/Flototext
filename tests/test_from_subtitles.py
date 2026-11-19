"""Tests for cutting a video's audio along its subtitles."""

import unittest

from benchmarks.from_subtitles import Cue, group_cues, parse_srt

SRT = """1
00:00:00,220 --> 00:00:06,340
le digne héritier de la 3.5

2
00:00:06,340 --> 00:00:11,380
de <i>Donjons et Dragons</i> d'origine.

3
00:00:11,820 --> 00:00:15,640
Très cadré au niveau
des règles.
"""


class ParseSrtTests(unittest.TestCase):
    def test_parses_times_tags_and_multiline_text(self):
        cues = parse_srt(SRT)
        self.assertEqual(len(cues), 3)
        self.assertAlmostEqual(cues[0].start, 0.22)
        self.assertAlmostEqual(cues[1].end, 11.38)
        self.assertEqual(cues[1].text, "de Donjons et Dragons d'origine.")
        self.assertEqual(cues[2].text, "Très cadré au niveau des règles.")


class GroupCuesTests(unittest.TestCase):
    def test_segments_stay_under_the_limit_and_keep_every_cue(self):
        cues = [Cue(i * 4.0, i * 4.0 + 3.5, f"c{i}") for i in range(20)]
        segments = group_cues(cues, max_seconds=20.0, pause=0.3)
        for seg in segments:
            self.assertLessEqual(seg[-1].end - seg[0].start, 20.0)
        self.assertEqual([c.text for seg in segments for c in seg], [c.text for c in cues])

    def test_cut_falls_on_a_pause_rather_than_between_glued_cues(self):
        # c0-c1 glued, pause, c2-c3 glued: overflow at c3 cuts at the pause.
        cues = [Cue(0, 5, "c0"), Cue(5, 9, "c1"), Cue(10, 15, "c2"), Cue(15, 21, "c3")]
        segments = group_cues(cues, max_seconds=20.0, pause=0.3)
        self.assertEqual([[c.text for c in seg] for seg in segments], [["c0", "c1"], ["c2", "c3"]])

    def test_overlong_single_cue_stays_alone(self):
        cues = [Cue(0, 3, "a"), Cue(3.5, 30, "long"), Cue(31, 33, "b")]
        segments = group_cues(cues, max_seconds=20.0, pause=0.3)
        self.assertEqual([[c.text for c in seg] for seg in segments], [["a"], ["long"], ["b"]])


if __name__ == "__main__":
    unittest.main()


class BuildTakesTests(unittest.TestCase):
    def test_manifest_marks_its_references_as_subtitles(self):
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        from benchmarks.from_subtitles import build_takes

        audio = np.zeros(16000 * 3, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            build_takes(audio, [[Cue(0.5, 1.5, "bonjour")]], Path(tmp))
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest[0]["source"], "subtitles")
        self.assertEqual(manifest[0]["reference"], "bonjour")
