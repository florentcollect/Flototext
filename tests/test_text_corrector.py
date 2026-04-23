import unittest

from flototext.core.text_corrector import TextCorrector


class TextCorrectorTests(unittest.TestCase):
    def _make_corrector(self, corrections):
        corrector = TextCorrector.__new__(TextCorrector)
        corrector.dictionary_path = None
        corrector._corrections = corrections
        corrector._pattern = None
        corrector._build_pattern()
        return corrector

    def test_corrects_phrase_ending_with_punctuation(self):
        corrector = self._make_corrector({"gitpo.": "Geek Powa"})

        self.assertEqual(corrector.correct("gitpo."), "Geek Powa")

    def test_does_not_replace_inside_larger_word(self):
        corrector = self._make_corrector({"art": "ART"})

        self.assertEqual(corrector.correct("cart art article"), "cart ART article")

    def test_normalizes_spoken_numbers_before_custom_dictionary(self):
        corrector = self._make_corrector({"euros": "EUR"})

        self.assertEqual(corrector.correct("deux-cent euros"), "200 EUR")


if __name__ == "__main__":
    unittest.main()
