import unittest

from flototext.core.number_normalizer import normalize_french_numbers


class NumberNormalizerTests(unittest.TestCase):
    def test_converts_basic_spoken_numbers_to_digits(self):
        self.assertEqual(normalize_french_numbers("deux-cent"), "200")
        self.assertEqual(normalize_french_numbers("deux cents"), "200")
        self.assertEqual(normalize_french_numbers("vingt et un"), "21")
        self.assertEqual(normalize_french_numbers("quatre-vingt-dix-neuf"), "99")

    def test_converts_numbers_inside_sentence(self):
        text = "j'ai deux cent cinquante trois euros et douze centimes"

        self.assertEqual(
            normalize_french_numbers(text),
            "j'ai 253 euros et 12 centimes",
        )

    def test_preserves_non_number_words(self):
        self.assertEqual(
            normalize_french_numbers("je veux deux pommes rouges"),
            "je veux 2 pommes rouges",
        )

    def test_preserves_standalone_un_and_une_as_words(self):
        self.assertEqual(normalize_french_numbers("un ou une option"), "un ou une option")
        self.assertEqual(normalize_french_numbers("un texte classique"), "un texte classique")

    def test_keeps_un_as_digit_in_compound_numbers(self):
        self.assertEqual(normalize_french_numbers("vingt et un jours"), "21 jours")


if __name__ == "__main__":
    unittest.main()
