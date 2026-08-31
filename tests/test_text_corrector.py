import unittest

from flototext.config import config
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


class FrenchProseIsLeftAloneTests(unittest.TestCase):
    """Garde-fou contre la recidive du dictionnaire empoisonne.

    En juillet puis en aout 2026, `data/custom_words.json` a accumule des cles
    qui sont des expressions francaises courantes : la substitution etant
    globale, elles se declenchaient sur de la prose ordinaire. Mesure le
    2026-08-31 avant le tri : 12 phrases detruites sur 30, dont
    "je l'ai deja fait" -> "JDR deja fait" et "il a 9 points de vie" ->
    "il a Mulch points de vie".

    Ce test lit le VRAI dictionnaire du projet, pas une copie : toute entree
    ajoutee plus tard qui abimerait du francais normal fait echouer la suite.
    Le corpus est code ici et non lu depuis data/, pour qu'il ne puisse pas
    deriver avec le fichier qu'il surveille.

    Si ce test casse : la faute est a la CLE, jamais a la valeur. Remplacer la
    cle par une deformation phonetique que le francais ne produit pas, ou la
    retirer. Ne jamais desactiver ce test pour faire passer une correction.
    """

    CORPUS = [
        "je l'ai déjà fait hier soir",
        "il a 9 points de vie",
        "on y va, d'accord.",
        "tu peux venir ? Dites-moi.",
        "on regarde la vision.",
        "c'est un dessin.",
        "une belle mouche.",
        "je ne vais pas y arriver.",
        "bon, j'y vais.",
        "j'ai passé une journée épuisante.",
        "mes journées sont bien remplies.",
        "on se retrouve à mi-journée",
        "vous êtes prêts pour la séance ?",
        "qui peut me passer les dés ?",
        "ouvre-moi la porte.",
        "je n'ai ni peur ni remords.",
        "ce sont des propos indécents.",
        "mais carrément, il fallait le faire.",
        "j'ai des idées pour le scénario.",
        "tu veux rester encore un peu ?",
        "ils ont gaspillé toute la réserve.",
        "l'activité anthropique du bassin.",
        "Agnès arrive demain.",
        "il y a eu 2 débats hier.",
        "le personnage lance un dé de dégâts.",
        "je prépare la séance de jeu de rôle.",
        "l'eau coule doucement.",
        "elle a acheté mes carottes au marché.",
        "un des joueurs est absent.",
        "ni toi ni moi ne le savions.",
        "il ouvre la porte, ouvre-le en grand.",
        "la roche sur laquelle il grimpe.",
        # Dictees reellement par l'utilisateur le 2026-08-31, apres le tri :
        # elles etaient toutes abimees avant, elles doivent rester intactes.
        "on y va, d'accord?",
        "qui peut me passer les des ?",
        # "mi-journee" est du francais ; "mid-journee", avec un D, est ce que
        # Canary produit pour midjourney. Les deux doivent rester distincts.
        "on se retrouve a mi-journee pour dejeuner.",
        "une reunion a mi-journee, ca arrange tout le monde.",
    ]

    def test_project_dictionary_leaves_ordinary_french_untouched(self):
        corrector = TextCorrector(dictionary_path=config.data_dir / "custom_words.json")

        abimees = []
        for phrase in self.CORPUS:
            sortie = corrector.correct(phrase)
            if sortie != phrase:
                abimees.append(f"  {phrase!r}\n    -> {sortie!r}")

        self.assertEqual(
            abimees, [],
            "Des entrées de data/custom_words.json se déclenchent sur du français "
            "ordinaire. La clé est en cause, pas la valeur :\n" + "\n".join(abimees)
        )
