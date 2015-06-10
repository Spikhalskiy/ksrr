import unittest
import pandas as pd
from resources import get_input
from spell_checking import SpellChecker, is_levenshtein_similar
from tokenizing import tokenize

__author__ = 'Dmitry Spikhalskiy <dmitry@spikhalskiy.com>'

train = pd.read_csv(get_input("train.csv"), encoding='utf-8')



class SpellCheckerTest(unittest.TestCase):
    def test(self):
        spell_checker = SpellChecker(tokenize, 5, 7)

        spell_checker.index_corpus_for_spell(train["product_description"])

        # self.assertEqual("philips", spell_checker.correct("phillips"))
        # self.assertEqual("a100500", spell_checker.correct("a100500"))
        # self.assertEqual("red/black", spell_checker.correct("red/black"))
        # self.assertEqual("black", spell_checker.correct("bleck"))
        # self.assertEqual("cooker", spell_checker.correct("Cooker/"))
        self.assertEqual("old", spell_checker.correct("'old"))

        # self.assertEqual(True, is_levenshtein_similar("abcd", "ab", 2))
