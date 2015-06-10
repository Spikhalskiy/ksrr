# -*- coding: utf-8 -*-
import unittest
import re
from spell_checking import SpellChecker
from tokenizing import tokenize_and_join, tokenize_with_spl, tokenize

__author__ = 'Dmitry Spikhalskiy <dmitry@spikhalskiy.com>'

class TokenizingTest(unittest.TestCase):
    def test(self):
        # self.assertEqual(tokenize_and_join("10.5FLOZ NRSH SHINE SH"), "10.5floz nrsh shine sh")
        # self.assertEqual(tokenize_and_join("The unique 180° flip design"), u"unique 180° flip design")
        # self.assertEqual(tokenize_and_join("Frigidaire FFCE1439LB - microwave oven"), "frigidaire ffce1439lb microwave oven")
        # self.assertEqual(tokenize_and_join("CIRAGO Cirago MHL1000 MHL Adapter"), "cirago cirago mhl1000 mhl adapter")
        # self.assertEqual(tokenize_and_join("Belkin WiFi Range Extender - White (F9K1015)"), "belkin wifi range extender white f9k1015")
        # self.assertEqual(tokenize_and_join("Obagi Elastiderm Eye Treatment Cream 0.5 oz / 15g Authentic NiB Sealed [5]"), "obagi elastiderm eye treatment cream 0.5 oz 15g authentic nib sealed")
        # self.assertEqual(tokenize_and_join("SP-LAMP-086"), "sp-lamp-086 lamp")
        # self.assertEqual(tokenize_and_join("32-inch"), "32-inch inch")
        # self.assertEqual(tokenize_and_join("Vizio E320Fi-B2 32-inch 1080p 60Hz Full-Array Smart LED HDTV with Built-in Wi-Fi"), "vizio e320fi-b2 32-inch 1080p 60hz full-array smart led hdtv built-in wi-fi inch full array built")
        #
        # self.assertEqual(tokenize_and_join("oven safe to 500°F"), u"oven safe 500°f")
        # self.assertEqual(tokenize_and_join("red/black"), "red/black red black")
        # self.assertEqual(tokenize_and_join("go..."), "go")
        self.assertEqual(tokenize_with_spl(SpellChecker(tokenize, 5, 7))("You'll"), [])
        self.assertEqual(tokenize_with_spl(SpellChecker(tokenize, 5, 7))("won't"), "")