import unittest

from ovos_classifiers.heuristics.numeric import EnglishNumberParser, Replaceablenumber


class TestKeywords(unittest.TestCase):

    def test_convert(self):
        parser = EnglishNumberParser()

        self.assertEquals(parser.convert_words_to_numbers("this is test number two"),
                          "this is test number 2")
        self.assertEquals(parser.convert_words_to_numbers("this is test number two and a half"),
                          "this is test number 2.5")
        self.assertEquals(parser.convert_words_to_numbers("this is the first test"),
                          "this is the first test")
        self.assertEquals(parser.convert_words_to_numbers("this is the first test", ordinals=True),
                          "this is the 1 test")

    def test_extract(self):
        parser = EnglishNumberParser()

        self.assertEquals(parser.extract_numbers("this is test number two".split())[0].value, 2)
        self.assertEquals(parser.extract_numbers("this is test number two and a half".split())[0].value, 2.5)
        self.assertEquals(parser.extract_numbers("this is the first test".split()), [])
        self.assertEquals(parser.extract_numbers("this is the first test".split(), ordinals=True)[0].value, 1)

        self.assertEquals(parser.extract_numbers("this is test number one 2 three".split())[0].value, 1)
        self.assertEquals(parser.extract_numbers("this is test number one 2 three".split())[1].value, 2)
        self.assertEquals(parser.extract_numbers("this is test number one 2 three".split())[2].value, 3)
