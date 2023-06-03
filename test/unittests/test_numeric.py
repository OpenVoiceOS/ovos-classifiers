import unittest

from ovos_classifiers.heuristics.numeric import EnglishNumberParser
from ovos_classifiers.heuristics.tokenize import word_tokenize


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

        def test_xtract(utt, expected_numbers, ordinals=False, short_scale=True):
            toks = word_tokenize(utt)
            numbers = [n.value for n in parser.extract_numbers(toks, ordinals=ordinals,
                                                               short_scale=short_scale)]
            if not isinstance(expected_numbers, list):
                expected_numbers = [expected_numbers]
            self.assertEqual(numbers, expected_numbers)

        test_xtract("this is test number two", [2])
        test_xtract("this is test number two and a half", [2.5])
        test_xtract("this is the first test", [])
        test_xtract("this is the first test", [1], ordinals=True)
        test_xtract("this is test number one 2 three", [1, 2, 3])

        test_xtract("this is a one two three  test", [1.0, 2.0, 3.0])
        test_xtract("it's  a four five six  test", [4.0, 5.0, 6.0])
        test_xtract("this is a ten eleven twelve  test", [10.0, 11.0, 12.0])
        test_xtract("this is a one twenty one  test", [1.0, 21.0])
        test_xtract("1 dog, seven pigs, macdonald had a farm, 3 times 5 macarena", [1, 7, 3, 5])
        test_xtract("two beers for two bears", [2.0, 2.0])
        test_xtract("twenty 20 twenty", [20, 20, 20])
        test_xtract("twenty 20 22", [20.0, 20.0, 22.0])
        test_xtract("twenty twenty two twenty", [20, 22, 20])
        test_xtract("twenty 2", [22.0])
        test_xtract("twenty 20 twenty 2", [20, 20, 22])
        test_xtract("third one", [1 / 3, 1])
        test_xtract("third one", [3], ordinals=True)
        test_xtract("six trillion", [6e12], short_scale=True)
        test_xtract("six trillion", [6e18], short_scale=False)
        test_xtract("two pigs and six trillion bacteria", [2, 6e12], short_scale=True)
        test_xtract("two pigs and six trillion bacteria", [2, 6e18], short_scale=False)
        test_xtract("thirty second or first", [32, 1], ordinals=True)
        test_xtract("this is a seven eight nine and a half test", [7.0, 8.0, 9.5])

        test_xtract("grobo 0", 0)
        test_xtract("a couple of beers", 2)
        test_xtract("a couple hundred beers", 200)
        test_xtract("a couple thousand beers", 2000)
        test_xtract("totally 100%", 100)

        test_xtract("this is 2 test", 2)
        test_xtract("this is test number 4", 4)
        test_xtract("three cups", 3)
        #        test_xtract("1/3 cups", 1.0 / 3.0)
        test_xtract("quarter cup", 0.25)
        #       test_xtract("1/4 cup", 0.25)
        test_xtract("one fourth cup", 0.25)
        #        test_xtract("2/3 cups", 2.0 / 3.0)
        #        test_xtract("3/4 cups", 3.0 / 4.0)
        #        test_xtract("1 and 3/4 cups", 1.75)
        test_xtract("1 cup and a half", 1.5)
        test_xtract("one cup and a half", 1.5)
        test_xtract("one and a half cups", 1.5)
        test_xtract("one and one half cups", 1.5)
        test_xtract("three quarter cups", 3.0 / 4.0)
        test_xtract("three quarters cups", 3.0 / 4.0)
        test_xtract("twenty two", 22)
        test_xtract("Twenty two with a leading capital letter", 22)
        test_xtract("twenty Two with Two capital letters", [22, 2])
        test_xtract("twenty Two with mixed capital letters", 22)
        test_xtract("two hundred", 200)
        test_xtract("nine thousand", 9000)
        test_xtract("six hundred sixty six", 666)
        test_xtract("two million", 2000000)
        test_xtract("two million five hundred thousand tons of spinning metal", 2500000)
        test_xtract("six trillion", 6000000000000.0)
        test_xtract("six trillion", 6e+18, short_scale=False)
        test_xtract("one point five", 1.5)
        test_xtract("three dot fourteen", 3.14)
        test_xtract("zero point two", 0.2)
        test_xtract("billions of years older", 1000000000.0)
        test_xtract("billions of years older", 1000000000000.0, short_scale=False)
        test_xtract("one hundred thousand", 100000)
        test_xtract("minus 2", -2)
        test_xtract("negative seventy", -70)
        test_xtract("thousand million", 1000000000)

        # Verify non-power multiples of ten no longer discard
        # adjacent multipliers
        test_xtract("twenty thousand", 20000)
        test_xtract("fifty million", 50000000)

        # Verify smaller powers of ten no longer cause miscalculation of larger
        # powers of ten (see MycroftAI#86)
        test_xtract("twenty billion three hundred million \
                                        nine hundred fifty thousand six hundred \
                                        seventy five point eight", 20300950675.8)
        test_xtract("nine hundred ninety nine million nine \
                                        hundred ninety nine thousand nine \
                                        hundred ninety nine point nine", 999999999.9)

        test_xtract("eight hundred trillion two hundred fifty seven", 800000000000257.0)

        # sanity check
        test_xtract("third", 3, ordinals=True)
        test_xtract("sixth", 6, ordinals=True)

        # test explicit ordinals
        test_xtract("this is the 1st", 1, ordinals=True)
        test_xtract("this is the 2nd", 2)
        test_xtract("this is the 3rd", 3)
        test_xtract("this is the 4th", 4)
        test_xtract("this is the 7th test", 7, ordinals=True)
        test_xtract("this is the 7th test", 7)
        test_xtract("this is the 1st test", 1)
        test_xtract("this is the 2nd test", 2)
        test_xtract("this is the 3rd test", 3)
        test_xtract("this is the 31st test", 31)
        test_xtract("this is the 32nd test", 32)
        test_xtract("this is the 33rd test", 33)
        test_xtract("this is the 34th test", 34)

        # test non ambiguous ordinals
        test_xtract("this is the first test", 1, ordinals=True)
        # test ambiguous ordinal/time unit
        test_xtract("this is second test", 2, ordinals=True)
        test_xtract("remind me in a second", 2, ordinals=True)
        test_xtract("remind me in a second", [])

        # test ambiguous ordinal/fractional
        test_xtract("this is the third test", 3.0, ordinals=True)
        test_xtract("this is the third test", 1.0 / 3.0)

        test_xtract("one third of a cup", 1.0 / 3.0)

        # test big numbers / short vs long scale
        test_xtract("this is the billionth test", 1e09, ordinals=True)
        test_xtract("this is the billionth test", 1e-9)
        test_xtract("this is the billionth test", 1e12,
                    ordinals=True,
                    short_scale=False)
        test_xtract("this is the billionth test", 1e-12, short_scale=False)

        # test the Nth one
        test_xtract("the fourth one", 4.0, ordinals=True)
        test_xtract("the thirty sixth one", 36.0, ordinals=True)
        test_xtract("you are the second one", 1)
        test_xtract("you are the second one", 2, ordinals=True)
        test_xtract("you are the 1st one", 1)
        test_xtract("you are the 2nd one", 2)
        test_xtract("you are the 3rd one", 3)
        test_xtract("you are the 8th one", 8)
