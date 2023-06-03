from collections import OrderedDict
from typing import List

from ovos_utils.json_helper import invert_dict
from ovos_classifiers.heuristics.tokenize import word_tokenize, partition_list, \
    Token, ReplaceableNumber


def is_numeric(word):
    """
    Takes in a string and tests to see if it is a number.
    Args:
        word (str): string to test if a number
    Returns:
        (bool): True if a number, else False

    """
    try:
        float(word)
        return True
    except ValueError:
        return False


def look_for_fractions(split_list):
    """"
    This function takes a list made by fraction & determines if a fraction.

    Args:
        split_list (list): list created by splitting on '/'
    Returns:
        (bool): False if not a fraction, otherwise True

    """

    if len(split_list) == 2:
        if is_numeric(split_list[0]) and is_numeric(split_list[1]):
            return True

    return False


class GermanNumberParser:
    # taken from lingua_franca
    _ARTICLES_DE = {'der', 'das', 'die', 'dem', 'den'}

    #_SPOKEN_NUMBER
    _NUM_STRING_DE = {
        0: 'null',
        1: 'eins',
        2: 'zwei',
        3: 'drei',
        4: 'vier',
        5: 'fünf',
        6: 'sechs',
        7: 'sieben',
        8: 'acht',
        9: 'neun',
        10: 'zehn',
        11: 'elf',
        12: 'zwölf',
        13: 'dreizehn',
        14: 'vierzehn',
        15: 'fünfzehn',
        16: 'sechzehn',
        17: 'siebzehn',
        18: 'achtzehn',
        19: 'neunzehn',
        20: 'zwanzig',
        30: 'dreißig',
        40: 'vierzig',
        50: 'fünfzig',
        60: 'sechzig',
        70: 'siebzig',
        80: 'achtzig',
        90: 'neunzig',
        100: 'hundert',
        200: 'zweihundert',
        300: 'dreihundert',
        400: 'vierhundert',
        500: 'fünfhundert',
        600: 'sechshundert',
        700: 'siebenhundert',
        800: 'achthundert',
        900: 'neunhundert',
        1000: 'tausend',
        1000000: 'million'
    }

    _STRING_NUM_DE = invert_dict(_NUM_STRING_DE)
    _STRING_NUM_DE.update({
        'ein': 1,
        'eine': 1,
        'einer': 1,
        'einem': 1,
        'einen': 1
    })

    _MONTHS_DE = ['januar', 'februar', 'märz', 'april', 'mai', 'juni',
                  'juli', 'august', 'september', 'oktober', 'november',
                  'dezember']

    # German uses "long scale" https://en.wikipedia.org/wiki/Long_and_short_scales
    # Currently, numbers are limited to 1000000000000000000000000,
    # but _NUM_POWERS_OF_TEN can be extended to include additional number words


    _NUM_POWERS_OF_TEN_DE = [
        '', 'tausend', 'Million', 'Milliarde', 'Billion', 'Billiarde', 'Trillion',
        'Trilliarde'
    ]

    _FRACTION_STRING_DE = {
        2: 'halb',
        3: 'drittel',
        4: 'viertel',
        5: 'fünftel',
        6: 'sechstel',
        7: 'siebtel',
        8: 'achtel',
        9: 'neuntel',
        10: 'zehntel',
        11: 'elftel',
        12: 'zwölftel',
        13: 'dreizehntel',
        14: 'vierzehntel',
        15: 'fünfzehntel',
        16: 'sechzehntel',
        17: 'siebzehntel',
        18: 'achtzehntel',
        19: 'neunzehntel',
        20: 'zwanzigstel'
    }

    _STRING_FRACTION_DE = invert_dict(_FRACTION_STRING_DE)
    _STRING_FRACTION_DE.update({
        'halb': 2,
        'halbe': 2,
        'halben': 2,
        'halbes': 2,
        'halber': 2,
        'halbem': 2
    })

    # Numbers below 1 million are written in one word in German, yielding very
    # long words
    # In some circumstances it may better to seperate individual words
    # Set _EXTRA_SPACE_DA=" " for separating numbers below 1 million (
    # orthographically incorrect)
    _EXTRA_SPACE_DE = ""

    _ORDINAL_BASE_DE = {
        "1.": "erst",
        "2.": "zweit",
        "3.": "dritt",
        "4.": "viert",
        "5.": "fünft",
        "6.": "sechst",
        "7.": "siebt",
        "8.": "acht",
        "9.": "neunt",
        "10.": "zehnt",
        "11.": "elft",
        "12.": "zwölft",
        "13.": "dreizehnt",
        "14.": "vierzehnt",
        "15.": "fünfzehnt",
        "16.": "sechzehnt",
        "17.": "siebzehnt",
        "18.": "achtzehnt",
        "19.": "neunzehnt",
        "20.": "zwanzigst",
        "21.": "einundzwanzigst",
        "22.": "zweiundzwanzigst",
        "23.": "dreiundzwanzigst",
        "24.": "vierundzwanzigst",
        "25.": "fünfundzwanzigst",
        "26.": "sechsundzwanzigst",
        "27.": "siebenundzwanzigst",
        "28.": "achtundzwanzigst",
        "29.": "neunundzwanzigst",
        "30.": "dreißigst",
        "31.": "einunddreißigst",
        "32.": "zweiunddreißigst",
        "33.": "dreiunddreißigst",
        "34.": "vierunddreißigst",
        "35.": "fünfunddreißigst",
        "36.": "sechsunddreißigst",
        "37.": "siebenunddreißigst",
        "38.": "achtunddreißigst",
        "39.": "neununddreißigst",
        "40.": "vierzigst",
        "41.": "einundvierzigst",
        "42.": "zweiundvierzigst",
        "43.": "dreiundvierzigst",
        "44.": "vierundvierzigst",
        "45.": "fünfundvierzigst",
        "46.": "sechsundvierzigst",
        "47.": "siebenundvierzigst",
        "48.": "achtundvierzigst",
        "49.": "neunundvierzigst",
        "50.": "fünfzigst",
        "51.": "einundfünfzigst",
        "52.": "zweiundfünfzigst",
        "53.": "dreiundfünfzigst",
        "60.": "sechzigst",
        "70.": "siebzigst",
        "80.": "achtzigst",
        "90.": "neunzigst",
        "100.": "einhundertst",
        "1000.": "eintausendst",
        "1000000.": "millionst"
        }

    _LONG_SCALE_DE = OrderedDict([
        (100, 'hundert'),
        (1000, 'tausend'),
        (1000000, 'million'),
        (1e9, "milliarde"),
        (1e12, 'billion'),
        (1e15, "billiarde"),
        (1e18, "trillion"),
        (1e21, "trilliarde"),
        (1e24, "quadrillion"),
        (1e27, "quadrilliarde")
    ])

    _MULTIPLIER_DE = set(_LONG_SCALE_DE.values())

    _STRING_LONG_SCALE_DE = invert_dict(_LONG_SCALE_DE)

    # ending manipulation
    for number, item in _LONG_SCALE_DE.items():
        if int(number) > 1000:
            if item.endswith('e'):
                name = item + 'n'
                _MULTIPLIER_DE.add(name)
                _STRING_LONG_SCALE_DE[name] = number
            else:
                name = item + 'en'
                _MULTIPLIER_DE.add(name)
                _STRING_LONG_SCALE_DE[name] = number

    _LONG_ORDINAL_DE = {
        1e6: "millionst",
        1e9: "milliardst",
        1e12: "billionst",
        1e15: "billiardst",
        1e18: "trillionst",
        1e21: "trilliardst",
        1e24: "quadrillionst",
        1e27: "quadrilliardst"
    }

    _LONG_ORDINAL_DE.update(_ORDINAL_BASE_DE)

    # dict für erste, drittem, millionstes ...
    _STRING_LONG_ORDINAL_DE = {ord+ending: num for ord, num in invert_dict(_LONG_ORDINAL_DE).items()
                               for ending in ("en", "em", "es", "er", "e")}
    _FRACTION_MARKER_DE = set()
    _NEGATIVES_DE = {"minus"}
    _NUMBER_CONNECTORS_DE = {"und"}
    _COMMA_DE = {"komma", "comma", "punkt"}


    def is_ordinal_de(self, input_str):
        """
        This function takes the given text and checks if it is an ordinal number.
        Args:
            input_str (str): the string to check if ordinal
        Returns:
            (bool) or (float): False if not an ordinal, otherwise the number
            corresponding to the ordinal
        ordinals for 1, 3, 7 and 8 are irregular
        only works for ordinals corresponding to the numbers in _STRING_NUM
        """
        val = self._STRING_LONG_ORDINAL_DE.get(input_str.lower(), False)
        # account for numbered ordinals
        if not val and input_str.endswith('.') and is_numeric(input_str[:-1]):
            val = input_str
        return val

    def is_fractional_de(self, input_str, short_scale=False):
        """
        This function takes the given text and checks if it is a fraction.
        Args:
            input_str (str): the string to check if fractional
            short_scale (bool): use short scale if True, long scale if False
        Returns:
            (bool) or (float): False if not a fraction, otherwise the fraction
        """
        # account for different numerators, e.g. zweidrittel

        input_str = input_str.lower()
        numerator = 1
        prev_number = 0
        denominator = False
        remainder = ""

        # first check if is a fraction containing a char (eg "2/3")
        _bucket = input_str.split('/')
        if look_for_fractions(_bucket):
            numerator = float(_bucket[0])
            denominator = float(_bucket[1])

        if not denominator:
            for fraction in sorted(self._STRING_FRACTION_DE.keys(),
                                key=lambda x: len(x),
                                reverse=True):
                if fraction in input_str and not denominator:
                    denominator = self._STRING_FRACTION_DE.get(fraction)
                    remainder = input_str.replace(fraction, "")
                    break

            if remainder:
                if not self._STRING_NUM_DE.get(remainder, False):
                    #acount for eineindrittel
                    for numstring, number in self._STRING_NUM_DE.items():
                        if remainder.endswith(numstring):
                            prev_number = self._STRING_NUM_DE.get(
                                remainder.replace(numstring, "", 1), 0)
                            numerator = number
                            break
                    else:
                        return False
                else:
                    numerator = self._STRING_NUM_DE.get(remainder)

        if denominator:
            return prev_number + (numerator / denominator)
        else:
            return False

    def is_number_de(self, word: str):
        if is_numeric(word):
            if word.isdigit():
                return int(word)
            else:
                return float(word)
        elif word in self._STRING_NUM_DE:
            return self._STRING_NUM_DE.get(word)
        elif word in self._STRING_LONG_SCALE_DE:
            return self._STRING_LONG_SCALE_DE.get(word)
        
        return None

    def convert_words_to_numbers(self, utterance, short_scale=False,
                                 ordinals=False, fractions=True):
        """
        Convert words in a string into their equivalent numbers.
        Args:
            text str:
            short_scale boolean: True if short scale numberres should be used.
            ordinals boolean: True if ordinals (e.g. first, second, third) should
                            be parsed to their number values (1, 2, 3...)
        Returns:
            str
            The original text, with numbers subbed in where appropriate.
        """
        tokens = [Token(word, index) for index, word in enumerate(word_tokenize(utterance))]
        numbers_to_replace = self.extract_numbers(tokens, short_scale, ordinals, fractions)

        results = []
        for token in tokens:
            if not numbers_to_replace or \
                    token.index < numbers_to_replace[0].start_index:
                results.append(token.word)
            else:
                if numbers_to_replace and \
                        token.index == numbers_to_replace[0].start_index:
                    results.append(str(numbers_to_replace[0].value))
                if numbers_to_replace and \
                        token.index == numbers_to_replace[0].end_index:
                    numbers_to_replace.pop(0)

        return ' '.join(results)


    def extract_numbers(self, tokens: list, short_scale: bool=False, ordinals: bool=False) -> List:
        """
        extract numeric values from a list of tokens.
        Args:
            tokens (list): list of tokens (str)
            short_scale boolean: True if short scale numbers should be used.
            ordinals boolean: True if ordinals (e.g. first, second, third) should
                              be parsed to their number values (1, 2, 3...)
        Returns:
            list of extraced numbers (ReplaceableNumber)

        """
        if not isinstance(tokens[0], Token): # list of string tokens
            tokens = [Token(word, index) for index, word in enumerate(tokens)]
        numbers_to_replace = self._extract_numbers_with_text_de(tokens, short_scale, ordinals)
        numbers_to_replace.sort(key=lambda number: number.start_index)
        return numbers_to_replace

    def _extract_numbers_with_text_de(self, tokens, short_scale=True,
                                      ordinals=False, fractions=True):
        """
        Extract all numbers from a list of Tokens, with the words that
        represent them.

        Args:
            [Token]: The tokens to parse.
            short_scale bool: True if short scale numbers should be used, False for
                            long scale. True by default.
            ordinals bool: True if ordinal words (first, second, third, etc) should
                        be parsed.
            fractional_numbers bool: True if we should look for fractions and
                                    decimals.

        Returns:
            [ReplaceableNumber]: A list of tuples, each containing a number and a
                            string.

        """
        placeholder = "<placeholder>"  # inserted to maintain correct indices
        results = []
        while True:
            to_replace = \
                self._extract_number_with_text_de(tokens, short_scale,
                                                  ordinals)

            if not to_replace:
                break

            if isinstance(to_replace.value, float) and not fractions:
                pass
            else:
                results.append(to_replace)

            tokens = [
                t if not
                to_replace.start_index <= t.index <= to_replace.end_index
                else
                Token(placeholder, t.index) for t in tokens
            ]
        results.sort(key=lambda n: n.start_index)
        return results


    def _extract_number_with_text_de(self, tokens, short_scale=True,
                                     ordinals=False):
        """
        This function extracts a number from a list of Tokens.

        Args:
            tokens str: the string to normalize
            short_scale (bool): use short scale if True, long scale if False
            ordinals (bool): consider ordinal numbers
            fractional_numbers (bool): True if we should look for fractions and
                                    decimals.
        Returns:
            ReplaceableNumber

        """
        number, tokens = \
            self._extract_number_with_text_de_helper(tokens, short_scale,
                                                     ordinals)
        return ReplaceableNumber(number, tokens)


    def _extract_number_with_text_de_helper(self, tokens,
                                            short_scale, ordinals):
        """
        Helper for _extract_number_with_text_de.

        Args:
            tokens [Token]:
            short_scale boolean:
            ordinals boolean:
            fractional_numbers boolean:
        Returns:
            int or float, [Tokens]
        """
        if ordinals:
            for token in tokens:
                ordinal = self.is_ordinal_de(token.word)
                if ordinal:
                    return ordinal, [token]

        return self._extract_real_number_with_text_de(tokens, short_scale)


    def _extract_real_number_with_text_de(self, tokens, short_scale):
        """
        This is handling real numbers.

        Args:
            tokens [Token]:
            short_scale boolean:
        Returns:
            int or float, [Tokens]
            The value parsed, and tokens that it corresponds to.
        """
        number_words = []
        val = _val = _current_val = None
        _comma = False
        to_sum = []

        for idx, token in enumerate(tokens):

            _prev_val = _current_val
            _current_val = None

            word = token.word

            if word in self._NUMBER_CONNECTORS_DE and not number_words:
                continue
            if word in (self._NEGATIVES_DE |
                        self._NUMBER_CONNECTORS_DE |
                        self._COMMA_DE):
                number_words.append(token)
                if word in self._COMMA_DE:
                    _comma = token
                    _current_val = _val or _prev_val
                continue

            prev_word = tokens[idx - 1].word if idx > 0 else ""
            next_word = tokens[idx + 1].word if idx + 1 < len(tokens) else ""

            if word not in self._STRING_LONG_SCALE_DE and \
                    word not in self._STRING_NUM_DE and \
                    word not in self._MULTIPLIER_DE and \
                    not is_numeric(word) and \
                    not self.is_fractional_de(word):
                words_only = [token.word for token in number_words]
                if _val is not None:
                    to_sum.append(_val)
                if to_sum:
                    val = sum(to_sum)

                if number_words and (not all([w in self._ARTICLES_DE |
                                                   self._NEGATIVES_DE|
                                                   self._NUMBER_CONNECTORS_DE
                                              for w in words_only])
                                or str(val) == number_words[-1].word):
                    break
                else:
                    number_words.clear()
                    to_sum.clear()
                    val = _val = _prev_val = None
                continue
            elif word not in self._MULTIPLIER_DE \
                    and prev_word not in self._MULTIPLIER_DE \
                    and prev_word not in self._NUMBER_CONNECTORS_DE \
                    and prev_word not in self._NEGATIVES_DE \
                    and prev_word not in self._COMMA_DE \
                    and prev_word not in self._STRING_LONG_SCALE_DE \
                    and prev_word not in self._STRING_NUM_DE \
                    and not self.is_ordinal_de(word) \
                    and not is_numeric(prev_word)  \
                    and not self.is_fractional_de(prev_word):
                number_words = [token]
            else:
                number_words.append(token)

            # is this word already a number or a word of a number?
            _val = _current_val = self.is_number_de(word)

            # is this a negative number?
            if _current_val is not None and prev_word in self._NEGATIVES_DE:
                _val = 0 - _current_val
            
            # is the prev word a number and should we multiply it?
            if _prev_val is not None and ( word in self._MULTIPLIER_DE or \
                word in ("einer", "eines", "einem")):
                to_sum.append(_prev_val * _current_val or _current_val)
                _val = _current_val = None
            
            # fraction handling
            _fraction_val = self.is_fractional_de(word, short_scale=short_scale)
            if _fraction_val:
                if _prev_val is not None and prev_word != "eine" and \
                        word not in self._STRING_FRACTION_DE:   # zusammengesetzter Bruch
                    _val = _prev_val + _fraction_val
                    if prev_word not in self._NUMBER_CONNECTORS_DE \
                            and tokens[idx -1] not in number_words:
                        number_words.append(tokens[idx - 1])
                elif _prev_val is not None:
                    _val = _prev_val * _fraction_val
                    if tokens[idx -1] not in number_words:
                        number_words.append(tokens[idx - 1])
                else:
                    _val = _fraction_val
                _current_val = _val
            
            # directly following numbers without relation
            if (is_numeric(prev_word) or prev_word in self._STRING_NUM_DE) \
                    and not _fraction_val \
                    and not self.is_fractional_de(next_word) \
                    and not to_sum:
                val = _prev_val
                number_words.pop(-1)
                break

            # is this a spoken time ("drei viertel acht")
            if isinstance(_prev_val, float) and self.is_number_de(word) and not to_sum:
                if idx+1 < len(tokens):
                    _, number = self._extract_real_number_with_text_de([tokens[idx + 1]],
                                                                       short_scale=short_scale)
                if not next_word or not number:
                    val = f"{_val-1}:{int(60*_prev_val)}"
                    break
 
            # spoken decimals
            if _current_val is not None and _comma:
                # to_sum = [ 1, 0.2, 0.04,...]
                to_sum.append(_current_val if _current_val >= 10 else (
                    _current_val) / (10 ** (token.index - _comma.index)))
                _val = _current_val = None

            if _current_val is not None and \
                    next_word in (self._NUMBER_CONNECTORS_DE | self._COMMA_DE | {""}):
                to_sum.append(_val or _current_val)
                _val = _current_val = None
            
            if not next_word and number_words:
                val = sum(to_sum) or _val

        return val, number_words


# TODO - finish adding public user facing methods
class EnglishNumberParser:
    # taken from lingua_franca

    # TODO - from json file
    _ARTICLES_EN = {'a', 'an', 'the'}
    _NUM_STRING_EN = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine',
        10: 'ten',
        11: 'eleven',
        12: 'twelve',
        13: 'thirteen',
        14: 'fourteen',
        15: 'fifteen',
        16: 'sixteen',
        17: 'seventeen',
        18: 'eighteen',
        19: 'nineteen',
        20: 'twenty',
        30: 'thirty',
        40: 'forty',
        50: 'fifty',
        60: 'sixty',
        70: 'seventy',
        80: 'eighty',
        90: 'ninety'
    }
    _FRACTION_STRING_EN = {
        2: 'half',
        3: 'third',
        4: 'forth',
        5: 'fifth',
        6: 'sixth',
        7: 'seventh',
        8: 'eigth',
        9: 'ninth',
        10: 'tenth',
        11: 'eleventh',
        12: 'twelveth',
        13: 'thirteenth',
        14: 'fourteenth',
        15: 'fifteenth',
        16: 'sixteenth',
        17: 'seventeenth',
        18: 'eighteenth',
        19: 'nineteenth',
        20: 'twentyith'
    }
    _LONG_SCALE_EN = OrderedDict([
        (100, 'hundred'),
        (1000, 'thousand'),
        (1000000, 'million'),
        (1e12, "billion"),
        (1e18, 'trillion'),
        (1e24, "quadrillion"),
        (1e30, "quintillion"),
        (1e36, "sextillion"),
        (1e42, "septillion"),
        (1e48, "octillion"),
        (1e54, "nonillion"),
        (1e60, "decillion"),
        (1e66, "undecillion"),
        (1e72, "duodecillion"),
        (1e78, "tredecillion"),
        (1e84, "quattuordecillion"),
        (1e90, "quinquadecillion"),
        (1e96, "sedecillion"),
        (1e102, "septendecillion"),
        (1e108, "octodecillion"),
        (1e114, "novendecillion"),
        (1e120, "vigintillion"),
        (1e306, "unquinquagintillion"),
        (1e312, "duoquinquagintillion"),
        (1e336, "sesquinquagintillion"),
        (1e366, "unsexagintillion")
    ])
    _SHORT_SCALE_EN = OrderedDict([
        (100, 'hundred'),
        (1000, 'thousand'),
        (1000000, 'million'),
        (1e9, "billion"),
        (1e12, 'trillion'),
        (1e15, "quadrillion"),
        (1e18, "quintillion"),
        (1e21, "sextillion"),
        (1e24, "septillion"),
        (1e27, "octillion"),
        (1e30, "nonillion"),
        (1e33, "decillion"),
        (1e36, "undecillion"),
        (1e39, "duodecillion"),
        (1e42, "tredecillion"),
        (1e45, "quattuordecillion"),
        (1e48, "quinquadecillion"),
        (1e51, "sedecillion"),
        (1e54, "septendecillion"),
        (1e57, "octodecillion"),
        (1e60, "novendecillion"),
        (1e63, "vigintillion"),
        (1e66, "unvigintillion"),
        (1e69, "uuovigintillion"),
        (1e72, "tresvigintillion"),
        (1e75, "quattuorvigintillion"),
        (1e78, "quinquavigintillion"),
        (1e81, "qesvigintillion"),
        (1e84, "septemvigintillion"),
        (1e87, "octovigintillion"),
        (1e90, "novemvigintillion"),
        (1e93, "trigintillion"),
        (1e96, "untrigintillion"),
        (1e99, "duotrigintillion"),
        (1e102, "trestrigintillion"),
        (1e105, "quattuortrigintillion"),
        (1e108, "quinquatrigintillion"),
        (1e111, "sestrigintillion"),
        (1e114, "septentrigintillion"),
        (1e117, "octotrigintillion"),
        (1e120, "noventrigintillion"),
        (1e123, "quadragintillion"),
        (1e153, "quinquagintillion"),
        (1e183, "sexagintillion"),
        (1e213, "septuagintillion"),
        (1e243, "octogintillion"),
        (1e273, "nonagintillion"),
        (1e303, "centillion"),
        (1e306, "uncentillion"),
        (1e309, "duocentillion"),
        (1e312, "trescentillion"),
        (1e333, "decicentillion"),
        (1e336, "undecicentillion"),
        (1e363, "viginticentillion"),
        (1e366, "unviginticentillion"),
        (1e393, "trigintacentillion"),
        (1e423, "quadragintacentillion"),
        (1e453, "quinquagintacentillion"),
        (1e483, "sexagintacentillion"),
        (1e513, "septuagintacentillion"),
        (1e543, "ctogintacentillion"),
        (1e573, "nonagintacentillion"),
        (1e603, "ducentillion"),
        (1e903, "trecentillion"),
        (1e1203, "quadringentillion"),
        (1e1503, "quingentillion"),
        (1e1803, "sescentillion"),
        (1e2103, "septingentillion"),
        (1e2403, "octingentillion"),
        (1e2703, "nongentillion"),
        (1e3003, "millinillion")
    ])
    _ORDINAL_BASE_EN = {
        1: 'first',
        2: 'second',
        3: 'third',
        4: 'fourth',
        5: 'fifth',
        6: 'sixth',
        7: 'seventh',
        8: 'eighth',
        9: 'ninth',
        10: 'tenth',
        11: 'eleventh',
        12: 'twelfth',
        13: 'thirteenth',
        14: 'fourteenth',
        15: 'fifteenth',
        16: 'sixteenth',
        17: 'seventeenth',
        18: 'eighteenth',
        19: 'nineteenth',
        20: 'twentieth',
        30: 'thirtieth',
        40: "fortieth",
        50: "fiftieth",
        60: "sixtieth",
        70: "seventieth",
        80: "eightieth",
        90: "ninetieth",
        1e2: "hundredth",
        1e3: "thousandth"
    }
    _SHORT_ORDINAL_EN = {
        1e6: "millionth",
        1e9: "billionth",
        1e12: "trillionth",
        1e15: "quadrillionth",
        1e18: "quintillionth",
        1e21: "sextillionth",
        1e24: "septillionth",
        1e27: "octillionth",
        1e30: "nonillionth",
        1e33: "decillionth"
        # TODO > 1e-33
    }
    _SHORT_ORDINAL_EN.update(_ORDINAL_BASE_EN)
    _LONG_ORDINAL_EN = {
        1e6: "millionth",
        1e12: "billionth",
        1e18: "trillionth",
        1e24: "quadrillionth",
        1e30: "quintillionth",
        1e36: "sextillionth",
        1e42: "septillionth",
        1e48: "octillionth",
        1e54: "nonillionth",
        1e60: "decillionth"
        # TODO > 1e60
    }
    _LONG_ORDINAL_EN.update(_ORDINAL_BASE_EN)
    # negate next number (-2 = 0 - 2)
    _NEGATIVES_EN = {"negative", "minus"}
    # sum the next number (twenty two = 20 + 2)
    _SUMS_EN = {'twenty', '20', 'thirty', '30', 'forty', '40', 'fifty', '50',
                'sixty', '60', 'seventy', '70', 'eighty', '80', 'ninety', '90'}
    _MULTIPLIES_LONG_SCALE_EN = set(_LONG_SCALE_EN.values()) | \
                                {value + "s" for value in _LONG_SCALE_EN.values()}
    _MULTIPLIES_SHORT_SCALE_EN = set(_SHORT_SCALE_EN.values()) | \
                                 {value + "s" for value in _SHORT_SCALE_EN.values()}
    # split sentence parse separately and sum ( 2 and a half = 2 + 0.5 )
    _FRACTION_MARKER_EN = {"and"}
    # decimal marker ( 1 point 5 = 1 + 0.5)
    _DECIMAL_MARKER_EN = {"point", "dot"}
    _STRING_NUM_EN = {v: k for k, v in _NUM_STRING_EN.items()}
    _STRING_NUM_EN.update({key + 's': value for key, value in _STRING_NUM_EN.items()})
    _SPOKEN_EXTRA_NUM_EN = {
        "half": 0.5,
        "halves": 0.5,
        "couple": 2
    }
    _STRING_SHORT_ORDINAL_EN = {v: k for k, v in _SHORT_ORDINAL_EN.items()}
    _STRING_LONG_ORDINAL_EN = {v: k for k, v in _LONG_ORDINAL_EN.items()}

    def is_fractional(self, input_str, short_scale=True, spoken=True):
        """
        This function takes the given text and checks if it is a fraction.

        Args:
            input_str (str): the string to check if fractional
            short_scale (bool): use short scale if True, long scale if False
            spoken (bool): consider "half", "quarter", "whole" a fraction
        Returns:
            (bool) or (float): False if not a fraction, otherwise the fraction

        """
        if input_str.endswith('s', -1):
            input_str = input_str[:len(input_str) - 1]  # e.g. "fifths"

        fracts = {"whole": 1, "half": 2, "halve": 2, "quarter": 4}
        if short_scale:
            for num in self._SHORT_ORDINAL_EN:
                if num > 2:
                    fracts[self._SHORT_ORDINAL_EN[num]] = num
        else:
            for num in self._LONG_ORDINAL_EN:
                if num > 2:
                    fracts[self._LONG_ORDINAL_EN[num]] = num

        if input_str.lower() in fracts and spoken:
            return 1.0 / fracts[input_str.lower()]
        return False

    def convert_words_to_numbers(self, utterance, short_scale=True, ordinals=False):
        """
        Convert words in a string into their equivalent numbers.
        Args:
            text str:
            short_scale boolean: True if short scale numbers should be used.
            ordinals boolean: True if ordinals (e.g. first, second, third) should
                              be parsed to their number values (1, 2, 3...)

        Returns:
            str
            The original text, with numbers subbed in where appropriate.

        """
        tokens = [Token(word, index) for index, word in enumerate(word_tokenize(utterance))]
        numbers_to_replace = self.extract_numbers(tokens, short_scale, ordinals)

        results = []
        for token in tokens:
            if not numbers_to_replace or \
                    token.index < numbers_to_replace[0].start_index:
                results.append(token.word)
            else:
                if numbers_to_replace and \
                        token.index == numbers_to_replace[0].start_index:
                    results.append(str(numbers_to_replace[0].value))
                if numbers_to_replace and \
                        token.index == numbers_to_replace[0].end_index:
                    numbers_to_replace.pop(0)

        return ' '.join(results)

    def extract_numbers(self, tokens: list, short_scale: bool=True, ordinals: bool=False) -> List:
        """
        extract numeric values from a list of tokens.
        Args:
            tokens (list): list of tokens (str)
            short_scale boolean: True if short scale numbers should be used.
            ordinals boolean: True if ordinals (e.g. first, second, third) should
                              be parsed to their number values (1, 2, 3...)
        Returns:
            list of extraced numbers (ReplaceableNumber)

        """
        if not isinstance(tokens[0], Token): # list of string tokens
            tokens = [Token(word, index) for index, word in enumerate(tokens)]
        numbers_to_replace = self._extract_numbers_with_text_en(tokens, short_scale, ordinals)
        numbers_to_replace.sort(key=lambda number: number.start_index)
        return numbers_to_replace

    # helper methods
    def _initialize_number_data_en(self, short_scale, speech=True):
        """
        Generate dictionaries of words to numbers, based on scale.

        This is a helper function for _extract_whole_number.

        Args:
            short_scale (bool):
            speech (bool): consider extra words (_SPOKEN_EXTRA_NUM_EN) to be numbers

        Returns:
            (set(str), dict(str, number), dict(str, number))
            multiplies, string_num_ordinal, string_num_scale

        """
        multiplies = self._MULTIPLIES_SHORT_SCALE_EN if short_scale \
            else self._MULTIPLIES_LONG_SCALE_EN

        string_num_ordinal_en = self._STRING_SHORT_ORDINAL_EN if short_scale \
            else self._STRING_LONG_ORDINAL_EN

        string_num_scale_en = self._SHORT_SCALE_EN if short_scale else self._LONG_SCALE_EN
        string_num_scale_en = {v: k for k, v in string_num_scale_en.items()}
        string_num_scale_en.update({key + 's': value for key, value in string_num_scale_en.items()})

        if speech:
            string_num_scale_en.update(self._SPOKEN_EXTRA_NUM_EN)
        return multiplies, string_num_ordinal_en, string_num_scale_en

    def _extract_fraction_with_text_en(self, tokens, short_scale, ordinals):
        """
        Extract fraction numbers from a string.

        This function handles text such as '2 and 3/4'. Note that "one half" or
        similar will be parsed by the whole number function.

        Args:
            tokens [Token]: words and their indexes in the original string.
            short_scale boolean:
            ordinals boolean:

        Returns:
            (int or float, [Token])
            The value found, and the list of relevant tokens.
            (None, None) if no fraction value is found.

        """
        for c in self._FRACTION_MARKER_EN:
            partitions = partition_list(tokens, lambda t: t.word == c)

            if len(partitions) == 3:
                numbers1 = \
                    self._extract_numbers_with_text_en(partitions[0], short_scale,
                                                       ordinals, fractional_numbers=False)
                numbers2 = \
                    self._extract_numbers_with_text_en(partitions[2], short_scale,
                                                       ordinals, fractional_numbers=True)

                if not numbers1 or not numbers2:
                    return None, None

                # ensure first is not a fraction and second is a fraction
                num1 = numbers1[-1]
                num2 = numbers2[0]
                if num1.value >= 1 and 0 < num2.value < 1:
                    return num1.value + num2.value, \
                           num1.tokens + partitions[1] + num2.tokens

        return None, None

    def _extract_decimal_with_text_en(self, tokens, short_scale, ordinals):
        """
        Extract decimal numbers from a string.

        This function handles text such as '2 point 5'.

        Notes:
            While this is a helper for extractnumber_en, it also depends on
            extractnumber_en, to parse out the components of the decimal.

            This does not currently handle things like:
                number dot number number number

        Args:
            tokens [Token]: The text to parse.
            short_scale boolean:
            ordinals boolean:

        Returns:
            (float, [Token])
            The value found and relevant tokens.
            (None, None) if no decimal value is found.

        """
        for c in self._DECIMAL_MARKER_EN:
            partitions = partition_list(tokens, lambda t: t.word == c)

            if len(partitions) == 3:
                numbers1 = \
                    self._extract_numbers_with_text_en(partitions[0], short_scale,
                                                       ordinals, fractional_numbers=False)
                numbers2 = \
                    self._extract_numbers_with_text_en(partitions[2], short_scale,
                                                       ordinals, fractional_numbers=False)

                if not numbers1 or not numbers2:
                    return None, None

                number = numbers1[-1]
                decimal = numbers2[0]

                # TODO handle number dot number number number
                if "." not in str(decimal.text):
                    return number.value + float('0.' + str(decimal.value)), \
                           number.tokens + partitions[1] + decimal.tokens
        return None, None

    def _extract_whole_number_with_text_en(self, tokens, short_scale, ordinals):
        """
        Handle numbers not handled by the decimal or fraction functions. This is
        generally whole numbers. Note that phrases such as "one half" will be
        handled by this function, while "one and a half" are handled by the
        fraction function.

        Args:
            tokens [Token]:
            short_scale boolean:
            ordinals boolean:

        Returns:
            int or float, [Tokens]
            The value parsed, and tokens that it corresponds to.

        """
        multiplies, string_num_ordinal, string_num_scale = \
            self._initialize_number_data_en(short_scale, speech=ordinals is not None)

        number_words = []  # type: List[Token]
        val = False
        prev_val = None
        next_val = None
        to_sum = []
        for idx, token in enumerate(tokens):
            current_val = None
            if next_val:
                next_val = None
                continue

            word = token.word.lower()
            if word in self._ARTICLES_EN or word in self._NEGATIVES_EN:
                number_words.append(token)
                continue

            prev_word = tokens[idx - 1].word.lower() if idx > 0 else ""
            next_word = tokens[idx + 1].word.lower() if idx + 1 < len(tokens) else ""

            if is_numeric(word[:-2]) and \
                    (word.endswith("st") or word.endswith("nd") or
                     word.endswith("rd") or word.endswith("th")):

                # explicit ordinals, 1st, 2nd, 3rd, 4th.... Nth
                word = word[:-2]

                # handle nth one
                if next_word == "one":
                    # would return 1 instead otherwise
                    tokens[idx + 1] = Token("", idx)
                    next_word = ""

            # TODO replaces the wall of "and" and "or" with all() or any() as
            #  appropriate, the whole codebase should be checked for this pattern
            if word not in string_num_scale and \
                    word not in self._STRING_NUM_EN and \
                    word not in self._SUMS_EN and \
                    word not in multiplies and \
                    not (ordinals and word in string_num_ordinal) and \
                    not is_numeric(word) and \
                    not self.is_fractional(word, short_scale=short_scale) and \
                    not look_for_fractions(word.split('/')):
                words_only = [token.word for token in number_words]

                if number_words and not all([w.lower() in self._ARTICLES_EN |
                                             self._NEGATIVES_EN for w in words_only]):
                    break
                else:
                    number_words = []
                    continue
            elif word not in multiplies \
                    and prev_word not in multiplies \
                    and prev_word not in self._SUMS_EN \
                    and not (ordinals and prev_word in string_num_ordinal) \
                    and prev_word not in self._NEGATIVES_EN \
                    and prev_word not in self._ARTICLES_EN:
                number_words = [token]

            elif prev_word in self._SUMS_EN and word in self._SUMS_EN:
                number_words = [token]
            elif ordinals is None and \
                    (word in string_num_ordinal or word in self._SPOKEN_EXTRA_NUM_EN):
                # flagged to ignore this token
                continue
            else:
                number_words.append(token)

            # is this word already a number ?
            if is_numeric(word):
                if word.isdigit():  # doesn't work with decimals
                    val = int(word)
                else:
                    val = float(word)
                current_val = val

            # is this word the name of a number ?
            if word in self._STRING_NUM_EN:
                val = self._STRING_NUM_EN.get(word)
                current_val = val
            elif word in string_num_scale:
                val = string_num_scale.get(word)
                current_val = val
            elif ordinals and word in string_num_ordinal:
                val = string_num_ordinal[word]
                current_val = val

            # is the prev word an ordinal number and current word is one?
            # second one, third one
            if ordinals and prev_word in string_num_ordinal and val == 1:
                val = prev_val

            # is the prev word a number and should we sum it?
            # twenty two, fifty six
            if (prev_word in self._SUMS_EN and val and val < 10) or all([prev_word in
                                                                         multiplies,
                                                                         val < prev_val if prev_val else False]):
                val = prev_val + val

            # is the prev word a number and should we multiply it?
            # twenty hundred, six hundred
            if word in multiplies:
                if not prev_val:
                    prev_val = 1
                val = prev_val * val

            # is this a spoken fraction?
            # half cup
            if val is False and \
                    not (ordinals is None and word in string_num_ordinal):
                val = self.is_fractional(word, short_scale=short_scale,
                                         spoken=ordinals is not None)

                current_val = val

            # 2 fifths
            if ordinals is False:
                next_val = self.is_fractional(next_word, short_scale=short_scale)
                if next_val:
                    if not val:
                        val = 1
                    val = val * next_val
                    number_words.append(tokens[idx + 1])

            # is this a negative number?
            if val and prev_word and prev_word in self._NEGATIVES_EN:
                val = 0 - val

            # let's make sure it isn't a fraction
            if not val:
                # look for fractions like "2/3"
                aPieces = word.split('/')
                if look_for_fractions(aPieces):
                    val = float(aPieces[0]) / float(aPieces[1])
                    current_val = val

            else:
                if current_val and all([
                    prev_word in self._SUMS_EN,
                    word not in self._SUMS_EN,
                    word not in multiplies,
                    current_val >= 10]):
                    # Backtrack - we've got numbers we can't sum.
                    number_words.pop()
                    val = prev_val
                    break
                prev_val = val

                if word in multiplies and next_word not in multiplies:
                    # handle long numbers
                    # six hundred sixty six
                    # two million five hundred thousand
                    #
                    # This logic is somewhat complex, and warrants
                    # extensive documentation for the next coder's sake.
                    #
                    # The current word is a power of ten. `current_val` is
                    # its integer value. `val` is our working sum
                    # (above, when `current_val` is 1 million, `val` is
                    # 2 million.)
                    #
                    # We have a dict `string_num_scale` containing [value, word]
                    # pairs for "all" powers of ten: string_num_scale[10] == "ten.
                    #
                    # We need go over the rest of the tokens, looking for other
                    # powers of ten. If we find one, we compare it with the current
                    # value, to see if it's smaller than the current power of ten.
                    #
                    # Numbers which are not powers of ten will be passed over.
                    #
                    # If all the remaining powers of ten are smaller than our
                    # current value, we can set the current value aside for later,
                    # and begin extracting another portion of our final result.
                    # For example, suppose we have the following string.
                    # The current word is "million".`val` is 9000000.
                    # `current_val` is 1000000.
                    #
                    #    "nine **million** nine *hundred* seven **thousand**
                    #     six *hundred* fifty seven"
                    #
                    # Iterating over the rest of the string, the current
                    # value is larger than all remaining powers of ten.
                    #
                    # The if statement passes, and nine million (9000000)
                    # is appended to `to_sum`.
                    #
                    # The main variables are reset, and the main loop begins
                    # assembling another number, which will also be appended
                    # under the same conditions.
                    #
                    # By the end of the main loop, to_sum will be a list of each
                    # "place" from 100 up: [9000000, 907000, 600]
                    #
                    # The final three digits will be added to the sum of that list
                    # at the end of the main loop, to produce the extracted number:
                    #
                    #    sum([9000000, 907000, 600]) + 57
                    # == 9,000,000 + 907,000 + 600 + 57
                    # == 9,907,657
                    #
                    # >>> foo = "nine million nine hundred seven thousand six
                    #            hundred fifty seven"
                    # >>> extract_number(foo)
                    # 9907657

                    time_to_sum = True
                    for other_token in tokens[idx + 1:]:
                        if other_token.word.lower() in multiplies:
                            if string_num_scale[other_token.word.lower()] >= current_val:
                                time_to_sum = False
                            else:
                                continue
                        if not time_to_sum:
                            break
                    if time_to_sum:
                        to_sum.append(val)
                        val = 0
                        prev_val = 0

        if val is not None and to_sum:
            val += sum(to_sum)

        return val, number_words

    def _extract_number_with_text_en_helper(self, tokens,
                                            short_scale=True, ordinals=False,
                                            fractional_numbers=True):
        """
        Helper for _extract_number_with_text_en.

        This contains the real logic for parsing, but produces
        a result that needs a little cleaning (specific, it may
        contain leading articles that can be trimmed off).

        Args:
            tokens [Token]:
            short_scale boolean:
            ordinals boolean:
            fractional_numbers boolean:

        Returns:
            int or float, [Tokens]

        """
        if fractional_numbers:
            fraction, fraction_text = \
                self._extract_fraction_with_text_en(tokens, short_scale, ordinals)
            if fraction:
                return fraction, fraction_text

            decimal, decimal_text = \
                self._extract_decimal_with_text_en(tokens, short_scale, ordinals)
            if decimal:
                return decimal, decimal_text

        return self._extract_whole_number_with_text_en(tokens, short_scale, ordinals)

    def _extract_number_with_text_en(self, tokens, short_scale=True,
                                     ordinals=False, fractional_numbers=True):
        """
        This function extracts a number from a list of Tokens.

        Args:
            tokens str: the string to normalize
            short_scale (bool): use short scale if True, long scale if False
            ordinals (bool): consider ordinal numbers, third=3 instead of 1/3
            fractional_numbers (bool): True if we should look for fractions and
                                       decimals.
        Returns:
            ReplaceableNumber

        """
        number, tokens = \
            self._extract_number_with_text_en_helper(tokens, short_scale,
                                                     ordinals, fractional_numbers)
        while tokens and tokens[0].word in self._ARTICLES_EN:
            tokens.pop(0)
        return ReplaceableNumber(number, tokens)

    def _extract_numbers_with_text_en(self, tokens, short_scale=True,
                                      ordinals=False, fractional_numbers=True):
        """
        Extract all numbers from a list of Tokens, with the words that
        represent them.

        Args:
            [Token]: The tokens to parse.
            short_scale bool: True if short scale numbers should be used, False for
                              long scale. True by default.
            ordinals bool: True if ordinal words (first, second, third, etc) should
                           be parsed.
            fractional_numbers bool: True if we should look for fractions and
                                     decimals.

        Returns:
            [ReplaceableNumber]: A list of tuples, each containing a number and a
                             string.

        """
        placeholder = "<placeholder>"  # inserted to maintain correct indices
        results = []
        while True:
            to_replace = \
                self._extract_number_with_text_en(tokens, short_scale,
                                                  ordinals, fractional_numbers)

            if not to_replace:
                break

            results.append(to_replace)

            tokens = [
                t if not
                to_replace.start_index <= t.index <= to_replace.end_index
                else
                Token(placeholder, t.index) for t in tokens
            ]
        results.sort(key=lambda n: n.start_index)
        return results


class AzerbaijaniNumberParser:
    # taken from lingua_franca

    # TODO - from json file
    _NUM_STRING_AZ = {
        0: 'sıfır',
        1: 'bir',
        2: 'iki',
        3: 'üç',
        4: 'dörd',
        5: 'beş',
        6: 'altı',
        7: 'yeddi',
        8: 'səkkiz',
        9: 'doqquz',
        10: 'on',
        11: 'on bir',
        12: 'on iki',
        13: 'on üç',
        14: 'on dörd',
        15: 'on beş',
        16: 'on altı',
        17: 'on yeddi',
        18: 'on səkkiz',
        19: 'on doqquz',
        20: 'iyirmi',
        30: 'otuz',
        40: 'qırx',
        50: 'əlli',
        60: 'altmış',
        70: 'yetmiş',
        80: 'səksən',
        90: 'doxsan'
    }
    _FRACTION_STRING_AZ = {
        2: 'ikidə',
        3: 'üçdə',
        4: 'dörddə',
        5: 'beşdə',
        6: 'altıda',
        7: 'yeddidə',
        8: 'səkkizdə',
        9: 'doqquzda',
        10: 'onda',
        11: 'on birdə',
        12: 'on ikidə',
        13: 'on üçdə',
        14: 'on dörddə',
        15: 'on beşdə',
        16: 'on altıda',
        17: 'on yeddidə',
        18: 'on səkkizdə',
        19: 'on doqquzda',
        20: 'iyirmidə',
        30: 'otuzda',
        40: 'qırxda',
        50: 'əllidə',
        60: 'altmışda',
        70: 'yetmişdə',
        80: 'səksəndə',
        90: 'doxsanda',
        1e2: 'yüzdə',
        1e3: 'mində'
    }
    _LONG_SCALE_AZ = OrderedDict([
        (100, 'yüz'),
        (1000, 'min'),
        (1000000, 'milyon'),
        (1e12, "milyard"),
        (1e18, 'trilyon'),
        (1e24, "kvadrilyon"),
        (1e30, "kvintilyon"),
        (1e36, "sekstilyon"),
        (1e42, "septilyon"),
        (1e48, "oktilyon"),
        (1e54, "nonilyon"),
        (1e60, "dekilyon")
    ])
    _SHORT_SCALE_AZ = OrderedDict([
        (100, 'yüz'),
        (1000, 'min'),
        (1000000, 'milyon'),
        (1e9, "milyard"),
        (1e12, 'trilyon'),
        (1e15, "kvadrilyon"),
        (1e18, "kvintilyon"),
        (1e21, "sekstilyon"),
        (1e24, "septilyon"),
        (1e27, "oktilyon"),
        (1e30, "nonilyon"),
        (1e33, "dekilyon")
    ])
    _ORDINAL_BASE_AZ = {
        1: 'birinci',
        2: 'ikinci',
        3: 'üçüncü',
        4: 'dördüncü',
        5: 'beşinci',
        6: 'altıncı',
        7: 'yeddinci',
        8: 'səkkizinci',
        9: 'doqquzuncu',
        10: 'onuncu',
        11: 'on birinci',
        12: 'on ikinci',
        13: 'on üçüncü',
        14: 'on dördüncü',
        15: 'on beşinci',
        16: 'on altıncı',
        17: 'on yeddinci',
        18: 'on səkkizinci',
        19: 'on doqquzuncu',
        20: 'iyirminci',
        30: 'otuzuncu',
        40: "qırxıncı",
        50: "əllinci",
        60: "altmışıncı",
        70: "yetmışinci",
        80: "səksəninci",
        90: "doxsanınçı",
        1e2: "yüzüncü",
        1e3: "mininci"
    }
    _SHORT_ORDINAL_AZ = {
        1e6: "milyonuncu",
        1e9: "milyardıncı",
        1e12: "trilyonuncu",
        1e15: "kvadrilyonuncu",
        1e18: "kvintilyonuncu",
        1e21: "sekstilyonuncu",
        1e24: "septilyonuncu",
        1e27: "oktilyonuncu",
        1e30: "nonilyonuncu",
        1e33: "dekilyonuncu"
        # TODO > 1e-33
    }
    _SHORT_ORDINAL_AZ.update(_ORDINAL_BASE_AZ)
    _LONG_ORDINAL_AZ = {
        1e6: "milyonuncu",
        1e12: "milyardıncı",
        1e18: "trilyonuncu",
        1e24: "kvadrilyonuncu",
        1e30: "kvintilyonuncu",
        1e36: "sekstilyonuncu",
        1e42: "septilyonuncu",
        1e48: "oktilyonuncu",
        1e54: "nonilyonuncu",
        1e60: "dekilyonuncu"
        # TODO > 1e60
    }
    _LONG_ORDINAL_AZ.update(_ORDINAL_BASE_AZ)
    # negate next number (-2 = 0 - 2)
    _NEGATIVES_AZ = {"mənfi", "minus"}
    # sum the next number (iyirmi iki = 20 + 2)
    _SUMS_AZ = {'on', '10', 'iyirmi', '20', 'otuz', '30', 'qırx', '40', 'əlli', '50',
                'altmış', '60', 'yetmiş', '70', 'səksən', '80', 'doxsan', '90'}
    _MULTIPLIES_LONG_SCALE_AZ = set(_LONG_SCALE_AZ.values()) | \
                                set(_LONG_SCALE_AZ.values())
    _MULTIPLIES_SHORT_SCALE_AZ = set(_SHORT_SCALE_AZ.values()) | \
                                 set(_SHORT_SCALE_AZ.values())
    # split sentence parse separately and sum ( 2 and a half = 2 + 0.5 )
    _FRACTION_MARKER_AZ = {"və"}
    # decimal marker ( 1 nöqtə 5 = 1 + 0.5)
    _DECIMAL_MARKER_AZ = {"nöqtə"}
    _STRING_NUM_AZ = {v: k for k, v in _NUM_STRING_AZ.items()}
    _SPOKEN_EXTRA_NUM_AZ = {
        "yarım": 0.5,
        "üçdəbir": 1 / 3,
        "dörddəbir": 1 / 4
    }
    _STRING_SHORT_ORDINAL_AZ = {v: k for k, v in _SHORT_ORDINAL_AZ.items()}
    _STRING_LONG_ORDINAL_AZ = {v: k for k, v in _LONG_ORDINAL_AZ.items()}

    def convert_words_to_numbers(self, text, short_scale=True, ordinals=False):
        """
        Convert words in a string into their equivalent numbers.
        Args:
            text str:
            short_scale boolean: True if short scale numbers should be used.
            ordinals boolean: True if ordinals (e.g. birinci, ikinci, üçüncü) should
                              be parsed to their number values (1, 2, 3...)

        Returns:
            str
            The original text, with numbers subbed in where appropriate.

        """
        tokens = [Token(word, index) for index, word in enumerate(word_tokenize(text))]
        numbers_to_replace = self.extract_numbers_az(tokens, short_scale, ordinals)
        results = []
        for token in tokens:
            if not numbers_to_replace or \
                    token.index < numbers_to_replace[0].start_index:
                results.append(token.word)
            else:
                if numbers_to_replace and \
                        token.index == numbers_to_replace[0].start_index:
                    results.append(str(numbers_to_replace[0].value))
                if numbers_to_replace and \
                        token.index == numbers_to_replace[0].end_index:
                    numbers_to_replace.pop(0)

        return ' '.join(results)

    def extract_numbers(self, tokens: list, short_scale: bool=False, ordinals: bool=False) -> List:
        """
        extract numeric values from a list of tokens.
        Args:
            tokens (list): list of tokens (str)
            short_scale boolean: True if short scale numbers should be used.
            ordinals boolean: True if ordinals (e.g. first, second, third) should
                              be parsed to their number values (1, 2, 3...)
        Returns:
            list of extraced numbers (ReplaceableNumber)

        """
        if not isinstance(tokens[0], Token): # list of string tokens
            tokens = [Token(word, index) for index, word in enumerate(tokens)]
        numbers_to_replace = self._extract_numbers_with_text_az(tokens, short_scale, ordinals)
        numbers_to_replace.sort(key=lambda number: number.start_index)
        return numbers_to_replace

    def is_fractional(self, input_str, short_scale=True, spoken=True):
        """
        This function takes the given text and checks if it is a fraction.

        Args:
            input_str (str): the string to check if fractional
            short_scale (bool): use short scale if True, long scale if False
            spoken (bool):
        Returns:
            (bool) or (float): False if not a fraction, otherwise the fraction

        """

        fracts = {"dörddəbir": 4, "yarım": 2, "üçdəbir": 3}
        for num in self._FRACTION_STRING_AZ:
            if num > 2:
                fracts[self._FRACTION_STRING_AZ[num]] = num

        if input_str.lower() in fracts and spoken:
            return 1.0 / fracts[input_str.lower()]
        return False

    # helper methods

    def _extract_numbers_with_text_az(self, tokens, short_scale=True,
                                      ordinals=False, fractional_numbers=True):
        """
        Extract all numbers from a list of Tokens, with the words that
        represent them.

        Args:
            [Token]: The tokens to parse.
            short_scale bool: True if short scale numbers should be used, False for
                              long scale. True by default.
            ordinals bool: True if ordinal words (birinci, ikinci, üçüncü, etc) should
                           be parsed.
            fractional_numbers bool: True if we should look for fractions and
                                     decimals.

        Returns:
            [ReplaceableNumber]: A list of tuples, each containing a number and a
                             string.

        """
        placeholder = "<placeholder>"  # inserted to maintain correct indices
        results = []
        while True:
            to_replace = \
                self._extract_number_with_text_az(tokens, short_scale,
                                                  ordinals, fractional_numbers)
            if not to_replace:
                break

            results.append(to_replace)

            tokens = [
                t if not
                to_replace.start_index <= t.index <= to_replace.end_index
                else
                Token(placeholder, t.index) for t in tokens
            ]
        results.sort(key=lambda n: n.start_index)
        return results

    def _extract_number_with_text_az(self, tokens, short_scale=True,
                                     ordinals=False, fractional_numbers=True):
        """
        This function extracts a number from a list of Tokens.

        Args:
            tokens str: the string to normalize
            short_scale (bool): use short scale if True, long scale if False
            ordinals (bool): consider ordinal numbers
            fractional_numbers (bool): True if we should look for fractions and
                                       decimals.
        Returns:
            ReplaceableNumber

        """
        number, tokens = \
            self._extract_number_with_text_az_helper(tokens, short_scale,
                                                     ordinals, fractional_numbers)
        return ReplaceableNumber(number, tokens)

    def _extract_number_with_text_az_helper(self, tokens,
                                            short_scale=True, ordinals=False,
                                            fractional_numbers=True):
        """
        Helper for _extract_number_with_text_az.

        This contains the real logic for parsing, but produces
        a result that needs a little cleaning (specific, it may
        contain leading articles that can be trimmed off).

        Args:
            tokens [Token]:
            short_scale boolean:
            ordinals boolean:
            fractional_numbers boolean:

        Returns:
            int or float, [Tokens]

        """
        if fractional_numbers:
            fraction, fraction_text = \
                self._extract_fraction_with_text_az(tokens, short_scale, ordinals)
            if fraction:
                # print("fraction")
                return fraction, fraction_text

            decimal, decimal_text = \
                self._extract_decimal_with_text_az(tokens, short_scale, ordinals)
            if decimal:
                # print("decimal")
                return decimal, decimal_text

        return self._extract_whole_number_with_text_az(tokens, short_scale, ordinals)

    def _extract_fraction_with_text_az(self, tokens, short_scale, ordinals):
        """
        Extract fraction numbers from a string.

        This function handles text such as '2 və dörddə üç'. Note that "yarım" or
        similar will be parsed by the whole number function.

        Args:
            tokens [Token]: words and their indexes in the original string.
            short_scale boolean:
            ordinals boolean:

        Returns:
            (int or float, [Token])
            The value found, and the list of relevant tokens.
            (None, None) if no fraction value is found.

        """
        for c in self._FRACTION_MARKER_AZ:
            partitions = partition_list(tokens, lambda t: t.word == c)

            if len(partitions) == 3:
                numbers1 = \
                    self._extract_numbers_with_text_az(partitions[0], short_scale,
                                                       ordinals, fractional_numbers=False)
                numbers2 = \
                    self._extract_numbers_with_text_az(partitions[2], short_scale,
                                                       ordinals, fractional_numbers=True)

                if not numbers1 or not numbers2:
                    return None, None

                # ensure first is not a fraction and second is a fraction
                num1 = numbers1[-1]
                num2 = numbers2[0]
                if num1.value >= 1 and 0 < num2.value < 1:
                    return num1.value + num2.value, \
                           num1.tokens + partitions[1] + num2.tokens

        return None, None

    def _extract_decimal_with_text_az(self, tokens, short_scale, ordinals):
        """
        Extract decimal numbers from a string.

        This function handles text such as '2 nöqtə 5'.

        Notes:
            While this is a helper for extractnumber_az, it also depends on
            extractnumber_az, to parse out the components of the decimal.

            This does not currently handle things like:
                number dot number number number

        Args:
            tokens [Token]: The text to parse.
            short_scale boolean:
            ordinals boolean:

        Returns:
            (float, [Token])
            The value found and relevant tokens.
            (None, None) if no decimal value is found.

        """
        for c in self._DECIMAL_MARKER_AZ:
            partitions = partition_list(tokens, lambda t: t.word == c)

            if len(partitions) == 3:
                numbers1 = \
                    self._extract_numbers_with_text_az(partitions[0], short_scale,
                                                       ordinals, fractional_numbers=False)
                numbers2 = \
                    self._extract_numbers_with_text_az(partitions[2], short_scale,
                                                       ordinals, fractional_numbers=False)
                if not numbers1 or not numbers2:
                    return None, None

                number = numbers1[-1]
                decimal = numbers2[0]

                # TODO handle number dot number number number
                if "." not in str(decimal.text):
                    return number.value + float('0.' + str(decimal.value)), \
                           number.tokens + partitions[1] + decimal.tokens
        return None, None

    def _extract_whole_number_with_text_az(self, tokens, short_scale, ordinals):
        """
        Handle numbers not handled by the decimal or fraction functions. This is
        generally whole numbers. Note that phrases such as "yarım" will be
        handled by this function.

        Args:
            tokens [Token]:
            short_scale boolean:
            ordinals boolean:

        Returns:
            int or float, [Tokens]
            The value parsed, and tokens that it corresponds to.

        """
        multiplies, string_num_ordinal, string_num_scale = \
            self._initialize_number_data_az(short_scale, speech=ordinals is not None)

        number_words = []  # type: List[Token]
        val = False
        prev_val = None
        next_val = None
        to_sum = []
        # print(tokens, ordinals)
        for idx, token in enumerate(tokens):
            current_val = None
            if next_val:
                next_val = None
                continue

            word = token.word.lower()
            if word in self._NEGATIVES_AZ:
                number_words.append(token)
                continue

            prev_word = tokens[idx - 1].word.lower() if idx > 0 else ""
            next_word = tokens[idx + 1].word.lower() if idx + 1 < len(tokens) else ""
            # print(prev_word, word, next_word, number_words)
            if word not in string_num_scale and \
                    word not in self._STRING_NUM_AZ and \
                    word not in self._SUMS_AZ and \
                    word not in multiplies and \
                    not (ordinals and word in string_num_ordinal) and \
                    not is_numeric(word) and \
                    not self.is_fractional(word, short_scale=short_scale) and \
                    not look_for_fractions(word.split('/')):
                # print("a1")
                words_only = [token.word for token in number_words]

                if number_words and not all([w.lower() in
                                             self._NEGATIVES_AZ for w in words_only]):
                    break
                else:
                    number_words = []
                    continue
            elif word not in multiplies \
                    and word not in self._SPOKEN_EXTRA_NUM_AZ \
                    and prev_word not in multiplies \
                    and prev_word not in self._SUMS_AZ \
                    and not (ordinals and prev_word in string_num_ordinal) \
                    and prev_word not in self._NEGATIVES_AZ:
                number_words = [token]
                # print("a2")
            elif prev_word in self._SUMS_AZ and word in self._SUMS_AZ:
                number_words = [token]
                # print("a3")
            elif ordinals is None and \
                    (word in string_num_ordinal or word in self._SPOKEN_EXTRA_NUM_AZ):
                # print("a4")
                # flagged to ignore this token
                continue
            else:
                # print("a5")
                number_words.append(token)

            # is this word already a number ?
            if is_numeric(word):
                # print("b")
                if word.isdigit():  # doesn't work with decimals
                    val = int(word)
                else:
                    val = float(word)
                current_val = val

            # is this word the name of a number ?
            if word in self._STRING_NUM_AZ:
                val = self._STRING_NUM_AZ.get(word)
                current_val = val
                # print("c1", current_val)
            elif word in string_num_scale:
                val = string_num_scale.get(word)
                current_val = val
                # print("c2")
            elif ordinals and word in string_num_ordinal:
                val = string_num_ordinal[word]
                current_val = val
                # print("c3")
            # is the prev word a number and should we sum it?
            # twenty two, fifty six
            if (prev_word in self._SUMS_AZ and val and val < 10) or all([prev_word in
                                                                         multiplies,
                                                                         val < prev_val if prev_val else False]):
                val = prev_val + val
                # print("d")

            # is the prev word a number and should we multiply it?
            # twenty hundred, six hundred
            if word in multiplies:
                if not prev_val:
                    prev_val = 1
                val = prev_val * val
                # print("e")

            # is this a spoken fraction?
            # 1 yarım fincan - yarım fincan
            if current_val is None and not (ordinals is None and word in self._SPOKEN_EXTRA_NUM_AZ):
                val = self.is_fractional(word, short_scale=short_scale,
                                         spoken=ordinals is not None)
                if val:
                    if prev_val:
                        val += prev_val
                    current_val = val
                    # print("f", current_val, prev_val)
                    if word in self._SPOKEN_EXTRA_NUM_AZ:
                        break

            # dörddə bir
            if ordinals is False:
                temp = prev_val
                prev_val = self.is_fractional(prev_word, short_scale=short_scale)
                if prev_val:
                    if not val:
                        val = 1
                    val = val * prev_val
                    if idx + 1 < len(tokens):
                        number_words.append(tokens[idx + 1])
                else:
                    prev_val = temp
                # print("g", prev_val)

            # is this a negative number?
            if val and prev_word and prev_word in self._NEGATIVES_AZ:
                val = 0 - val
                # print("h")

            # let's make sure it isn't a fraction
            if not val:
                # look for fractions like "2/3"
                aPieces = word.split('/')
                if look_for_fractions(aPieces):
                    val = float(aPieces[0]) / float(aPieces[1])
                    current_val = val
                # print("i")

            else:
                if current_val and all([
                    prev_word in self._SUMS_AZ,
                    word not in self._SUMS_AZ,
                    word not in multiplies,
                    current_val >= 10]):
                    # Backtrack - we've got numbers we can't sum.
                    # print("j", number_words, prev_val)
                    number_words.pop()
                    val = prev_val
                    break
                prev_val = val

                if word in multiplies and next_word not in multiplies:
                    # handle long numbers
                    # six hundred sixty six
                    # two million five hundred thousand
                    #
                    # This logic is somewhat complex, and warrants
                    # extensive documentation for the next coder's sake.
                    #
                    # The current word is a power of ten. `current_val` is
                    # its integer value. `val` is our working sum
                    # (above, when `current_val` is 1 million, `val` is
                    # 2 million.)
                    #
                    # We have a dict `string_num_scale` containing [value, word]
                    # pairs for "all" powers of ten: string_num_scale[10] == "ten.
                    #
                    # We need go over the rest of the tokens, looking for other
                    # powers of ten. If we find one, we compare it with the current
                    # value, to see if it's smaller than the current power of ten.
                    #
                    # Numbers which are not powers of ten will be passed over.
                    #
                    # If all the remaining powers of ten are smaller than our
                    # current value, we can set the current value aside for later,
                    # and begin extracting another portion of our final result.
                    # For example, suppose we have the following string.
                    # The current word is "million".`val` is 9000000.
                    # `current_val` is 1000000.
                    #
                    #    "nine **million** nine *hundred* seven **thousand**
                    #     six *hundred* fifty seven"
                    #
                    # Iterating over the rest of the string, the current
                    # value is larger than all remaining powers of ten.
                    #
                    # The if statement passes, and nine million (9000000)
                    # is appended to `to_sum`.
                    #
                    # The main variables are reset, and the main loop begins
                    # assembling another number, which will also be appended
                    # under the same conditions.
                    #
                    # By the end of the main loop, to_sum will be a list of each
                    # "place" from 100 up: [9000000, 907000, 600]
                    #
                    # The final three digits will be added to the sum of that list
                    # at the end of the main loop, to produce the extracted number:
                    #
                    #    sum([9000000, 907000, 600]) + 57
                    # == 9,000,000 + 907,000 + 600 + 57
                    # == 9,907,657
                    #
                    # >>> foo = "nine million nine hundred seven thousand six
                    #            hundred fifty seven"
                    # >>> extract_number(foo)
                    # 9907657
                    # print("k", tokens[idx+1:])
                    time_to_sum = True
                    for other_token in tokens[idx + 1:]:
                        if other_token.word.lower() in multiplies:
                            if string_num_scale[other_token.word.lower()] >= current_val:
                                time_to_sum = False
                            else:
                                continue
                        if not time_to_sum:
                            break
                    if time_to_sum:
                        # print("l")
                        to_sum.append(val)
                        val = 0
                        prev_val = 0

        if val is not None and to_sum:
            # print("m", to_sum)
            val += sum(to_sum)
        # print(val, number_words, "end")
        return val, number_words

    def _initialize_number_data_az(self, short_scale, speech=True):
        """
        Generate dictionaries of words to numbers, based on scale.

        This is a helper function for _extract_whole_number.

        Args:
            short_scale (bool):
            speech (bool): consider extra words (_SPOKEN_EXTRA_NUM_AZ) to be numbers

        Returns:
            (set(str), dict(str, number), dict(str, number))
            multiplies, string_num_ordinal, string_num_scale

        """
        multiplies = self._MULTIPLIES_SHORT_SCALE_AZ if short_scale \
            else self._MULTIPLIES_LONG_SCALE_AZ

        string_num_ordinal_az = self._STRING_SHORT_ORDINAL_AZ if short_scale \
            else self._STRING_LONG_ORDINAL_AZ

        string_num_scale_az = self._SHORT_SCALE_AZ if short_scale else self._LONG_SCALE_AZ
        string_num_scale_az = {v: k for k, v in string_num_scale_az.items()}

        return multiplies, string_num_ordinal_az, string_num_scale_az
