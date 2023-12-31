import re
from collections import namedtuple
from datetime import datetime, date, timedelta, time
from typing import List, Any

from ovos_utils import flatten_list
from quebra_frases import word_tokenize as _wtok, sentence_tokenize as _stok

# Token is intended to be used in the number processing functions in
# this module. The parsing requires slicing and dividing of the original
# text. To ensure things parse correctly, we need to know where text came
# from in the original input, hence this nametuple.
Token = namedtuple('Token', 'word index')


class ReplaceableEntity:
    """
    Similar to Token, this class is used in entity parsing.

    Once we've found an entity in a string, this class contains all
    the info about the value, and where it came from in the original text.
    In other words, it is the text, and the entity that can replace it in
    the string.
    """

    def __init__(self, value: Any, tokens: List):
        self.value = value
        self.tokens = tokens

    @property
    def type(self):
        return type(self.value)

    def __bool__(self):
        return bool(self.value is not None and self.value is not False)

    @property
    def start_index(self):
        return self.tokens[0].index

    @property
    def end_index(self):
        return self.tokens[-1].index

    @property
    def text(self):
        return ' '.join([t.word for t in self.tokens])

    def __setattr__(self, key, value):
        try:
            getattr(self, key)
        except AttributeError:
            super().__setattr__(key, value)
        else:
            raise Exception("Immutable!")

    def __str__(self):
        return f"({self.value}, {self.tokens})"

    def __repr__(self):
        return "{n}({v}, {t})".format(n=self.__class__.__name__, v=self.value,
                                      t=[t.word for t in self.tokens])


class ReplaceableNumber(ReplaceableEntity):
    """
    Similar to Token, this class is used in number parsing.

    Once we've found a number in a string, this class contains all
    the info about the value, and where it came from in the original text.
    In other words, it is the text, and the number that can replace it in
    the string.
    """


class ReplaceableDate(ReplaceableEntity):
    """
    Similar to Token, this class is used in date parsing.

    Once we've found a date in a string, this class contains all
    the info about the value, and where it came from in the original text.
    In other words, it is the text, and the date that can replace it in
    the string.
    """

    def __init__(self, value: date, tokens: List):
        if isinstance(value, datetime):
            value = value.date()
        assert isinstance(value, date)
        super().__init__(value, tokens)


class ReplaceableTime(ReplaceableEntity):
    """
    Similar to Token, this class is used in date parsing.

    Once we've found a time in a string, this class contains all
    the info about the value, and where it came from in the original text.
    In other words, it is the text, and the time that can replace it in
    the string.
    """

    def __init__(self, value: time, tokens: List):
        if isinstance(value, datetime):
            value = value.time()
        assert isinstance(value, time)
        super().__init__(value, tokens)


class ReplaceableTimedelta(ReplaceableEntity):
    """
    Similar to Token, this class is used in date parsing.

    Once we've found a timedelta in a string, this class contains all
    the info about the value, and where it came from in the original text.
    In other words, it is the text, and the duration that can replace it in
    the string.
    """

    def __init__(self, value: timedelta, tokens: List):
        assert isinstance(value, timedelta)
        super().__init__(value, tokens)


def partition_list(items, split_on):
    """
    Partition a list of items.

    Works similarly to str.partition

    Args:
        items:
        split_on callable:
            Should return a boolean. Each item will be passed to
            this callable in succession, and partitions will be
            created any time it returns True.

    Returns:
        [[any]]

    """
    splits = []
    current_split = []
    for item in items:
        if split_on(item):
            splits.append(current_split)
            splits.append([item])
            current_split = []
        else:
            current_split.append(item)
    splits.append(current_split)
    return list(filter(lambda x: len(x) != 0, splits))


def sentence_tokenize(text):
    sents = [_stok(s) for s in text.split("\n")]
    return flatten_list(sents)


def word_tokenize(utterance, lang=None):
    if lang is not None and lang.startswith("pt"):
        return word_tokenize_pt(utterance)
    elif lang is not None and lang.startswith("ca"):
        return word_tokenize_ca(utterance)
    # Split things like 12%
    utterance = re.sub(r"([0-9]+)([\%])", r"\1 \2", utterance)
    # Split thins like #1
    utterance = re.sub(r"(\#)([0-9]+\b)", r"\1 \2", utterance)
    return _wtok(utterance)


def word_tokenize_pt(utterance):
    # Split things like 12%
    utterance = re.sub(r"([0-9]+)([\%])", r"\1 \2", utterance)
    # Split things like #1
    utterance = re.sub(r"(\#)([0-9]+\b)", r"\1 \2", utterance)
    # Split things like amo-te
    utterance = re.sub(r"([a-zA-Z]+)(-)([a-zA-Z]+\b)", r"\1 \2 \3",
                       utterance)
    tokens = utterance.split()
    if tokens[-1] == '-':
        tokens = tokens[:-1]

    return tokens


def word_tokenize_ca(utterance):
    # Split things like 12%
    utterance = re.sub(r"([0-9]+)([\%])", r"\1 \2", utterance)
    # Split things like #1
    utterance = re.sub(r"(\#)([0-9]+\b)", r"\1 \2", utterance)
    # Don't split at -
    tokens = utterance.split()
    if tokens[-1] == '-':
        tokens = tokens[:-1]

    return tokens


def subword_tokenize(utterance):
    """phonetically meaningful subwords,
    Pronunciation-assisted Subword Modeling, generates linguistically
    meaningful subwords by analyzing a corpus and a dictionary.

    @inproceedings{xu2019improving,
        title={Improving End-to-end Speech Recognition with Pronunciation-assisted Sub-word Modeling},
        author={Xu, Hainan and Ding, Shuoyang and Watanabe, Shinji},
        booktitle={ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        pages={7110--7114},
        year={2019},
        organization={IEEE}
    }

    see https://github.com/hainan-xv/PASM
    """
    from ovos_classifiers.heuristics.phonemizer import EnglishARPAHeuristicPhonemizer
    return EnglishARPAHeuristicPhonemizer.subword_tokenize(utterance)


def syllable_tokenize(utterance):
    """
    The Sonority Sequencing Principle (SSP) is a language agnostic algorithm proposed
    by Otto Jesperson in 1904. The sonorous quality of a phoneme is judged by the
    openness of the lips. Syllable breaks occur before troughs in sonority. For more
    on the SSP see Selkirk (1984).

    The default implementation uses the English alphabet, but the `sonority_hiearchy`
    can be modified to IPA or any other alphabet for the use-case. The SSP is a
    universal syllabification algorithm, but that does not mean it performs equally
    across languages. Bartlett et al. (2009) is a good benchmark for English accuracy
    if utilizing IPA (pg. 311).

    Importantly, if a custom hierarchy is supplied and vowels span across more than
    one level, they should be given separately to the `vowels` class attribute.

    References:

    - Otto Jespersen. 1904. Lehrbuch der Phonetik.
      Leipzig, Teubner. Chapter 13, Silbe, pp. 185-203.
    - Elisabeth Selkirk. 1984. On the major class features and syllable theory.
      In Aronoff & Oehrle (eds.) Language Sound Structure: Studies in Phonology.
      Cambridge, MIT Press. pp. 107-136.
    - Susan Bartlett, et al. 2009. On the Syllabification of Phonemes.
      In HLT-NAACL. pp. 308-316.
    """
    from nltk.tokenize.sonority_sequencing import SyllableTokenizer
    return SyllableTokenizer().tokenize(utterance)
