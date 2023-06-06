import json
import re
from os.path import dirname
from typing import List, Dict

from ovos_classifiers.heuristics.tokenize import word_tokenize
from ovos_classifiers.heuristics.numeric import EnglishNumberParser, AzerbaijaniNumberParser, GermanNumberParser


class Normalizer:
    # taken from lingua_franca
    """
    individual languages may subclass this if needed

    normalize_XX should pass a valid config read from json
    """
    _default_config = {}

    def __init__(self, config=None):
        self.config = config or self._default_config

    @staticmethod
    def tokenize(utterance) -> List[str]:
        return word_tokenize(utterance)

    @property
    def should_lowercase(self) -> bool:
        return self.config.get("lowercase", False)

    @property
    def should_numbers_to_digits(self) -> bool:
        return self.config.get("numbers_to_digits", True)

    @property
    def should_expand_contractions(self) -> bool:
        return self.config.get("expand_contractions", True)

    @property
    def should_remove_symbols(self) -> bool:
        return self.config.get("remove_symbols", False)

    @property
    def should_remove_accents(self) -> bool:
        return self.config.get("remove_accents", False)

    @property
    def should_remove_articles(self) -> bool:
        return self.config.get("remove_articles", False)

    @property
    def should_remove_stopwords(self) -> bool:
        return self.config.get("remove_stopwords", False)

    @property
    def contractions(self) -> Dict[str, str]:
        return self.config.get("contractions", {})

    @property
    def word_replacements(self) -> Dict[str, str]:
        return self.config.get("word_replacements", {})

    @property
    def number_replacements(self) -> Dict[str, str]:
        return self.config.get("number_replacements", {})

    @property
    def accents(self) -> Dict[str, str]:
        return self.config.get("accents",
                               {"á": "a", "à": "a", "ã": "a", "â": "a",
                                "é": "e", "è": "e", "ê": "e", "ẽ": "e",
                                "í": "i", "ì": "i", "î": "i", "ĩ": "i",
                                "ò": "o", "ó": "o", "ô": "o", "õ": "o",
                                "ú": "u", "ù": "u", "û": "u", "ũ": "u",
                                "Á": "A", "À": "A", "Ã": "A", "Â": "A",
                                "É": "E", "È": "E", "Ê": "E", "Ẽ": "E",
                                "Í": "I", "Ì": "I", "Î": "I", "Ĩ": "I",
                                "Ò": "O", "Ó": "O", "Ô": "O", "Õ": "O",
                                "Ú": "U", "Ù": "U", "Û": "U", "Ũ": "U"
                                })

    @property
    def stopwords(self) -> List[str]:
        return self.config.get("stopwords", [])

    @property
    def articles(self) -> List[str]:
        return self.config.get("articles", [])

    @property
    def symbols(self) -> List[str]:
        return self.config.get("symbols",
                               [";", "_", "!", "?", "<", ">", "|",
                                "(", ")", "=", "[", "]", "{", "}",
                                "»", "«", "*", "~", "^", "`", "\""])

    def expand_contractions(self, utterance: str) -> str:
        """ Expand common contractions, e.g. "isn't" -> "is not" """
        words = self.tokenize(utterance)
        for idx, w in enumerate(words):
            if w in self.contractions:
                words[idx] = self.contractions[w]
        utterance = " ".join(words)
        return utterance

    def numbers_to_digits(self, utterance: str) -> str:
        words = self.tokenize(utterance)
        for idx, w in enumerate(words):
            if w in self.number_replacements:
                words[idx] = self.number_replacements[w]
        utterance = " ".join(words)
        return utterance

    def remove_articles(self, utterance: str) -> str:
        words = self.tokenize(utterance)
        for idx, w in enumerate(words):
            if w in self.articles:
                words[idx] = ""
        utterance = " ".join(words)
        return utterance

    def remove_stopwords(self, utterance: str) -> str:
        words = self.tokenize(utterance)
        for idx, w in enumerate(words):
            if w in self.stopwords:
                words[idx] = ""
        # if words[-1] == '-':
        #    words = words[:-1]
        utterance = " ".join(words)
        # Remove trailing whitespaces from utterance along with orphaned
        # hyphens, more characters may be added later
        utterance = re.sub(r'- *$', '', utterance)
        return utterance

    def remove_symbols(self, utterance: str) -> str:
        mapping = str.maketrans('', '', "".join(self.symbols))
        return utterance.translate(mapping)

    def remove_accents(self, utterance : str) -> str:
        for s in self.accents:
            utterance = utterance.replace(s, self.accents[s])
        return utterance

    def replace_words(self, utterance: str) -> str:
        words = self.tokenize(utterance)
        for idx, w in enumerate(words):
            if w in self.word_replacements:
                words[idx] = self.word_replacements[w]
        utterance = " ".join(words)
        return utterance

    def normalize(self, utterance: str = "", remove_articles: bool = None):
        # mutations
        if self.should_lowercase:
            utterance = utterance.lower()
        if self.should_expand_contractions:
            utterance = self.expand_contractions(utterance)
        if self.should_numbers_to_digits:
            utterance = self.numbers_to_digits(utterance)
        utterance = self.replace_words(utterance)

        # removals
        if self.should_remove_symbols:
            utterance = self.remove_symbols(utterance)
        if self.should_remove_accents:
            utterance = self.remove_accents(utterance)
        # TODO deprecate remove_articles param, backwards compat
        if remove_articles is not None and remove_articles:
            utterance = self.remove_articles(utterance)
        elif self.should_remove_articles:
            utterance = self.remove_articles(utterance)
        if self.should_remove_stopwords:
            utterance = self.remove_stopwords(utterance)
        # remove extra spaces
        utterance = " ".join([w for w in utterance.split(" ") if w])
        return utterance


class CatalanNormalizer(Normalizer):
    with open(f"{dirname(dirname(__file__))}/res/ca/normalize.json") as f:
        _default_config = json.load(f)

    @staticmethod
    def tokenize(utterance : str) -> List[str]:
        return word_tokenize(utterance, lang="ca")


class CzechNormalizer(Normalizer):
    with open(f"{dirname(dirname(__file__))}/res/cz/normalize.json", encoding='utf8') as f:
        _default_config = json.load(f)


class PortugueseNormalizer(Normalizer):
    with open(f"{dirname(dirname(__file__))}/res/pt/normalize.json") as f:
        _default_config = json.load(f)

    @staticmethod
    def tokenize(utterance: str) -> List[str]:
        return word_tokenize(utterance, lang="pt")


class RussianNormalizer(Normalizer):
    with open(f"{dirname(dirname(__file__))}/res/ru/normalize.json", encoding='utf8') as f:
        _default_config = json.load(f)


class UkrainianNormalizer(Normalizer):
    with open(f"{dirname(dirname(__file__))}/res/uk/normalize.json", encoding='utf8') as f:
        _default_config = json.load(f)


class EnglishNormalizer(Normalizer):
    with open(f"{dirname(dirname(__file__))}/res/en/normalize.json") as f:
        _default_config = json.load(f)

    def numbers_to_digits(self, utterance: str) -> str:
        return EnglishNumberParser().convert_words_to_numbers(utterance)


class AzerbaijaniNormalizer(Normalizer):
    with open(f"{dirname(dirname(__file__))}/res/az/normalize.json") as f:
        _default_config = json.load(f)

    def numbers_to_digits(self, utterance: str) -> str:
        return AzerbaijaniNumberParser().convert_words_to_numbers(utterance)


class GermanNormalizer(Normalizer):
    with open(f"{dirname(dirname(__file__))}/res/de/normalize.json") as f:
        _default_config = json.load(f)

    def numbers_to_digits(self, utterance: str) -> str:
        return GermanNumberParser().convert_words_to_numbers(utterance)
    
    def remove_symbols(self, utterance: str) -> str:
        # special rule for hyphanated words in german as some STT engines falsely
        # return them pretty regularly
        utterance = re.sub(r'\b(\w*)-(\w*)\b', r'\1 \2', utterance)
        return super().remove_symbols(utterance)
