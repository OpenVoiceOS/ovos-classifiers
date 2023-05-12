import random
from typing import Optional, List

from nltk.corpus import wordnet as wn
from ovos_classifiers.corefiob import OVOSCorefIOBTagger
from ovos_classifiers.datasets.wordnet import Wordnet
from ovos_classifiers.heuristics.normalize import Normalizer, CatalanNormalizer, CzechNormalizer, \
    PortugueseNormalizer, AzerbaijaniNormalizer, RussianNormalizer, EnglishNormalizer, UkrainianNormalizer
from ovos_classifiers.postag import OVOSPostag

from ovos_plugin_manager.templates.solvers import QuestionSolver
from ovos_plugin_manager.templates.transformers import UtteranceTransformer


class UtteranceNormalizer(UtteranceTransformer):

    def __init__(self, name="ovos-utterance-normalizer", priority=1):
        super().__init__(name, priority)

    @staticmethod
    def get_normalizer(lang: str):
        if lang.startswith("en"):
            return EnglishNormalizer()
        elif lang.startswith("pt"):
            return PortugueseNormalizer()
        elif lang.startswith("uk"):
            return UkrainianNormalizer()
        elif lang.startswith("ca"):
            return CatalanNormalizer()
        elif lang.startswith("cz"):
            return CzechNormalizer()
        elif lang.startswith("az"):
            return AzerbaijaniNormalizer()
        elif lang.startswith("ru"):
            return RussianNormalizer()
        return Normalizer()

    @staticmethod
    def strip_punctuation(utterance: str):
        return utterance.rstrip('.').rstrip('?').rstrip('!').rstrip(',').rstrip(';').strip()

    def transform(self, utterances: List[str],
                  context: Optional[dict] = None) -> (list, dict):
        context = context or {}
        lang = context.get("lang") or self.config.get("lang", "en-us")
        normalizer = self.get_normalizer(lang)
        norm = []
        for u in utterances:
            norm.append(u)
            norm.append(normalizer.normalize(u))
            norm.append(normalizer.normalize(u, remove_articles=True))
        norm = [self.strip_punctuation(u) for u in norm]
        # this deduplicates the list while keeping order
        return list(dict.fromkeys(norm)), context


class CoreferenceNormalizer(UtteranceTransformer):

    def __init__(self, name="ovos-utterance-coref-normalizer", priority=3):
        super().__init__(name, priority)

    @staticmethod
    def get_normalizer(lang: str):
        tagger = OVOSCorefIOBTagger(lang.split("-")[0])
        post = OVOSPostag(lang=lang)
        return tagger, post

    def transform(self, utterances: List[str],
                  context: Optional[dict] = None) -> (list, dict):
        context = context or {}
        lang = context.get("lang") or self.config.get("lang", "en-us")
        tagger, post = self.get_normalizer(lang)

        def _coref(u):
            pos = post.postag(u)
            iob = tagger.iob_tag(pos)
            return UtteranceNormalizer.strip_punctuation(tagger.normalize_corefs([iob])[0])

        norm = []
        for u in utterances:
            norm.append(_coref(u))
            norm.append(u)

        # this deduplicates the list while keeping order
        return list(dict.fromkeys(norm)), context


class WordnetSolver(QuestionSolver):
    enable_tx = True
    priority = 80

    def __init__(self, config=None):
        config = config or {}
        config["lang"] = "en"  # only english supported
        super(WordnetSolver, self).__init__(config)

    def extract_keyword(self, query, lang="en"):
        query = query.lower()

        # regex from narrow to broader matches
        match = None
        if lang == "en":
            # TODO - keyword extractor class / localization
            starts = ["who is ", "what is ", "when is ", "tell me about "]
            for s in starts:
                if query.startswith(s):
                    return query.split(s)[-1]
        return None

    def get_data_key(self, query, lang="en"):
        # TODO localization
        if lang == "en":
            query = self.extract_keyword(query, lang) or query
            words = query.split()

            stop_words = ["the", "on", "of", "in", "a", "is", "what", "when", "for", "an", "at"]
            if any("antonym" in w for w in words):
                query = " ".join(w for w in words
                                 if "antonym" not in w
                                 and w not in stop_words)
                return "antonyms", query

            if any("synonym" in w for w in words):
                query = " ".join(w for w in words
                                 if "synonym" not in w
                                 and w not in stop_words)
                return "lemmas", query

            if any("definition" in w or "meaning" in w for w in words) or words[0] == "what":
                query = " ".join(w for w in words
                                 if "definition" not in w and "meaning" not in w
                                 and w not in stop_words)
                return "definition", query

        return None, query

    # officially exported Solver methods
    def get_data(self, query, context=None):
        pos = wn.NOUN  # TODO check context for postag
        synsets = wn.synsets(query, pos=pos)
        if not len(synsets):
            return {}
        synset = synsets[0]
        res = {"lemmas": Wordnet.get_lemmas(query, pos=pos, synset=synset),
               "antonyms": Wordnet.get_antonyms(query, pos=pos, synset=synset),
               "holonyms": Wordnet.get_holonyms(query, pos=pos, synset=synset),
               "hyponyms": Wordnet.get_hyponyms(query, pos=pos, synset=synset),
               "hypernyms": Wordnet.get_hypernyms(query, pos=pos, synset=synset),
               "root_hypernyms": Wordnet.get_root_hypernyms(query, pos=pos, synset=synset),
               "definition": Wordnet.get_definition(query, pos=pos, synset=synset)}
        return res

    def get_spoken_answer(self, query, context=None):
        lang = context.get("lang") or self.default_lang
        lang = lang.split("-")[0]
        # extract the best keyword with some regexes or fallback to RAKE
        k, query = self.get_data_key(query, lang)
        if not query:
            query = self.extract_keyword(query, lang) or query
        data = self.search(query, context)
        if k and k in data:
            v = data[k]
            if k in ["lemmas", "antonyms"] and len(v):
                return random.choice(v)
            if isinstance(v, list) and len(v):
                v = v[0]
            if isinstance(v, str):
                return v
        # definition
        return data.get("definition")


if __name__ == "__main__":
    d = WordnetSolver()
    sentence = d.spoken_answer("what is the definition of computer")
    print(sentence)
    # a machine for performing calculations automatically

    d = WordnetSolver()
    sentence = d.spoken_answer("qual é a definição de computador", lang="pt")
    print(sentence)
    # uma máquina para realizar cálculos automaticamente

    u, _ = UtteranceNormalizer().transform(["Mom is awesome, she said she loves me!"])
    print(u)  # ['Mom is awesome, she said she loves me!', 'Mom is awesome , she said she loves me']
    u, _ = CoreferenceNormalizer().transform(u)  #
    print(u)  # ['Mom is awesome , Mom said Mom loves me', 'Mom is awesome, she said she loves me!', 'Mom is awesome , she said she loves me']