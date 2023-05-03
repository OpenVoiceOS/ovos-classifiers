from typing import Optional, List

from ovos_plugin_manager.templates.transformers import UtteranceTransformer
from ovos_classifiers.corefiob import OVOSCorefIOBTagger
from ovos_classifiers.postag import OVOSPostag
from ovos_classifiers.heuristics.normalize import Normalizer, CatalanNormalizer, CzechNormalizer, \
    PortugueseNormalizer, AzerbaijaniNormalizer, RussianNormalizer, EnglishNormalizer, UkrainianNormalizer


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
        norm = [normalizer.normalize(u) for u in utterances]
        norm = [self.strip_punctuation(u) for u in norm]
        return list(set(norm + utterances)), context


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

        for u in set(utterances):
            pos = post.postag(u)
            iob = tagger.iob_tag(pos)
            utterances += tagger.normalize_corefs([iob])

        return list(set(utterances)), context


if __name__ == "__main__":
    u, _ = UtteranceNormalizer().transform(["Mom is awesome, she said she loves me!"])
    print(u) # ['Mom is awesome , she said she loves me', 'Mom is awesome, she said she loves me']
    u, _ = CoreferenceNormalizer().transform(u) #
    print(u) # ['Mom is awesome , Mom said Mom loves me', 'Mom is awesome , she said she loves me', 'Mom is awesome, she said she loves me']
