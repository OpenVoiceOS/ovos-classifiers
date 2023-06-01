# these plugins will load ovos-classifiers trained models
# the best available model or heuristic will be used for each language
# config for model settings comes from ovos-config under classifiers section

from quebra_frases import span_indexed_word_tokenize

from ovos_classifiers.opm.heuristics import UtteranceNormalizerPlugin
from ovos_classifiers.corefiob import OVOSCorefIOBTagger
from ovos_classifiers.postag import OVOSPostag
from typing import Optional, List

from ovos_plugin_manager.templates.coreference import CoreferenceSolverEngine
from ovos_plugin_manager.templates.postag import PosTagger
from ovos_plugin_manager.templates.transformers import UtteranceTransformer


class OVOSPostagPlugin(PosTagger):
    """postag via models trained in ovos-classifiers"""

    def postag(self, spans, lang=None):
        if isinstance(spans, str):
            spans = span_indexed_word_tokenize(spans)
        tagger = OVOSPostag(lang=lang)  # tagger model from ovos-config
        tags = tagger.postag(" ".join(t for s, e, t in spans))
        tagged_spans = [(s, e, t, tags[idx][1])
                        for idx, (s, e, t) in enumerate(spans)]
        return tagged_spans


class OVOSCoreferenceSolverPlugin(CoreferenceSolverEngine):
    """coreference solver via models trained in ovos-classifiers"""

    @classmethod
    def solve_corefs(cls, text, lang=None):
        tagger = OVOSCorefIOBTagger(lang.split("-")[0])  # tagger model from ovos-config
        pos = OVOSPostag(lang=lang).postag(text)  # tagger model from ovos-config
        iob = [tagger.iob_tag(pos)]
        return UtteranceNormalizerPlugin.strip_punctuation(tagger.normalize_corefs(iob)[0])


class CoreferenceNormalizerPlugin(UtteranceTransformer):
    """plugin to normalize utterances by replacing pronoun coreferences
    this helps intent parsers"""

    def __init__(self, name="ovos-utterance-coref-normalizer", priority=3):
        super().__init__(name, priority)

    def transform(self, utterances: List[str],
                  context: Optional[dict] = None) -> (list, dict):
        context = context or {}
        lang = context.get("lang") or self.config.get("lang", "en-us")

        norm = []
        for u in utterances:
            norm += [OVOSCoreferenceSolverPlugin.solve_corefs(u, lang), u]

        # this deduplicates the list while keeping order
        return list(dict.fromkeys(norm)), context


if __name__ == "__main__":
    print(OVOSPostagPlugin().postag("I like pizza", "en"))
    # [(0, 1, 'I', 'PRON'), (2, 6, 'like', 'VERB'), (7, 12, 'pizza', 'NOUN')]

    coref = OVOSCoreferenceSolverPlugin()
    print(coref.solve_corefs("Mom is awesome, she said she loves me!", "en"))

    u, _ = UtteranceNormalizerPlugin().transform(["Mom is awesome, she said she loves me!"])
    print(u)
    # ['Mom is awesome, she said she loves me!', 'Mom is awesome , she said she loves me']
    u, _ = CoreferenceNormalizerPlugin().transform(u)  #
    print(u)
    # ['Mom is awesome , Mom said Mom loves me', 'Mom is awesome, she said she loves me!', 'Mom is awesome , she said she loves me']

