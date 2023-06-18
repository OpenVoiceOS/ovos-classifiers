# these plugins do not have external dependencies and do not download any data
# they should be available in all platforms
import string
from typing import Optional, List

from ovos_plugin_manager.templates.coreference import CoreferenceSolverEngine
from ovos_plugin_manager.templates.g2p import Grapheme2PhonemePlugin
from ovos_plugin_manager.templates.keywords import KeywordExtractor
from ovos_plugin_manager.templates.postag import PosTagger
from ovos_plugin_manager.templates.solvers import TldrSolver, EvidenceSolver
from ovos_plugin_manager.templates.transformers import UtteranceTransformer
from quebra_frases import sentence_tokenize, word_tokenize, span_indexed_word_tokenize

from ovos_classifiers.heuristics.corefiob import CorefIOBHeuristicTagger
from ovos_classifiers.heuristics.keyword_extraction import HeuristicExtractor
from ovos_classifiers.heuristics.machine_comprehension import BM25
from ovos_classifiers.heuristics.normalize import Normalizer, CatalanNormalizer, CzechNormalizer, \
    PortugueseNormalizer, AzerbaijaniNormalizer, RussianNormalizer, EnglishNormalizer, UkrainianNormalizer, \
    GermanNormalizer
from ovos_classifiers.heuristics.phonemizer import EnglishARPAHeuristicPhonemizer
from ovos_classifiers.heuristics.postag import RegexPostag
from ovos_classifiers.heuristics.summarization import WordFrequencySummarizer


class RegexPostagPlugin(PosTagger):
    """very low accuracy regex based postag

    this plugin is meant as a fallback only, when external models can not be downloaded
    """

    def postag(self, spans, lang=None):
        if isinstance(spans, str):
            spans = span_indexed_word_tokenize(spans)
        tagger = RegexPostag({"lang": lang or self.lang})
        tags = tagger.tag(" ".join(t for s, e, t in spans))
        tagged_spans = [(s, e, t, tags[idx][1])
                        for idx, (s, e, t) in enumerate(spans)]
        return tagged_spans


class UtteranceNormalizerPlugin(UtteranceTransformer):
    """plugin to normalize utterances by normalizing numbers, punctuation and contractions
    language specific pre-processing is handled here too
    this helps intent parsers"""

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
        elif lang.startswith("de"):
            return GermanNormalizer()
        return Normalizer()

    @staticmethod
    def strip_punctuation(utterance: str):
        return utterance.strip(string.punctuation).strip()

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


class HeuristicSummarizerPlugin(TldrSolver):
    """heuristic summarizer, picks best sentences based on word frequencies"""

    def get_tldr(self, document, context=None):
        context = context or {}
        lang = context.get("lang") or "en"
        return WordFrequencySummarizer().summarize(document, lang)


class BM25SolverPlugin(EvidenceSolver):
    """extract best sentence from text that answers the question, using BM25 algorithm"""

    def get_best_passage(self, evidence, question, context=None):
        """
        evidence and question assured to be in self.default_lang
         returns summary of provided document
        """
        bm25 = BM25()

        sents = []
        for s in evidence.split("\n"):
            sents += sentence_tokenize(s)
        corpus = [word_tokenize(s) for s in sents]
        bm25.fit(corpus)
        scores = bm25.search(word_tokenize(question))
        ans = max([s for s in zip(scores, corpus)],
                  key=lambda k: k[0])[1]
        return " ".join(ans)


class HeuristicKeywordExtractorPlugin(KeywordExtractor):
    """regex based keyword extractor,
    handles common questions to make search keywords more relevant in downstream tasks"""

    def extract(self, text, lang):
        kw = HeuristicExtractor.extract_subject(text, lang)
        if kw:
            return {kw: 1.0}
        return {text: 0.0}


class HeuristicCoreferenceSolverPlugin(CoreferenceSolverEngine):
    """heuristic coreference solver based on pronoun lookups

    it is recommended to use OVOSCoreferenceSolverPlugin instead with "heuristic" model
    this has the advantage of better postag pipeline

    this plugin is meant as a fallback only, when external models can not be downloaded
    """

    @classmethod
    def solve_corefs(cls, text, lang=None):
        tagger = CorefIOBHeuristicTagger(lang.split("-")[0])
        pos = RegexPostag({"lang": lang}).tag(text)
        iob = [tagger.tag(pos)]
        return UtteranceNormalizerPlugin.strip_punctuation(tagger.normalize_corefs(iob)[0])


class ARPAHeuristicPhonemizerPlugin(Grapheme2PhonemePlugin):

    def get_arpa(self, word, lang="en", ignore_oov=False):
        phones = EnglishARPAHeuristicPhonemizer.phonemize(word)
        return phones

    def get_ipa(self, word, lang="en", ignore_oov=False):
        # just not requiring lang arg
        return super().get_ipa(word, lang, ignore_oov)

    def utterance2arpa(self, utterance, lang="en", ignore_oov=False):
        # just not requiring lang arg
        return super().utterance2arpa(utterance, lang, ignore_oov)

    def utterance2ipa(self, utterance, lang="en", ignore_oov=False):
        # just not requiring lang arg
        return super().utterance2ipa(utterance, lang, ignore_oov)

    @staticmethod
    def get_languages():
        return {'en'}

    @property
    def available_languages(self):
        """Return languages supported by this G2P implementation in this state
        This property should be overridden by the derived class to advertise
        what languages that engine supports.
        Returns:
            set: supported languages
        """
        return self.get_languages()


if __name__ == "__main__":
    pho = ARPAHeuristicPhonemizerPlugin()
    pho.utterance2arpa("hello world")
    # ['HH', 'EH', 'L', 'L', 'OW', '.', 'W', 'OW', 'R', 'L', 'D']
    pho.utterance2ipa("hello world")
    # ['h', 'ɛ', 'l', 'l', 'oʊ', '.', 'w', 'oʊ', 'ɹ', 'l', 'd']

    print(RegexPostagPlugin().postag("I like pizza", "en"))
    # [(0, 1, 'I', 'PRON'), (2, 6, 'like', 'VERB'), (7, 12, 'pizza', 'NOUN')]

    coref = HeuristicCoreferenceSolverPlugin()
    print(coref.solve_corefs("Mom is awesome, she said she loves me!", "en"))

    doc = """
    Introducing OpenVoiceOS - The Free and Open-Source Personal Assistant and Smart Speaker.

    OpenVoiceOS is a new player in the smart speaker market, offering a powerful and flexible alternative to proprietary solutions like Amazon Echo and Google Home.

    With OpenVoiceOS, you have complete control over your personal data and the ability to customize and extend the functionality of your smart speaker.

    Built on open-source software, OpenVoiceOS is designed to provide users with a seamless and intuitive voice interface for controlling their smart home devices, playing music, setting reminders, and much more.

    The platform leverages cutting-edge technology, including machine learning and natural language processing, to deliver a highly responsive and accurate experience.

    In addition to its voice capabilities, OpenVoiceOS features a touch-screen GUI made using QT5 and the KF5 framework.

    The GUI provides an intuitive, user-friendly interface that allows you to access the full range of OpenVoiceOS features and functionality.

    Whether you prefer voice commands or a more traditional touch interface, OpenVoiceOS has you covered.

    One of the key advantages of OpenVoiceOS is its open-source nature, which means that anyone with the technical skills can contribute to the platform and help shape its future.

    Whether you're a software developer, data scientist, or just someone with a passion for technology, you can get involved and help build the next generation of personal assistants and smart speakers.

    With OpenVoiceOS, you have the option to run the platform fully offline, giving you complete control over your data and ensuring that your information is never shared with third parties. This makes OpenVoiceOS the perfect choice for anyone who values privacy and security.

    So if you're looking for a personal assistant and smart speaker that gives you the freedom and control you deserve, be sure to check out OpenVoiceOS today!
    """

    k = HeuristicKeywordExtractorPlugin()
    k.extract("who invented the telephone", "en")  # {'telephone': 1.0}
    k.extract("what is the speed of light", "en")  # {'speed of light': 1.0}

    b = BM25SolverPlugin()
    print(b.get_best_passage(doc, "does OpenVoiceOS run offline"))
    # With OpenVoiceOS , you have the option to run the platform fully offline , giving you complete control over your data and ensuring that your information is never shared with third parties .

    h = HeuristicSummarizerPlugin()
    print(h.tldr(doc, lang="en"))
    #     Built on open-source software, OpenVoiceOS is designed to provide users with a seamless and intuitive voice interface for controlling their smart home devices, playing music, setting reminders, and much more.
    #     Whether you're a software developer, data scientist, or just someone with a passion for technology, you can get involved and help build the next generation of personal assistants and smart speakers.
    #     With OpenVoiceOS, you have complete control over your personal data and the ability to customize and extend the functionality of your smart speaker.
    #     With OpenVoiceOS, you have the option to run the platform fully offline, giving you complete control over your data and ensuring that your information is never shared with third parties.
    #     So if you're looking for a personal assistant and smart speaker that gives you the freedom and control you deserve, be sure to check out OpenVoiceOS today!
    #     One of the key advantages of OpenVoiceOS is its open-source nature, which means that anyone with the technical skills can contribute to the platform and help shape its future.
    #     OpenVoiceOS is a new player in the smart speaker market, offering a powerful and flexible alternative to proprietary solutions like Amazon Echo and Google Home.

    u, _ = UtteranceNormalizerPlugin().transform(["Mom is awesome, she said she loves me!"])
    print(u)
    # ['Mom is awesome, she said she loves me!', 'Mom is awesome , she said she loves me']
