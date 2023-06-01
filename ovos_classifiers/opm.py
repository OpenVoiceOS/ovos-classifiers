import random
from typing import Optional, List

from nltk.corpus import wordnet as wn
from ovos_plugin_manager.templates.coreference import CoreferenceSolverEngine
from ovos_plugin_manager.templates.keywords import KeywordExtractor
from ovos_plugin_manager.templates.language import LanguageDetector
from ovos_plugin_manager.templates.postag import PosTagger
from ovos_plugin_manager.templates.solvers import QuestionSolver, TldrSolver, EvidenceSolver
from ovos_plugin_manager.templates.transformers import UtteranceTransformer
from quebra_frases import sentence_tokenize, word_tokenize, span_indexed_word_tokenize

from ovos_classifiers.corefiob import OVOSCorefIOBTagger
from ovos_classifiers.datasets.wordnet import Wordnet
from ovos_classifiers.heuristics.keyword_extraction import Rake, HeuristicExtractor
from ovos_classifiers.heuristics.lang_detect import LMLangClassifier
from ovos_classifiers.heuristics.machine_comprehension import BM25
from ovos_classifiers.heuristics.normalize import Normalizer, CatalanNormalizer, CzechNormalizer, \
    PortugueseNormalizer, AzerbaijaniNormalizer, RussianNormalizer, EnglishNormalizer, UkrainianNormalizer, \
    GermanNormalizer
from ovos_classifiers.heuristics.summarization import HeuristicSummarizer
from ovos_classifiers.postag import OVOSPostag


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
        elif lang.startswith("de"):
            return GermanNormalizer()
        return Normalizer()

    @staticmethod
    def strip_punctuation(utterance: str):
        return utterance.lstrip('"').rstrip('"').rstrip('.').rstrip('?')\
                .rstrip('!').rstrip(',').rstrip(';').strip()

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

    def transform(self, utterances: List[str],
                  context: Optional[dict] = None) -> (list, dict):
        context = context or {}
        lang = context.get("lang") or self.config.get("lang", "en-us")

        norm = []
        for u in utterances:
            norm += [HeuristicCoreferenceSolver.solve_corefs(u, lang), u]

        # this deduplicates the list while keeping order
        return list(dict.fromkeys(norm)), context


class WordnetSolver(QuestionSolver):
    enable_tx = True
    priority = 80

    def __init__(self, config=None):
        config = config or {}
        config["lang"] = "en"  # only english supported
        super(WordnetSolver, self).__init__(config)

    def get_data_key(self, query, lang="en"):
        query = HeuristicExtractor.extract_subject(query, lang) or query

        # TODO localization
        if lang == "en":
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
            query = HeuristicExtractor.extract_subject(query, lang) or query
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


class NltkSummarizer(TldrSolver):

    def get_tldr(self, document, context=None):
        context = context or {}
        lang = context.get("lang") or "en"
        return HeuristicSummarizer().summarize(document, lang)


class BM25Solver(EvidenceSolver):

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


class RakeExtractor(KeywordExtractor):
    def extract(self, text, lang):
        r = Rake()
        r.extract_keywords_from_text(text)
        scores = r.get_ranked_phrases_with_scores()
        total = sum(s[0] for s in scores)
        scores = {k: v / total for v, k in scores}
        return scores


class HeuristicKeywordExtractor(KeywordExtractor):
    def extract(self, text, lang):
        kw = HeuristicExtractor.extract_subject(text, lang)
        if kw:
            return {kw: 1.0}
        return {text: 0.0}


class HeuristicCoreferenceSolver(CoreferenceSolverEngine):

    @classmethod
    def solve_corefs(cls, text, lang=None):
        tagger = OVOSCorefIOBTagger(lang.split("-")[0])
        pos = OVOSPostag(lang=lang).postag(text)
        iob = [tagger.iob_tag(pos)]
        return UtteranceNormalizer.strip_punctuation(tagger.normalize_corefs(iob)[0])


class OVOSPostagPlugin(PosTagger):
    def postag(self, spans, lang=None):
        if isinstance(spans, str):
            spans = span_indexed_word_tokenize(spans)
        tagger = OVOSPostag(lang=lang)
        tags = tagger.postag(" ".join(t for s, e, t in spans))
        tagged_spans = [(s, e, t, tags[idx][1])
                        for idx, (s, e, t) in enumerate(spans)]
        return tagged_spans


class LMLangDetectPlugin(LanguageDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf = LMLangClassifier()

    def detect(self, text):
        return self.clf.identify_language(text)

    def detect_probs(self, text):
        return self.clf.predict(text)


if __name__ == "__main__":
    print(OVOSPostagPlugin().postag("I like pizza", "en"))
    # [(0, 1, 'I', 'PRON'), (2, 6, 'like', 'VERB'), (7, 12, 'pizza', 'NOUN')]

    coref = HeuristicCoreferenceSolver()
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

    k = HeuristicKeywordExtractor()
    k.extract("who invented the telephone", "en")  # {'telephone': 1.0}
    k.extract("what is the speed of light", "en")  # {'speed of light': 1.0}

    k = RakeExtractor()
    k.extract("who invented the telephone", "en")  # {'telephone': 0.5, 'invented': 0.5}
    k.extract("what is the speed of light", "en")  # {'speed': 0.5, 'light': 0.5}

    b = BM25Solver()
    print(b.get_best_passage(doc, "does OpenVoiceOS run offline"))
    # With OpenVoiceOS , you have the option to run the platform fully offline , giving you complete control over your data and ensuring that your information is never shared with third parties .

    h = NltkSummarizer()
    print(h.tldr(doc, lang="en"))
    #     Built on open-source software, OpenVoiceOS is designed to provide users with a seamless and intuitive voice interface for controlling their smart home devices, playing music, setting reminders, and much more.
    #     Whether you're a software developer, data scientist, or just someone with a passion for technology, you can get involved and help build the next generation of personal assistants and smart speakers.
    #     With OpenVoiceOS, you have complete control over your personal data and the ability to customize and extend the functionality of your smart speaker.
    #     With OpenVoiceOS, you have the option to run the platform fully offline, giving you complete control over your data and ensuring that your information is never shared with third parties.
    #     So if you're looking for a personal assistant and smart speaker that gives you the freedom and control you deserve, be sure to check out OpenVoiceOS today!
    #     One of the key advantages of OpenVoiceOS is its open-source nature, which means that anyone with the technical skills can contribute to the platform and help shape its future.
    #     OpenVoiceOS is a new player in the smart speaker market, offering a powerful and flexible alternative to proprietary solutions like Amazon Echo and Google Home.

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
    print(
        u)  # ['Mom is awesome , Mom said Mom loves me', 'Mom is awesome, she said she loves me!', 'Mom is awesome , she said she loves me']
