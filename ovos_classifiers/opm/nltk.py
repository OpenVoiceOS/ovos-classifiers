# these plugins require nltk and may download external data/models at runtime
import random

from nltk import pos_tag as _pt
from nltk.corpus import wordnet as wn
from ovos_plugin_manager.templates.language import LanguageDetector
from ovos_plugin_manager.templates.postag import PosTagger
from ovos_plugin_manager.templates.solvers import QuestionSolver
from ovos_plugin_manager.templates.keywords import KeywordExtractor
from quebra_frases import span_indexed_word_tokenize

from ovos_classifiers.datasets.wordnet import Wordnet
from ovos_classifiers.heuristics.keyword_extraction import Rake, HeuristicExtractor
from ovos_classifiers.heuristics.lang_detect import LMLangClassifier


class WordnetSolverPlugin(QuestionSolver):
    """ question answerer that uses wordnet for definitions synonyms and antonyms"""
    enable_tx = True
    priority = 80

    def __init__(self, config=None):
        config = config or {}
        config["lang"] = "en"  # only english supported
        super().__init__(config)

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


class LMLangDetectPlugin(LanguageDetector):
    """language detector that uses a language model via nltk corpus to id language"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf = LMLangClassifier()

    def detect(self, text):
        return self.clf.identify_language(text)

    def detect_probs(self, text):
        return self.clf.predict(text)


class NltkPostagPlugin(PosTagger):
    """postag via nltk pretrained model"""

    @property
    def tagset(self):
        return self.config.get("tagset") or "universal"

    def postag(self, spans, lang=None):
        lang = lang or self.lang  # TODO - ensure lang string for nltk
        if isinstance(spans, str):
            spans = span_indexed_word_tokenize(spans)
        tags = _pt([t for s, e, t in spans],
                   tagset=self.tagset,
                   lang=lang)
        tagged_spans = [(s, e, t, tags[idx][1])
                        for idx, (s, e, t) in enumerate(spans)]
        return tagged_spans


class RakeExtractorPlugin(KeywordExtractor):
    """implementation of the classic RAKE algorithm for keyword extraction"""

    def extract(self, text, lang):
        r = Rake()
        r.extract_keywords_from_text(text)
        scores = r.get_ranked_phrases_with_scores()
        total = sum(s[0] for s in scores)
        scores = {k: v / total for v, k in scores}
        return scores


if __name__ == "__main__":
    print(NltkPostagPlugin().postag("I like pizza", "en"))
    # [(0, 1, 'I', 'PRON'), (2, 6, 'like', 'VERB'), (7, 12, 'pizza', 'NOUN')]

    k = RakeExtractorPlugin()
    k.extract("who invented the telephone", "en")  # {'telephone': 0.5, 'invented': 0.5}
    k.extract("what is the speed of light", "en")  # {'speed': 0.5, 'light': 0.5}

    d = WordnetSolverPlugin()
    sentence = d.spoken_answer("what is the definition of computer")
    print(sentence)
    # a machine for performing calculations automatically

    d = WordnetSolverPlugin()
    sentence = d.spoken_answer("qual é a definição de computador", lang="pt")
    print(sentence)
    # uma máquina para realizar cálculos automaticamente
