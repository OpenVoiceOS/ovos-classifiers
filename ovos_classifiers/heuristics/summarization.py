from ovos_classifiers.heuristics.tokenize import word_tokenize
from quebra_frases import sentence_tokenize, paragraph_tokenize
from heapq import nlargest
from nltk.corpus import stopwords
import nltk
from string import punctuation


class WordFrequencySummarizer:
    def __init__(self):
        nltk.download("stopwords")
        self.langs = {
            "en": "english",
            "ar": "arabic",
            "az": "azerbaijani",
            "ca": "catalan",
            "eu": "basque",
            "da": "danish",
            "de": "german",
            "nl": "dutch",
            "fi": "finnish",
            "fr": "french",
            "hu": "hungarian",
            "it": "italian",
            "no": "norwegian",
            "pt": "portuguese",
            "ru": "russian",
            "es": "spanish",
            "sw": "swedish",
            "ro": "romanian"
        }

    def summarize(self, document, lang="en"):
        lang = lang.split("-")[0].lower()
        lang = self.langs.get(lang) or lang

        stop_words = stopwords.words(lang)

        tokens = word_tokenize(document, lang)

        word_frequencies = {}
        for word in tokens:
            if word.lower() not in stop_words:
                if word.lower() not in punctuation:
                    if word not in word_frequencies.keys():
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency

        sentence_scores = {}
        sentence_candidates = []
        paragraphs = document.split("\n")
        for p in paragraphs:
            if not p:
                continue
            sentence_candidates += sentence_tokenize(p)

        for sent in sentence_candidates:
            sentence = word_tokenize(sent)
            for word in sentence:
                if word.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.lower()]

        select_length = int(len(sentence_candidates) * 0.3)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        final_summary = [word for word in summary]
        return '\n'.join(final_summary) or document


