import unittest

from ovos_plugin_manager.keywords import find_keyword_extract_plugins
from ovos_plugin_manager.solvers import find_question_solver_plugins, find_reading_comprehension_solver_plugins, \
    find_tldr_solver_plugins, find_entailment_solver_plugins, find_multiple_choice_solver_plugins
from ovos_plugin_manager.text_transformers import find_utterance_transformer_plugins
from ovos_plugin_manager.coreference import find_coref_plugins
from ovos_plugin_manager.postag import find_postag_plugins
from ovos_plugin_manager.language import find_lang_detect_plugins


class TestKeywords(unittest.TestCase):
    expected_kw_plugs = ['ovos-keyword-plugin-dummy', "ovos-keyword-extractor-heuristic", "ovos-keyword-extractor-rake"]
    expected_utt_plugs = ['ovos-utterance-normalizer', 'ovos-utterance-coref-normalizer']
    expected_solver_plugs = ['ovos-question-solver-wordnet']
    expected_summ = ["ovos-summarizer-solver-wordfreq"]
    expected_qa = ["ovos-evidence-solver-bm25"]
    expected_coref = ["ovos-coref-solver-heuristic", "ovos-classifiers-coref-solver"]
    expected_post = ["ovos-postag-plugin-regex", "ovos-postag-plugin-nltk",
                     "ovos-classifiers-postag-plugin"]
    expected_lang = ["ovos-lang-detect-ngram-lm"]

    def test_lang(self):
        plugs = list(find_lang_detect_plugins().keys())
        for kw in self.expected_lang:
            self.assertTrue(kw in plugs)

    def test_postag(self):
        plugs = list(find_postag_plugins().keys())
        for kw in self.expected_post:
            self.assertTrue(kw in plugs)

    def test_coref(self):
        plugs = list(find_coref_plugins().keys())
        for kw in self.expected_coref:
            self.assertTrue(kw in plugs)

    def test_qa(self):
        plugs = list(find_reading_comprehension_solver_plugins().keys())
        for kw in self.expected_qa:
            self.assertTrue(kw in plugs)

    def test_summarizer(self):
        plugs = list(find_tldr_solver_plugins().keys())
        for kw in self.expected_summ:
            self.assertTrue(kw in plugs)

    def test_solver(self):
        plugs = list(find_question_solver_plugins().keys())
        for kw in self.expected_solver_plugs:
            self.assertTrue(kw in plugs)

    def test_utt(self):
        plugs = list(find_utterance_transformer_plugins().keys())
        for kw in self.expected_utt_plugs:
            self.assertTrue(kw in plugs)

    def test_kw(self):
        plugs = find_keyword_extract_plugins()
        for kw in self.expected_kw_plugs:
            self.assertTrue(kw in plugs)
