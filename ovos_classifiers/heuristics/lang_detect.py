import collections
import math
import os
import pickle
import typing

import nltk
from nltk.corpus import udhr
from ovos_utils.xdg_utils import xdg_data_home


class LMLangClassifier:
    def __init__(self, path=None):
        if path:
            with open(path, "rb") as f:
                self.language_models = pickle.load(f)
            print(f"lang models loaded from {path}")
        else:
            self.fit()

    def fit(self, save=True):
        model = f"{xdg_data_home()}/ovos-classifiers/lang_lms.pkl"
        os.makedirs(os.path.dirname(model), exist_ok=True)
        if os.path.isfile(model):
            with open(model, "rb") as f:
                self.language_models = pickle.load(f)
            print(f"lang models loaded from {model}")
            return model

        nltk.download('udhr')  # udhr = Universal Declaration of Human Rights
        languages = ['en', 'de', 'nl', 'fr', 'it', 'es', "pt", "no", "ca", "da", "fi", "sw"]
        language_ids = ['English-Latin1', 'German_Deutsch-Latin1', 'Dutch_Nederlands-Latin1', 'French_Francais-Latin1',
                        'Italian_Italiano-Latin1', 'Spanish_Espanol-Latin1', 'Portuguese_Portugues-Latin1',
                        'Norwegian-Latin1', "Catalan-Latin1", 'Danish_Dansk-Latin1', 'Finnish_Suomi-Latin1',
                        'Swedish_Svenska-Latin1']

        raw_texts = {language: udhr.raw(language_id) for language, language_id in zip(languages, language_ids)}

        self.language_models = {language: self.build_model(text=raw_texts[language], n_vals=range(1, 4)) for language in
                                languages}
        if save:
            with open(model, "wb") as f:
                pickle.dump(self.language_models, f)
            print(f"lang models saved to {model}")
        return model

    @staticmethod
    def calculate_cosine(a: typing.Dict[str, float], b: typing.Dict[str, float]) -> float:
        """
        Calculate the cosine between two numeric vectors
        Params:
            a, b: two dictionaries containing items and their corresponding numeric values
            (e.g. ngrams and their corresponding probabilities)
        """
        numerator = sum([a[k] * b[k] for k in a if k in b])
        denominator = (math.sqrt(sum([a[k] ** 2 for k in a])) * math.sqrt(sum([b[k] ** 2 for k in b])))
        return numerator / denominator

    @staticmethod
    def extract_xgrams(text: str, n_vals: typing.List[int]) -> typing.List[str]:
        """
        Extract a list of n-grams of different sizes from a text.
        Params:
            text: the test from which to extract ngrams
            n_vals: the sizes of n-grams to extract
            (e.g. [1, 2, 3] will produce uni-, bi- and tri-grams)
        """
        xgrams = []

        for n in n_vals:
            # if n > len(text) then no ngrams will fit, and we would return an empty list
            if n < len(text):
                for i in range(len(text) - n + 1):
                    ng = text[i:i + n]
                    xgrams.append(ng)

        return xgrams

    @classmethod
    def build_model(cls, text: str, n_vals=range(1, 4)) -> typing.Dict[str, int]:
        """
        Build a simple model of probabilities of xgrams of various lengths in a text
        Parms:
            text: the text from which to extract the n_grams
            n_vals: a list of n_gram sizes to extract
        Returns:
            A dictionary of ngrams and their probabilities given the input text
        """
        model = collections.Counter(cls.extract_xgrams(text, n_vals))
        num_ngrams = sum(model.values())

        for ng in model:
            model[ng] = model[ng] / num_ngrams

        return model

    def identify_language(self,
                          text: str,
                          n_vals=range(1, 4)
                          ) -> str:
        scores = self.predict(text, n_vals)
        return max(scores.items(), key=lambda k: k[1])[0]

    def predict(self,
                text: str,
                n_vals=range(1, 4)
                ) -> str:
        """
        Given a text and a dictionary of language models, return the language model
        whose ngram probabilities best match those of the test text
        Params:
            text: the text whose language we want to identify
            language_models: a Dict of Dicts, where each key is a language name and
            each value is a dictionary of ngram: probability pairs
            n_vals: a list of n_gram sizes to extract to build a model of the test
            text; ideally reflect the n_gram sizes used in 'language_models'
        """
        text_model = self.build_model(text, n_vals)
        scores = {m: self.calculate_cosine(self.language_models[m], text_model)
                  for m in self.language_models}
        return scores


if __name__ == "__main__":
    clf = LMLangClassifier()
    text = "I was taught that the way of progress was neither swift nor easy.".lower()
    # Quote from Marie Curie, the first woman to win a Nobel Prize, the only woman to win it twice, and the only human to win it in two different sciences.

    print(f"Test text: {text}")
    print(f"Identified language: {clf.identify_language(text, n_vals=range(1, 4))}")
    # Test text: i was taught that the way of progress was neither swift nor easy.
    # Identified language: english
