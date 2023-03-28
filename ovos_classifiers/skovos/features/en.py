import re

import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from ovos_classifiers.utils import normalize


class WordNetLemmatizerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize(X, stemmer=nltk.stem.WordNetLemmatizer(),
                         **transform_params)


class QuestionFeaturesTransformerEN(BaseEstimator, TransformerMixin):
    # begin of postagged_tokens indicators for Yes/No questions
    YES_NO_STARTERS = ["would", "is", "will", "does", "can", "has", "if",
                       "could", "are", "should", "have", "has", "did"]

    # begin of postagged_tokens indicators for "command" questions, eg, "do this"
    # non exhaustive list, should capture common voice interactions
    COMMAND_STARTERS = [
        "name", "define", "list", "tell", "say"
    ]

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []

        for sent in X:
            first_word = sent.split(" ")[0]
            has_please = "please" in sent.lower()
            has_thank_you = "thank you" in sent.lower()
            has_opinion = any(word in sent.lower()
                              for word in ["like", "love", "hate", "enjoy", "dislike"])
            has_exclamation = any(word in sent.lower()
                                  for word in ["wow", "amazing", "awesome", "great",
                                               "cool", "fantastic", "incredible"])
            has_social = any(sent.startswith(word)
                             for word in ["hello", "hi", "hey", "good",
                                          "morning", "afternoon", "evening", "night", "bye",
                                          "thanks", "thank", "thank you"])
            has_negative = any(word in sent.lower()
                               for word in ["do not", "stop", "leave", "drop", "throw"])
            statement_like = bool(re.match(
                r'^\w+(?:\s+\w+)*\s+[a-z]+(?:s|ed|ing)?(?:\s+\w+(?:s|ed|ing)?)?(?:\s+[a-z]+(?:s|ed|ing)?(?:\s+\w+(?:s|ed|ing)?)?)?$',
                sent))

            s_feature = {
                'is_yes_no': first_word in self.YES_NO_STARTERS,
                'is_wh': sent.startswith('wh'),
                "is_command": first_word in self.COMMAND_STARTERS,
                "is_statement": statement_like,
                "has_please": has_please,
                "has_negative": has_negative,
                "has_thank_you": has_thank_you,
                "has_opinion": has_opinion,
                "has_exclamation": has_exclamation,
                "has_social": has_social
            }
            feats += [s_feature]
        return feats


class QuestionFeaturesVectorizerEN(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transformer = QuestionFeaturesTransformerEN()
        self._vectorizer = DictVectorizer(sparse=False)

    def get_feature_names(self):
        return self._vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        X = self._transformer.transform(X)
        self._vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = self._transformer.transform(X, **transform_params)
        return self._vectorizer.transform(X)
