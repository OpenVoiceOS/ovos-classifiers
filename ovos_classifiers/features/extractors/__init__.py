# feature extraction utils

import nltk
from ovos_config import Configuration
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from ovos_classifiers.corefiob import OVOSCorefIOBTagger
from ovos_classifiers.features.utils import DummyStemmer, extract_postag_features, \
    extract_word_features, normalize, get_stemmer
from ovos_classifiers.postag import OVOSPostag


class SnowballStemmerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lang="en"):
        super().__init__()
        self.stemmer = get_stemmer(lang)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize(X, stemmer=self.stemmer, **transform_params)


class WordFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lang=None, stemmer=None, memory=2):
        super().__init__()
        lang = lang or Configuration().get("lang", "en-us")
        lang = lang.split("-")[0]
        self.stemmer = stemmer or get_stemmer(lang)
        self.memory = memory
        self.tagger = OVOSPostag(lang)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = [
            extract_word_features(X, index)
            for index in range(len(X))]
        return feats


class WordFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang=None, stemmer=None, memory=2):
        self._transformer = WordFeaturesTransformer(lang=lang, stemmer=stemmer, memory=memory)
        self._vectorizer = DictVectorizer(sparse=False)
        super().__init__()

    def get_feature_names(self):
        return self._vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        X = self._transformer.transform(X)
        self._vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = self._transformer.transform(X, **transform_params)
        return self._vectorizer.transform(X)


class POSTaggerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lang="nltk", stemmer=None, memory=2):
        super().__init__()
        self.tagger = OVOSPostag(lang.split("-")[0])
        self.stemmer = stemmer or get_stemmer(lang)
        self.memory = memory

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X = [self.tagger.postag(sent) for sent in X]
        feats = [extract_postag_features(
            X, index, memory=self.memory, stemmer=self.stemmer)
            for index in range(len(X))]
        return feats


class POSTaggerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang=None, stemmer=None, memory=2):
        super().__init__()
        self._pos_transformer = POSTaggerTransformer(lang=lang, stemmer=stemmer, memory=memory)
        self._dict_vectorizer = DictVectorizer()

    def get_feature_names(self):
        return self._dict_vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        X = self._pos_transformer.transform(X)
        self._dict_vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = self._pos_transformer.transform(X, **transform_params)
        return self._dict_vectorizer.transform(X)


class CorefIOBTaggerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lang="en", stemmer=None, memory=2):
        super().__init__()
        self.postagger = OVOSPostag(lang.split("-")[0])
        self.corefiob = OVOSCorefIOBTagger(lang)
        self.stemmer = stemmer or get_stemmer(lang)
        self.memory = memory

    @staticmethod
    def extract_corefiob_features(tokens, index, coreftagger=None, stemmer=None, memory=2):
        """
        `tokens`  = a POS-tagged postagged_tokens [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous predicted IOB tags
        """
        coreftagger = coreftagger or OVOSCorefIOBTagger("heuristic")
        feat_dict = extract_postag_features(tokens, index, stemmer=stemmer,
                                            memory=memory)
        coref_tags = coreftagger.iob_tag(tokens)

        word, ciob = coref_tags[index]

        # update with CorefIOB features
        feat_dict["ciob"] = ciob

        # look ahead N words
        for i in range(1, memory + 1):
            k = "next-" * i
            nextword, nextciob = coref_tags[index + i]
            feat_dict[k + "ciob"] = nextciob

        # look back N words
        for i in range(1, memory + 1):
            k = "prev-" * i
            prevword, prevciob = coref_tags[index - i]
            feat_dict[k + "ciob"] = prevciob

        return feat_dict

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X = [self.postagger.postag(sent) for sent in X]
        feats = [self.extract_corefiob_features(
            X, index, coreftagger=self.corefiob,
            memory=self.memory, stemmer=self.stemmer)
            for index in range(len(X))]
        return feats


class CorefIOBTaggerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang=None, stemmer=None, memory=2):
        super().__init__()
        self._coref_transformer = CorefIOBTaggerTransformer(lang=lang, stemmer=stemmer, memory=memory)
        self._dict_vectorizer = DictVectorizer()

    def get_feature_names(self):
        return self._dict_vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        X = self._coref_transformer.transform(X)
        self._dict_vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = self._coref_transformer.transform(X, **transform_params)
        return self._dict_vectorizer.transform(X)


class PronounTaggerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lang="en", stemmer=None, memory=2):
        super().__init__()
        self.tagger = OVOSPostag(lang.split("-")[0])
        self.stemmer = stemmer or get_stemmer(lang)
        self.memory = memory

    @staticmethod
    def extract_pronoun_features(tokens, index, stemmer=None, memory=2):
        """
        `tokens`  = a POS-tagged postagged_tokens [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        """
        word = tokens[index][0].lower().rstrip("s")
        sent = tokens[:index]

        feat_dict = extract_postag_features(tokens, index, stemmer=stemmer,
                                            memory=memory)

        feat_dict["prev_noun"] = any(t[1] == "NOUN" or t[1] == "PROPN"
                                     for t in sent)

        # Match keywords for male coreferent
        mp = ["he", "him", "his"]
        pp = ["they", "them", "their"]
        fp = ["she", "her", "hers"]
        np = ["it"]
        feat_dict["male_pron"] = word in mp
        feat_dict["prev_male_pron"] = any(t[0] in mp for t in sent)
        feat_dict["neutral_pron"] = word in pp
        feat_dict["prev_neutral_pron"] = any(t[0] in pp for t in sent)
        feat_dict["female_pron"] = word in fp
        feat_dict["prev_female_pron"] = any(t[0] in fp for t in sent)
        feat_dict["inanimate_pron"] = word in np
        feat_dict["prev_inanimate_pron"] = any(t[0] in np for t in sent)

        # Match keywords for male entity
        mwords = ["man", "men", "boy", "guy", "male", "gentleman",
                  "brother", "father", "uncle", "grandfather"]
        feat_dict["implicit_male"] = word in mwords
        feat_dict["prev_male"] = any(t[0] in mwords for t in sent)

        # Match keywords for female entity
        fwords = ["woman", "women", "girl", "lady", "female",
                  "sister", "mother", "aunt", "grandmother"]
        feat_dict["implicit_female"] = word in fwords
        feat_dict["prev_female"] = any(t[0] in fwords for t in sent)

        return feat_dict

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X = [self.tagger.postag(sent) for sent in X]
        feats = [self.extract_pronoun_features(
            X, index, memory=self.memory, stemmer=self.stemmer)
            for index in range(len(X))]
        return feats


class PronounTaggerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang=None, stemmer=None, memory=2):
        super().__init__()
        self._coref_transformer = PronounTaggerTransformer(lang=lang, stemmer=stemmer, memory=memory)
        self._dict_vectorizer = DictVectorizer()

    def get_feature_names(self):
        return self._dict_vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        X = self._coref_transformer.transform(X)
        self._dict_vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = self._coref_transformer.transform(X, **transform_params)
        return self._dict_vectorizer.transform(X)


class SentenceWordFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []

        def is_numeric(input_str):
            try:
                float(input_str)
                return True
            except ValueError:
                return False

        for sent in X:
            toks = nltk.word_tokenize(sent)
            wfeat = {}
            for idx, w in enumerate(toks):
                wfeat.update({
                    f'word_{idx}': w,
                    f'is_first_{idx}': idx == 0,
                    f'is_last_{idx}': idx == len(toks) - 1,
                    f'is_capitalized_{idx}': w[0].upper() == w[0],
                    f'is_all_caps_{idx}': w.upper() == w,
                    f'is_all_lower_{idx}': w.lower() == w,
                    f'prefix-1_{idx}': w[0],
                    f'prefix-2_{idx}': w[:2],
                    f'prefix-3_{idx}': w[:3],
                    f'suffix-1_{idx}': w[-1],
                    f'suffix-2_{idx}': w[-2:],
                    f'suffix-3_{idx}': w[-3:],
                    f'prev_word_{idx}': '' if idx == 0 else toks[idx - 1],
                    f'next_word_{idx}': '' if idx == len(toks) - 1 else toks[idx + 1],
                    f'has_hyphen_{idx}': '-' in w,
                    f'is_numeric_{idx}': is_numeric(w),
                    f'capitals_inside_{idx}': w[1:].lower() != w[1:]
                })
            feats += [wfeat]
        return feats


class SentenceWordFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self._transformer = SentenceWordFeaturesTransformer()
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
