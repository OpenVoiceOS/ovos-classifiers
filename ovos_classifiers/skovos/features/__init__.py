# feature extraction utils

import functools

import ahocorasick
import numpy as np
from nltk.util import skipgrams
from normality.transliteration import latinize_text
from ovos_config import Configuration
from ovos_utils.xdg_utils import xdg_data_home
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron

from ovos_classifiers.corefiob import OVOSCorefIOBTagger
from ovos_classifiers.datasets import get_ocp_entities_dataset
from ovos_classifiers.heuristics.lang_detect import LMLangClassifier
from ovos_classifiers.heuristics.tokenize import word_tokenize
from ovos_classifiers.postag import OVOSPostag
from ovos_classifiers.utils import extract_postag_features, \
    extract_word_features, normalize, get_stemmer, extract_single_word_features
from ovos_classifiers.utils import get_stopwords


class TokenizerTransformer(BaseEstimator, TransformerMixin):

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return [word_tokenize(x) for x in X]


class SnowballStemmerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lang="en"):
        super().__init__()
        self.stemmer = get_stemmer(lang)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        return normalize(X, stemmer=self.stemmer, **transform_params)


class SingleWordFeaturesTransformer(BaseEstimator, TransformerMixin):

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = [extract_single_word_features(w) for w in X]
        return feats


class SingleWordFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._transformer = SingleWordFeaturesTransformer()
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


class WordFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lang=None, stemmer=None, memory=2):
        super().__init__()
        lang = lang or Configuration().get("lang", "en-us")
        lang = lang.split("-")[0]
        self.stemmer = stemmer or get_stemmer(lang)
        self.memory = memory

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = [
            extract_word_features(X, index, stemmer=self.stemmer,
                                  memory=self.memory)
            for index in range(len(X))]
        return feats


class WordFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang=None, stemmer=None, memory=2):
        self.lang = lang
        self.memory = memory
        self.stemmer = stemmer
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


class LangFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.stopwords = {}
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
        self.clf = None

    def extract_features(self, sentence):
        tokens = word_tokenize(sentence)
        feats = {
            l + "_stopword_count": 0 for l in self.stopwords.keys()
        }
        for lang, swords in self.stopwords.items():
            feats[lang] = sum(1 for w in tokens if w in swords)
        for lang, score in self.clf.predict(sentence):
            feats[lang + "_score"] = score
        return feats

    def fit(self, *args, **kwargs):
        for l, lang in self.langs.items():
            self.stopwords[l] = get_stopwords(lang)
        self.clf = LMLangClassifier()
        return self

    def transform(self, X, **transform_params):
        feats = [self.extract_features(x) for x in X]
        return feats


class LangFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang=None, stemmer=None, memory=2):
        self.lang = lang
        self.memory = memory
        self.stemmer = stemmer
        self._transformer = LangFeaturesTransformer()
        self._vectorizer = DictVectorizer(sparse=False)
        super().__init__()

    def get_feature_names(self):
        return self._vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        self._transformer.fit()
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
        `index`   = the index of the token we want to extract utils for
        `history` = the previous predicted IOB tags
        """
        coreftagger = coreftagger or OVOSCorefIOBTagger("heuristic")
        feat_dict = extract_postag_features(tokens, index, stemmer=stemmer,
                                            memory=memory)
        coref_tags = coreftagger.iob_tag(tokens)

        word, ciob = coref_tags[index]

        # update with CorefIOB utils
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
        `index`   = the index of the token we want to extract utils for
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
            toks = word_tokenize(sent)
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


class SkipGramVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, n=2, k=2):
        skipper = functools.partial(skipgrams, n=n, k=k)
        self._vectorizer = CountVectorizer(analyzer=skipper)

    def get_feature_names(self):
        return self._vectorizer.get_feature_names_out()

    def fit(self, X, y=None, **kwargs):
        X = [word_tokenize(t) for t in X]
        self._vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = [word_tokenize(t) for t in X]
        return self._vectorizer.transform(X)


class SkipGramTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n=2, k=2):
        self._vectorizer = SkipGramVectorizer(n=n, k=k)

    def fit(self, *args, **kwargs):
        self._vectorizer.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        vectorized_text = self._vectorizer.transform(X, **transform_params)
        feats = dict(zip([tuple(a) for a in self._vectorizer.get_feature_names()],
                         vectorized_text.toarray().sum(axis=0)))
        return feats


class KeywordFeatures:
    def __init__(self, csv_path=None, ignore_list=None):
        ignore_list = ignore_list or []
        self.ignore_list = ignore_list
        self.bias = {}  # just for logging
        self.automatons = {}
        self._needs_building = []
        self.entities = {}
        if csv_path:
            self.load_entities(csv_path)

    def reset_automatons(self):
        # "untrain" the automatons
        self._needs_building = [name for name in self.automatons]
        self.automatons = {name: ahocorasick.Automaton() for name in self.automatons.keys()}
        for name, samples in self.entities.items():
            for s in samples:
                self.automatons[name].add_word(s.lower(), s)

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        if name not in self.entities:
            self.entities[name] = []
        self.entities[name] += samples
        if name not in self.bias:
            self.bias[name] = []
        self.bias[name] += samples

        if name not in self.automatons:
            self.automatons[name] = ahocorasick.Automaton()
        for s in samples:
            self.automatons[name].add_word(s.lower(), s)

        self._needs_building.append(name)

    def deregister_entity(self, name):
        """ register runtime entity samples,
            eg from skills"""
        if name in self.entities:
            self.entities.pop(name)
        if name in self.bias:
            self.bias.pop(name)
        if name in self.automatons:
            self.automatons.pop(name)
        if name in self._needs_building:
            self._needs_building.remove(name)

    def load_entities(self, csv_path):
        ents = {}
        if isinstance(csv_path, str):
            files = [csv_path]
        else:
            files = csv_path
        data = []
        for csv_path in files:
            with open(csv_path) as f:
                lines = f.read().split("\n")[1:]
                data += [l.split(",", 1) for l in lines if "," in l]

        for n, s in data:
            if n not in ents:
                ents[n] = []
            s = latinize_text(s)
            ents[n].append(s)
            self._needs_building.append(n)

        for k, samples in ents.items():
            self._needs_building.append(k)
            if k not in self.automatons:
                self.automatons[k] = ahocorasick.Automaton()
            for s in samples:
                self.automatons[k].add_word(s.lower(), s)
        self.entities.update(ents)
        return ents

    def match(self, utt):
        for k, automaton in self.automatons.items():
            if k in self._needs_building:
                automaton.make_automaton()

        self._needs_building = []

        utt = utt.lower().strip(".!?,;:")

        for k, automaton in self.automatons.items():
            # skip automatons without registered samples
            if not self.entities.get(k):
                continue

            for idx, v in automaton.iter(utt):
                if len(v) < 3:
                    continue

                if "_name" in k and v.lower() in self.ignore_list:
                    # LOG.debug(f"ignoring {k}:  {v}")
                    continue

                # filter partial words
                if " " not in v:
                    if v.lower() not in utt.split(" "):
                        continue
                if v.lower() + " " in utt or utt.endswith(v.lower()):
                    # if k in self.bias:
                    #    LOG.debug(f"BIAS {k} : {v}")
                    yield k, v

    def count(self, sentence):
        match = {k: 0 for k in self.entities.keys()}
        for k, v in self.match(sentence):
            if "_name" in k and v.lower() in self.ignore_list:
                continue
            match[k] += 1
            if v in self.bias.get(k, []):
                # LOG.debug(f"Feature Bias: {k} +1 because of: {v}")
                match[k] += 1
        return match

    def extract(self, sentence):
        match = {}
        for k, v in self.match(sentence):
            if k not in match:
                match[k] = v
            elif v in self.bias.get(k, []) or len(v) > len(match[k]):
                match[k] = v

        return match


class KeywordFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, csv_path=None, ignore_list=None, **kwargs):
        self.wordlist = KeywordFeatures(csv_path, ignore_list)
        super().__init__(**kwargs)

    @property
    def labels(self):
        return sorted(list(self.wordlist.entities.keys()))

    def ignore(self, samples):
        self.wordlist.ignore_list += samples

    def load_entities(self, csv_path):
        self.wordlist.load_entities(csv_path)

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        self.wordlist.register_entity(name, samples)

    def deregister_entity(self, name):
        """ register runtime entity samples,
            eg from skills"""
        self.wordlist.deregister_entity(name)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        if isinstance(X, str):
            X = [X]
        feats = []
        for sent in X:
            s_feature = self.wordlist.count(sent)
            feats += [s_feature]
        return feats


class KeywordFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, csv_path=None, ignore_list=None, **kwargs):
        super().__init__(**kwargs)
        self._transformer = KeywordFeaturesTransformer(csv_path, ignore_list, **kwargs)
        # NOTE: changing this list requires retraining the classifier
        self.labels_index = []

    @property
    def labels(self):
        return self._transformer.labels

    def ignore(self, samples):
        self._transformer.ignore(samples)

    def load_entities(self, csv_path):
        self._transformer.load_entities(csv_path)
        self.fit()

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        self._transformer.register_entity(name, samples)
        self.fit()

    def deregister_entity(self, name):
        """ register runtime entity samples,
            eg from skills"""
        self._transformer.deregister_entity(name)
        self.fit()

    def fit(self, *args, **kwargs):
        self.labels_index = sorted(self.labels)
        return self

    def transform(self, X, **transform_params):
        X2 = []
        for x in self._transformer.transform(X):
            feats = []
            for label in self.labels_index:
                if label in x:
                    feats.append(x[label])
                else:
                    feats.append(0)
            X2.append(feats)

        return np.array(X2)


class OCPKeywordFeaturesVectorizer(KeywordFeaturesVectorizer):
    def __init__(self, ignore_list=None, **kwargs):
        get_ocp_entities_dataset()  # ensure file exists
        csv_path = f"{xdg_data_home()}/OpenVoiceOS/datasets/ocp_entities_v0.csv"
        super().__init__(csv_path, ignore_list, **kwargs)


class ClassifierProbaVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, base_clf=None, prefit=True, **kwargs):
        super().__init__(**kwargs)
        if base_clf is None:
            prefit = False
            base_clf = Perceptron()
        self.clf = base_clf
        self.prefit = prefit

    def fit(self, *args, **kwargs):
        if not self.prefit:
            self.clf.fit(*args, **kwargs)
            self.prefit = True
        return self

    def transform(self, X, **transform_params):
        # provide a vector of probabilities per class
        return self.clf.clf.predict_proba(X)


if __name__ == '__main__':
    f = OCPKeywordFeaturesVectorizer()
    f.fit()
    t = f.transform(["play metallica", "play a horror movie", "watch netflix"])
    print(t)
    exit()
    s = SkipGramTransformer(2, 2)
    text = ['Insurgents killed in ongoing fighting.', "i love apple", "i love watermelon"]
    s.fit(text)

    vectorized_text = s.transform(text)

    print(vectorized_text)
    # {('Insurgents', 'in'): 1,
    #  ('Insurgents', 'killed'): 1,
    #  ('Insurgents', 'ongoing'): 1,
    #  ('fighting', '.'): 1,
    #  ('i', 'apple'): 1,
    #  ('i', 'love'): 2,
    #  ('i', 'watermelon'): 1,
    #  ('in', '.'): 1,
    #  ('in', 'fighting'): 1,
    #  ('in', 'ongoing'): 1,
    #  ('killed', 'fighting'): 1,
    #  ('killed', 'in'): 1,
    #  ('killed', 'ongoing'): 1,
    #  ('love', 'apple'): 1,
    #  ('love', 'watermelon'): 1,
    #  ('ongoing', '.'): 1,
    #  ('ongoing', 'fighting'): 1}

    csv = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/ocp_nlp/models/ocp_entities_v0.csv"
    l = KeywordFeatures()
    l.load_entities(csv)

    # efficient keyword matching, search many strings inside query string
    print(l.extract("play metallica"))
    # {'album_name': 'Metallica', 'artist_name': 'Metallica'}

    print(l.extract("play the beatles"))
    # {'album_name': 'The Beatles', 'series_name': 'The Beatles',
    # 'artist_name': 'The Beatles', 'movie_name': 'The Beatles'}

    print(l.extract("play rob zombie"))
    # {'artist_name': 'Rob Zombie', 'album_name': 'Zombie',
    # 'book_name': 'Zombie', 'game_name': 'Zombie', 'movie_name': 'Zombie'}

    print(l.extract("play horror movie"))
    # {'film_genre': 'Horror', 'cartoon_genre': 'Horror', 'anime_genre': 'Horror',
    # 'radio_drama_genre': 'horror', 'video_genre': 'horror',
    # 'book_genre': 'Horror', 'movie_name': 'Horror Movie'}

    print(l.extract("play science fiction"))
    #  {'film_genre': 'Science Fiction', 'cartoon_genre': 'Science Fiction',
    #  'podcast_genre': 'Fiction', 'anime_genre': 'Science Fiction',
    #  'documentary_genre': 'Science', 'book_genre': 'Science Fiction',
    #  'artist_name': 'Fiction', 'tv_channel': 'Science',
    #  'album_name': 'Science Fiction', 'short_film_name': 'Science',
    #  'book_name': 'Science Fiction', 'movie_name': 'Science Fiction'}

    v = KeywordFeaturesVectorizer()
    v.load_entities(csv)
    print(v.transform(["play my morning jams"]))
    # keyword count vector
    # [[0 3 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0
    #   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0]]

    # keyword counts
    v = KeywordFeaturesTransformer()
    v.load_entities(csv)
    print(v.transform(["play my morning jams"]))
    # [{'film_genre': 0, 'cartoon_genre': 0, 'news_streaming_service': 0,
    # 'media_type_documentary': 0, 'media_type_adult': 0, 'media_type_bw_movie': 0,
    # 'podcast_genre': 0, 'comic_streaming_service': 0, 'music_genre': 0,
    # 'media_type_video_episodes': 0, 'asmr_trigger': 0, 'anime_genre': 0,
    # 'media_type_audio': 0, 'media_type_bts': 0, 'media_type_silent_movie': 0,
    # 'audiobook_streaming_service': 0, 'radio_drama_genre': 0, 'media_type_podcast': 0,
    # 'hentai_streaming_service': 0, 'radio_theatre_company': 0, 'ad_keyword': 0,
    # 'media_type_short_film': 0, 'media_type_sound': 0, 'media_type_movie': 0,
    # 'sound_name': 0, 'news_provider': 0, 'music_streaming_service': 0,
    # 'documentary_genre': 0, 'radio_theatre_streaming_service': 0,
    # 'podcast_streaming_service': 0, 'media_type_tv': 0, 'comic_name': 0,
    # 'soundtrack_keyword': 0, 'media_type_adult_audio': 0, 'media_type_news': 0,
    # 'media_type_music': 0, 'media_type_cartoon': 0, 'play_verb_audio': 0,
    # 'documentary_streaming_service': 0, 'cartoon_streaming_service': 0,
    # 'anime_streaming_service': 0, 'media_type_hentai': 0, 'movie_streaming_service': 0,
    # 'media_type_trailer': 0, 'shorts_streaming_service': 0, 'video_genre': 0,
    # 'asmr_keyword': 0, 'porn_streaming_service': 0, 'playback_device': 0,
    # 'media_type_game': 0, 'playlist_name': 0, 'media_type_video': 0,
    # 'media_type_visual_story': 0, 'media_type_radio_theatre': 0, 'play_verb_video': 0,
    # 'media_type_audiobook': 0, 'porn_genre': 0, 'book_genre': 1, 'media_type_anime': 0,
    # 'media_type_radio': 0, 'album_name': 3, 'country_name': 0, 'movie_director': 0,
    # 'generic_streaming_service': 0, 'tv_streaming_service': 0, 'radio_drama_name': 0,
    # 'film_studio': 0, 'video_streaming_service': 0, 'short_film_name': 1, 'tv_channel': 0,
    # 'youtube_channel': 0, 'bw_movie_name': 0, 'audiobook_narrator': 0,
    # 'radio_program_name': 0, 'game_name': 0, 'series_name': 1, 'artist_name': 1,
    # 'tv_genre': 0, 'hentai_name': 0, 'podcast_name': 0, 'silent_movie_name': 0,
    # 'book_name': 1, 'gaming_console_name': 0, 'book_author': 0, 'record_label': 0,
    # 'radio_streaming_service': 0, 'podcaster': 0, 'game_genre': 0, 'anime_name': 0,
    # 'documentary_name': 0, 'movie_actor': 0, 'cartoon_name': 0, 'radio_drama_actor': 0,
    # 'audio_genre': 0, 'song_name': 0, 'movie_name': 2, 'porn_film_name': 0,
    # 'comics_genre': 0, 'radio_program': 0, 'pornstar_name': 0}]
