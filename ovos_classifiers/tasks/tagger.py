import nltk
from nltk import AffixTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger, RegexpTagger

from ovos_classifiers.tasks.classifier import OVOSAbstractClassifier
from ovos_classifiers.heuristics.tokenize import word_tokenize


class OVOSAbstractClassifierTagger(OVOSAbstractClassifier):
    def __init__(self, pipeline_clf=None, pipeline_id="naive"):
        super().__init__(pipeline_clf=pipeline_clf, pipeline_id=pipeline_id)
        self._pipeline_clf = pipeline_clf
        self._retrainable = True

    def train(self, train_data, target_data):
        if not self._retrainable or self._pipeline_clf is None or not self.pipeline:
            raise RuntimeError("model is not retrainable")
        super().train(train_data, target_data)

    def score(self, X_test, y_test):
        return self.clf.score(X_test, y_test)

    def tag(self, sentence):
        if isinstance(sentence, str):
            words = word_tokenize(sentence)
        else:
            words = sentence
        return self.predict(words)


class OVOSNgramTagger(OVOSAbstractClassifierTagger):
    def __init__(self, regex_patterns=None, default_tag=None, pipeline_id="ngram"):
        self.patterns = regex_patterns
        self.default_tag = default_tag
        super().__init__(pipeline_id=pipeline_id, pipeline_clf=self.pipeline)

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        return cls().load_from_file(path)

    @property
    def pipeline(self):
        ngrams = [DefaultTagger, AffixTagger, UnigramTagger, BigramTagger, TrigramTagger]
        if self.patterns:
            ngrams.insert(2, RegexpTagger)
        return ngrams

    def get_ngram_tagger(self, tagged_data, clazzes=None):
        clazzes = clazzes or self.pipeline
        clf = None
        for idx, tagger in enumerate(clazzes):
            if tagger == DefaultTagger:
                if not self.default_tag:
                    continue
                clf = DefaultTagger(self.default_tag)
            elif tagger == RegexpTagger:
                clf = tagger(self.patterns, backoff=clf)
            else:
                clf = tagger(tagged_data, backoff=clf)
        return clf

    def train(self, tagged_data):
        self.clf = self.get_ngram_tagger(tagged_data)

    def predict(self, text):
        if isinstance(text, str):
            text = word_tokenize(text)
        return self.clf.tag(text)

    def score(self, tagged_data):
        return self.clf.evaluate(tagged_data)


class OVOSBrillTagger(OVOSNgramTagger):

    def __init__(self, templates=None, regex_patterns=None, default_tag=None, pipeline_id="brill"):
        self.templates = templates
        super().__init__(regex_patterns=regex_patterns,
                         default_tag=default_tag,
                         pipeline_id=pipeline_id)

    @property
    def pipeline(self):
        ngrams = [DefaultTagger, AffixTagger, UnigramTagger, BigramTagger, TrigramTagger]
        if self.patterns:
            ngrams.insert(2, RegexpTagger)
        if self.templates:
            ngrams.append(self.templates)
        else:
            ngrams.append(nltk.brill.fntbl37())
        return ngrams

    def train(self, tagged_data, max_rules=100, deterministic=True):
        taggers = self.pipeline[:-1]
        brill = self.pipeline[-1]
        tagger = self.get_ngram_tagger(tagged_data, taggers)
        trainer = nltk.BrillTaggerTrainer(tagger, brill,
                                          deterministic=True)
        self.clf = trainer.train(tagged_data, max_rules=100)
