from sklearn.pipeline import FeatureUnion

from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier
from ovos_classifiers.skovos.features import ClassifierProbaVectorizer
from ovos_classifiers.skovos.features import KeywordFeaturesVectorizer, OCPKeywordFeaturesVectorizer

# models with _kw in the name expect raw features, not a text string
# this file illustrates how to use them


# TODO - replace paths
ents_csv_path = "datasets/ocp_entities_v0.csv"
clf_base = "models/ocp"


class Featurizer:
    # this is implementation specific, must match what was used at training time
    def __init__(self, base_clf, ents):
        self.feats2 = ClassifierProbaVectorizer(base_clf)
        self.feats = KeywordFeaturesVectorizer()
        self.feats.load_entities(ents)

    def transform(self, X):
        vec = FeatureUnion([
            ("kw", self.feats),
            ("clf", self.feats2)
        ])
        return vec.transform(X)


# load  featurizer
base_clf = SklearnOVOSClassifier.from_file(
    f"{clf_base}/binary_ocp_cv2_small.clf")
f = Featurizer(base_clf, ents_csv_path)

# transform utterances into features
utterances = ["play a song",
              "play my morning jams",
              "i want to watch the matrix",
              "tell me a joke", "who are you", "you suck"]
X = f.transform(utterances)

# load classifier
path = f"{clf_base}/binary_ocp_cv2_kw_medium.clf"
clf = SklearnOVOSClassifier.from_file(path)

# predict on features, NOT raw text
preds = clf.predict(X)
print(preds)
# ['OCP' 'OCP' 'OCP' 'other' 'other' 'other']


for preds in clf.predict_labels(X):
    print(preds)
    # {'OCP': 0.9987840201471372, 'other': 0.001215979852862784}
    # {'OCP': 0.9976782751502448, 'other': 0.0023217248497553367}
    # {'OCP': 0.9975936769349861, 'other': 0.0024063230650138508}
    # {'OCP': 0.0049925023790918345, 'other': 0.995007497620908}
    # {'OCP': 0.0023720677093612918, 'other': 0.9976279322906387}
    # {'OCP': 0.004691585390693631, 'other': 0.9953084146093063}
