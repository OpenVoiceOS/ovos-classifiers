import json
from os import makedirs
from os.path import join, dirname

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from ovos_classifiers.datasets import get_utterance_tags_trainset
from ovos_classifiers.tasks.tagger import OVOSVotingClassifierTagger

TAGSET = "utttags"
CORPUS = "utterance_tags_v0.1"
NAME = "voting"
PIPELINE = "questions_en"
TASK = "clf"

MODEL_META = {
    "corpus": CORPUS,
    "corpus_homepage": "https://github.com/OpenVoiceOS/ovos-datasets",
    "model_id": f"sklearn-{CORPUS}-{TAGSET}-{NAME}-{PIPELINE}-{TASK}",
    "tagset": CORPUS,
    "lang": "en",
    "algo": "sklearn.ensemble.VotingClassifier",
    "voters": ["sklearn.linear_model.LogisticRegression",
               "sklearn.linear_model.Perceptron",
               "sklearn.naive_bayes.MultinomialNB",
               "sklearn.svm.LinearSVC"],
    "required_packages": ["scikit-learn"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "utttags")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

(X, y), (X_test, y_test) = get_utterance_tags_trainset()

# Training classifiers
clf1 = LinearSVC()
clf2 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2)
clf3 = Perceptron()
clf4 = MultinomialNB()

estimators = [clf1, clf2, clf3, clf4]
clf = OVOSVotingClassifierTagger(estimators, PIPELINE)

clf.train(X, y)

print('Training completed')

acc = clf.score(X_test, y_test)
MODEL_META["accuracy"] = acc
print("Accuracy:", acc)
# Accuracy:  0.9228

# save pickle
path = join(MODEL_BASE, MODEL_META["model_id"] + ".pkl")
clf.save(path)

meta_path = join(META, MODEL_META["model_id"] + ".json")
with open(meta_path, "w") as f:
    json.dump(MODEL_META, f, indent=4)
