import json
from os import makedirs
from os.path import join, dirname

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from ovos_classifiers.datasets import get_treebank_trainset
from ovos_classifiers.tasks.tagger import OVOSVotingClassifierTagger

TAGSET = "udep"
CORPUS = "treebank"
NAME = "voting_lsvc+etree+percep+nb"
TASK = "postag"

MODEL_META = {
    "corpus": "treebank",
    "corpus_homepage": "https://www.nltk.org/nltk_data",
    "model_id": f"sklearn-{CORPUS}-{TAGSET}-{NAME}-{TASK}",
    "tagset": "Universal Dependencies",
    "lang": "en",
    "algo": "sklearn.ensemble.VotingClassifier",
    "voters": ["sklearn.linear_model.LogisticRegression",
               "sklearn.linear_model.Perceptron",
               "sklearn.naive_bayes.MultinomialNB",
               "sklearn.svm.LinearSVC"],
    "required_packages": ["scikit-learn"]
}
META = join(dirname(dirname(dirname(dirname(__file__)))), "models", "metadata")
MODEL_BASE = join(dirname(dirname(dirname(dirname(__file__)))), "models", "postag")
makedirs(MODEL_BASE, exist_ok=True)
makedirs(META, exist_ok=True)

# initializing training and testing set
(X, y), (X_test, y_test) = get_treebank_trainset(udep=True)

X = X[:5000]
y = y[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]

# Training classifiers
clf1 = LinearSVC()
clf2 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2)
clf3 = Perceptron()
clf4 = MultinomialNB()

estimators = [clf1, clf2, clf3, clf4]
clf = OVOSVotingClassifierTagger(estimators)

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
